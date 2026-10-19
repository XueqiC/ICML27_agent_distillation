"""Exact OFF-prefix branches over one native history, without dense GPU masks.

Only token IDs, label spans and explicit template boundaries enter this module.
No environment/event dictionaries are model inputs. Flex BlockMask metadata is
built at block resolution (128 tokens), never by constructing an L x L mask.
"""
from baselines.common.conversation import sequence_rows

BLOCK_SIZE = 128
CPU_DENSE_LIMIT = 4096


def pack_shared_history(row):
    """Interleave history deltas and isolated generation copies, without truncation.

    The final history-form reply has no consumers and is omitted. The original
    32768 limit applies to each evaluation context, not this branched layout.
    """
    ids, labels, positions, branches, spans = [], [], [], [], []
    history = []
    previous_completed = []
    for branch, turn in enumerate(sequence_rows(row), 1):
        source, target = turn["input_ids"], turn["labels"]
        start = turn.get("generation_start")
        prompt = turn.get("prompt_tokens")
        if (not isinstance(start, int) or not isinstance(prompt, int)
                or not 0 < start < prompt < len(source) <= 32768
                or len(source) != len(target)
                or target != [-100] * prompt + source[prompt:]):
            raise ValueError("Shared history requires validated exact-prefix turn rows")
        prefix = source[:start]
        if prefix[:len(history)] != history or prefix[:len(previous_completed)] != previous_completed:
            raise ValueError("Native history is not a consistent extension of previous turns")
        delta = prefix[len(history):]
        ids.extend(delta)
        labels.extend([-100] * len(delta))
        positions.extend(range(len(history), len(prefix)))
        branches.extend([0] * len(delta))
        offset = len(ids)
        ids.extend(source[start:])
        labels.extend(target[start:])
        positions.extend(range(start, len(source)))
        branches.extend([branch] * (len(source) - start))
        spans.append({"start": offset + prompt - start, "stop": len(ids)})
        history = prefix
        previous_completed = prefix + source[prompt:]
    if not spans:
        raise ValueError("No supervised turns")
    return {"input_ids": ids, "labels": labels, "position_ids": positions,
            "branches": branches, "assistant_spans": spans}


def mask_function(positions, branches, lengths, window=None):
    """Window distances are evaluation positions, never packed token offsets."""
    length = positions.shape[1]

    def allowed(b, h, q, k):
        # Flex evaluates partial edge blocks beyond sequence length as well.
        qi, ki = q.clamp(max=length - 1), k.clamp(max=length - 1)
        qp, kp = positions[b, qi], positions[b, ki]
        qb, kb = branches[b, qi], branches[b, ki]
        valid = (q < lengths[b]) & (k < lengths[b])
        visible = (q >= k) & (qp >= kp) & ((kb == 0) | (qb == kb))
        if window is not None:
            visible = visible & (qp - kp < window)
        # Unsupervised padding rows see themselves, avoiding all-masked softmax.
        return (valid & visible) | ((q >= lengths[b]) & (q == k) & (q < length))

    return allowed


def _block_metadata(rows, length, window):
    """Conservative nonempty blocks and provably full blocks; O((L/128)^2).

    Conservative blocks are filtered by the exact token predicate in the kernel.
    Full blocks bypass it only when every token pair is known to be visible.
    """
    import torch
    size = BLOCK_SIZE
    blocks = (length + size - 1) // size
    summaries = []
    for row in rows:
        pos, branch = row["position_ids"], row["branches"]
        n = len(pos)
        summaries.append([
            (min(pos[i:i+size]), max(pos[i:i+size]), min(branch[i:i+size]),
             max(branch[i:i+size]), i + size <= n) if i < n else (0, 0, -1, -1, False)
            for i in range(0, blocks * size, size)])
    stats = torch.tensor(summaries)
    pmin, pmax, bmin, bmax, complete = stats.unbind(-1)
    qi = torch.arange(blocks)[None, :, None]
    ki = torch.arange(blocks)[None, None, :]
    history_key = bmin[:, None, :] == 0
    overlap = (bmin[:, :, None] <= bmax[:, None, :]) & (bmax[:, :, None] >= bmin[:, None, :])
    possible = (qi >= ki) & (pmax[:, :, None] >= pmin[:, None, :]) & (history_key | overlap)
    homogeneous = (bmin == bmax)
    same_branch = homogeneous[:, :, None] & homogeneous[:, None, :] & (bmin[:, :, None] == bmin[:, None, :])
    full = ((qi > ki) & (pmin[:, :, None] >= pmax[:, None, :])
            & ((bmax[:, None, :] == 0) | same_branch)
            & complete[:, :, None].bool() & complete[:, None, :].bool())
    if window is not None:
        possible &= pmin[:, :, None] - pmax[:, None, :] < window
        full &= pmax[:, :, None] - pmin[:, None, :] < window
    # Include padding diagonals and any mixed real/padded diagonal block.
    possible |= qi == ki
    return (possible & ~full).unsqueeze(1), full.unsqueeze(1)


def attention_masks(rows, config, device, *, backend):
    import torch
    length = max(len(row["input_ids"]) for row in rows)
    positions = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    branches = torch.full_like(positions, -1)
    lengths = torch.tensor([len(row["input_ids"]) for row in rows], device=device)
    for i, row in enumerate(rows):
        positions[i, :len(row["input_ids"])] = torch.tensor(row["position_ids"], device=device)
        branches[i, :len(row["input_ids"])] = torch.tensor(row["branches"], device=device)
    masks = {}
    if set(config.layer_types) - {"full_attention", "sliding_attention"}:
        raise ValueError("Unsupported Gemma layer type")
    for kind in set(config.layer_types):
        window = config.sliding_window if kind == "sliding_attention" else None
        if window is not None and window <= 0:
            raise ValueError("Positive sliding window required")
        predicate = mask_function(positions, branches, lengths, window)
        if backend == "flex_attention":
            from torch.nn.attention.flex_attention import BlockMask
            partial, full = _block_metadata(rows, length, window)

            def sparse(value):
                counts = value.sum(-1).to(device=device, dtype=torch.int32)
                indices = torch.argsort(value.to(torch.int8), dim=-1, descending=True, stable=True)
                return counts, indices.to(device=device, dtype=torch.int32)

            counts, indices = sparse(partial)
            full_counts, full_indices = sparse(full)
            masks[kind] = BlockMask.from_kv_blocks(counts, indices, full_counts, full_indices,
                BLOCK_SIZE=BLOCK_SIZE, mask_mod=predicate, seq_lengths=(length, length))
        else:
            if torch.device(device).type != "cpu" or length > CPU_DENSE_LIMIT:
                raise ValueError("Shared history requires flex_attention; dense masks are CPU-test-only (<=4096)")
            b = torch.arange(len(rows), device=device)[:, None, None, None]
            q = torch.arange(length, device=device)[None, None, :, None]
            k = torch.arange(length, device=device)[None, None, None, :]
            mask = predicate(b, 0, q, k)
            if backend == "eager":
                mask = torch.zeros(mask.shape, device=device).masked_fill(~mask, float("-inf"))
            elif backend != "sdpa":
                raise ValueError("Unsupported shared-history attention backend")
            masks[kind] = mask
    return positions, masks


def validate_model(model):
    """Fail closed on stochastic/sequence-dependent features that break sharing."""
    import torch
    config = model.config.get_text_config()
    if config.model_type != "gemma4_unified_text" or getattr(config, "use_bidirectional_attention", None) == "all":
        raise ValueError("Shared history requires causal Gemma4 unified text attention")
    if config.attention_dropout or any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
        raise ValueError("Exact shared history requires zero dropout")
    # Length-dependent RoPE scaling may depend on the longest branch in a call.
    for params in (getattr(config, "rope_parameters", {}) or {}).values():
        if isinstance(params, dict) and params.get("rope_type", "default") not in {"default", "proportional"}:
            raise ValueError("Shared history does not support length-dependent RoPE")
    return config


def shared_history_logps(model, rows, device, *, per_turn=False):
    """One packed forward per side (rows may also be batched for diagnostics)."""
    import torch
    from baselines.eto.official_loss import OfficialLoss
    config = validate_model(model)
    packed = [pack_shared_history(row) for row in rows]
    length = max(len(row["input_ids"]) for row in packed)
    ids = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(packed):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], device=device)
        labels[i, :n] = torch.tensor(row["labels"], device=device)
    position_ids, masks = attention_masks(packed, config, device, backend=config._attn_implementation)
    keep = torch.nonzero((labels[:, 1:] != -100).any(dim=0)).flatten()
    logits = model(input_ids=ids, attention_mask=masks, position_ids=position_ids,
                   use_cache=False, logits_to_keep=keep).logits
    target = labels[:, keep + 1]
    if per_turn:
        return [torch.stack([OfficialLoss.get_batch_logps(
            logits[i:i+1, (keep + 1 >= span["start"]) & (keep + 1 < span["stop"])].float(),
            target[i:i+1, (keep + 1 >= span["start"]) & (keep + 1 < span["stop"])].clone(),
            average_log_prob=False, is_encoder_decoder=True)[0]
            for span in row["assistant_spans"]]) for i, row in enumerate(packed)]
    return sum(OfficialLoss.get_batch_logps(logits[:, start:start+128].float(),
               target[:, start:start+128].clone(), average_log_prob=False, is_encoder_decoder=True)
               for start in range(0, len(keep), 128))
