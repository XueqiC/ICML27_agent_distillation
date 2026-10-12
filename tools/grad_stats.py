"""R1 rankers and R2 gradient second moments at the restored student state.

sigma_s^2 = mean_states mean_student_labels ||grad[-log p(label | state)]||^2.
Use the full action/reply token SUM, without clipping or centering. Student
temperature-1 labels give an unbiased Monte Carlo estimate of the Fisher trace
at each state, in the trainable parameter space, when sampling and scoring use
the same student distribution. Greedy/teacher labels do not have this property.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time

from sed.storage import digest, file_hash, write_json
from tools.crn_tail import (Experiment, add_state_arguments, message_backward, require_no_dropout,
                            restore, surrogate_gradient, surrogate_score, trainables, validate_examples)
from train.sft import turn_loss_sum


def sampled_reply_example(tokenizer, messages, sample, *, max_length):
    """Score sampled text verbatim; a cap finish has no invented stop token.

    Unlike SFT encoding, do not strip whitespace or turn a parsed command into
    a new reply: that would score a different label than the student's draw.
    """
    from train.dataset import EMPTY_THOUGHT, TURN_TOKEN
    from tools.rollout_pool import PoolStudent
    metadata = sample.get("metadata", {})
    if (sample.get("status") != "completed"
            or any(metadata.get(k) != v for k, v in PoolStudent.label_sampling.items())
            or metadata.get("chat_template_kwargs") != {"enable_thinking": False}
            or metadata.get("thinking_chars") != 0):
        raise ValueError("Require completed temperature-1 student labels with thinking off")
    content = sample.get("raw_content")
    finish = metadata.get("finish_reason")
    if not isinstance(content, str) or finish not in {"stop", "length"}:
        raise ValueError("Sample lacks raw text or supported finish reason")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if not prompt.endswith(EMPTY_THOUGHT):
        raise ValueError("Gemma OFF generation prefix changed")
    # skip_special_tokens=False retains the native end token when served. When
    # the server omits its stop token, use the shared native Gemma turn end.
    if finish == "stop" and not content.endswith((TURN_TOKEN, "<eos>")):
        content += TURN_TOKEN
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(content, add_special_tokens=False)
    ids = tokenizer.encode(prompt + content, add_special_tokens=False)
    if not target_ids or ids != prompt_ids + target_ids or len(ids) > max_length:
        raise ValueError("Empty sample, tokenizer boundary mismatch, or context overflow; no truncation")
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + target_ids,
            "attention_mask": [1] * len(ids), "supervised_tokens": len(target_ids),
            "prompt_tokens": len(prompt_ids), "packing": "turn", "turn_count": 1}


def rollout_examples(records, tokenizer, *, max_length=32768, states_per_cluster=10, seed=0):
    """One masked example per (visible state, sampled label), selected before scoring.

    Callers loading files must first use read_rollouts for frozen-pool/evidence
    checks. Rebuild prompts from visible-field whitelists, never saved messages
    or verification fields. Failed labels cause an error, never survivor filtering.
    """
    from sed.react import build_messages
    from tools.r2_allocate import cluster_id
    if type(states_per_cluster) is not int or states_per_cluster < 1:
        raise ValueError("states_per_cluster must be positive")
    clusters, seen = defaultdict(list), set()
    for record in records:
        if record.get("sealed") or record.get("split") != "train":
            raise ValueError("Rollout labels require unsealed training states")
        for state in record["states"]:
            key = state["state_hash"]
            if key in seen:
                raise ValueError("Duplicate rollout state; use one rollout per game")
            seen.add(key)
            clusters[cluster_id(record["task_type"], state["step"])].append((record, state))
    result = {}
    for cid, states in sorted(clusters.items()):
        chosen = sorted(states, key=lambda pair: digest([seed, "gradient-probe", pair[1]["state_hash"]]))[:states_per_cluster]
        rows = []
        for record, state in chosen:
            samples = state.get("label_samples", [])
            if not samples or [s.get("sample_index") for s in samples] != list(range(len(samples))):
                raise ValueError("Selected state lacks complete --label-samples records")
            messages = build_messages(state["history"], {}, state["visible_context"])
            for sample in samples:
                row = sampled_reply_example(tokenizer, messages, sample, max_length=max_length)
                row.update(split="train", sealed=False, label_provenance={
                    "source": "student_temperature_1", "training_eligible": False,
                    "task_id": record["task_id"], "episode_id": record["episode_id"],
                    "state_hash": state["state_hash"], "turn_index": state["step"],
                    "sample_index": sample["sample_index"], "message_sha256": digest(messages),
                    "sample_sha256": digest(sample), "verification": "unknown"})
                rows.append(row)
        result[cid] = rows
    if not result:
        raise ValueError("No student-visited states")
    return result


def load_rollout_examples(experiment, args):
    from tools.rollout_pool import read_rollouts
    records = read_rollouts(args.rollouts, experiment.manifest, experiment.freeze, args.pool, subset="R2")
    # Fisher labels must come from the exact restored checkpoint, not merely
    # another adapter named p10. Original serving hashes survive portability.
    for record, directory in records:
        provenance = json.loads((directory / "provenance.json").read_text())
        config, spec = provenance["config"], provenance["details"]["policy_spec"]
        identity = config.get("lora_equivalence", {}).get("identity", {}).get("adapter", {})
        if (identity.get("training_state_sha256") != file_hash(experiment.checkpoint / "training_state.json")
                or identity.get("adapter_weights_sha256") != file_hash(experiment.checkpoint / "adapter_model.safetensors")):
            raise ValueError("Sampled student differs from restored checkpoint")
        for state in record["states"]:
            if len(state.get("label_samples", [])) != spec.get("label_samples"):
                raise ValueError("Sample count differs from collection provenance")
            for sample in state["label_samples"]:
                meta = sample.get("metadata", {})
                if sample["status"] == "completed" and (
                        meta.get("returned_model_id") != spec["model"]
                        or meta.get("student_revision") != spec["revision"]
                        or meta.get("max_tokens") != spec["max_tokens"]):
                    raise ValueError("Sampled student identity/cap differs from rollout")
    rows = rollout_examples([r for r, _ in records], experiment.tokenizer,
        max_length=experiment.recipe["max_length"], states_per_cluster=args.states_per_cluster, seed=args.seed)
    if experiment.surrogate is not None:
        experiment.check_surrogate_disjoint([r for group in rows.values() for r in group])
    return rows


def squared_norm(values):
    return sum(float(v.double().square().sum()) for v in values.values() if v is not None)


def dot(left, right):
    return sum(float(left[n].double().mul(v.double()).sum()) for n, v in right.items()
               if v is not None and left[n] is not None)


def gradient_and_step(model, optimizer, state, examples, message, *, datasets=None):
    """Actual native-dtype Adam step; all returned vectors stay in CPU memory."""
    restore(model, optimizer, state)
    raw, stats = message_backward(model, message, examples, datasets)
    consumed = {n: None if p.grad is None else p.grad.detach().cpu().clone()
                for n, p in trainables(model).items()}
    rate = state.get("learning_rate", optimizer.param_groups[0]["lr"])
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.step()
    delta = {n: p.detach().cpu().double() - state["trainable"][n].double()
             for n, p in trainables(model).items()}
    optimizer.zero_grad(set_to_none=True)
    stats["update_norm"] = math.sqrt(squared_norm(delta))
    return raw, consumed, delta, stats


def gradient_statistics(model, optimizer, state, examples, surrogate, *, candidates=None,
                        example_sets=None, datasets=None, sampled_sets=None):
    """Sets are messages or diagnostic sampled rows; R2 uses state/label means."""
    import torch
    from tools.crn_tail import resolve_message
    require_no_dropout(model)
    start = time.monotonic()
    restore(model, optimizer, state)
    try:
        if candidates and surrogate is None:
            raise ValueError("Candidate rankers require a surrogate")
        dF = surrogate_gradient(model, surrogate) if surrogate is not None else {}
        surrogate_norm = math.sqrt(squared_norm(dF)) if surrogate is not None else None
        candidate_results, moments = {}, {}
        for name, message in (candidates or {}).items():
            raw, consumed, delta, stats = gradient_and_step(model, optimizer, state, examples, message,
                                                           datasets=datasets)
            norms = {key: math.sqrt(squared_norm(g)) for key, g in (("unclipped", raw), ("clipped", consumed))}
            cosine = {key: dot(dF, g) / (surrogate_norm * norms[key]) if surrogate_norm * norms[key] else None
                      for key, g in (("unclipped", raw), ("clipped", consumed))}
            restore(model, optimizer, state)
            rows, _, _, _ = resolve_message(message, examples, datasets)
            candidate_results[name] = {"message": message, "stats": stats, "gradient_norm": norms,
                "cosine_loss_gradient_with_surrogate": cosine,
                "parameter_first_order_score": dot(dF, delta),
                "source_log_likelihood": surrogate_score(model, rows) if rows else None}
        sets = dict(sampled_sets or {})
        for name, message in (example_sets or {}).items():
            if name in sets:
                raise ValueError("Duplicate second-moment set")
            rows, multipliers, _, _ = resolve_message(message, examples, datasets)
            if not rows or any(c != 1 for c in multipliers):
                raise ValueError("Second-moment sets require nonempty, unscaled examples")
            sets[name] = rows
        for name, rows in sets.items():
            validate_examples(rows, training=False)
            entries = []
            restore(model, optimizer, state)
            model.train()  # Dropout is zero; keep gradient checkpointing available.
            for index, row in enumerate(rows):
                model.zero_grad(set_to_none=True)
                loss = turn_loss_sum(model, row, next(model.parameters()).device)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite sampled-action loss")
                loss.backward()  # Full -log p(action), NOT a token mean.
                gradients = {n: p.grad for n, p in trainables(model).items()}
                raw_squared = squared_norm(gradients)
                norm = torch.nn.utils.clip_grad_norm_(list(trainables(model).values()), 1.0,
                                                      error_if_nonfinite=True)
                entries.append({"position": index, "label_provenance": row.get("label_provenance", {}),
                    "supervised_tokens": row["supervised_tokens"],
                    "unclipped_squared_norm": raw_squared,
                    "clipped_squared_norm": squared_norm(gradients), "clip_active": bool(norm + 1e-6 > 1)})
            by_state = defaultdict(list)
            for entry in entries:
                key = entry["label_provenance"].get("state_hash", entry["position"])
                by_state[key].append(entry)
            def state_mean(field):
                return statistics.mean(statistics.mean(e[field] for e in labels) for labels in by_state.values())
            moments[name] = {"examples": len(rows), "per_example": entries,
                "sampled_states": len(by_state),
                "sigma_squared_unclipped": state_mean("unclipped_squared_norm"),
                "sigma_squared_clipped": state_mean("clipped_squared_norm")}
        result = {"schema_version": 1, "surrogate_gradient_norm": surrogate_norm,
                "candidates": candidate_results, "example_sets": moments,
                "clusters": {name: {"sigma2": value["sigma_squared_unclipped"],
                                     "sigma2_unclipped": value["sigma_squared_unclipped"],
                                     "sigma2_clipped_DIAGNOSTIC": value["sigma_squared_clipped"],
                                     "state_count": value["sampled_states"],
                                     "sample_count": value["examples"]} for name, value in moments.items()},
                "second_moment_definition": "mean_states mean_labels ||grad[-log p(sampled reply | state)]||^2; token sum, unclipped, uncentered",
                "fisher_interpretation": "unbiased trace estimate only for labels drawn from the same restored student distribution",
                "parameter_space": "requires_grad parameters only (LoRA coordinates for production)",
                "sampled_examples_sha256": digest(sampled_sets) if sampled_sets else None,
                "cosine_convention": "training loss gradient versus positive log-likelihood surrogate gradient",
                "teacher_calls": 0, "wall_seconds": time.monotonic() - start}
    finally:
        restore(model, optimizer, state)
    result["wall_seconds"] = time.monotonic() - start
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_state_arguments(parser, surrogate=False)
    parser.add_argument("--surrogate", help="Required for --spec candidate rankers; optional for rollout moments")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="JSON {candidates:{name:message}, example_sets:{name:message}}")
    source.add_argument("--rollouts", nargs="+", help="Audited rollout JSONL with --label-samples k > 0")
    parser.add_argument("--pool", default="data/manifest/r03_pool200.json")
    parser.add_argument("--states-per-cluster", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Frozen gradient-probe state selection seed")
    args = parser.parse_args(argv)
    if args.states_per_cluster < 1:
        parser.error("--states-per-cluster must be positive")
    spec = json.loads(Path(args.spec).read_text()) if args.spec else {}
    if spec.get("candidates") and not args.surrogate:
        parser.error("Candidate rankers require --surrogate")
    experiment = Experiment(args)
    experiment.load_messages([m for key in ("candidates", "example_sets") for m in spec.get(key, {}).values()])
    sampled = load_rollout_examples(experiment, args) if args.rollouts else None
    inputs = [args.spec] if args.spec else [*args.rollouts, args.pool]
    provenance = experiment.write_provenance(args.output, "grad_stats", extra_inputs=inputs)
    result = gradient_statistics(experiment.model, experiment.optimizer, experiment.state,
        experiment.reference, experiment.surrogate, candidates=spec.get("candidates"),
        example_sets=spec.get("example_sets"), datasets=experiment.datasets, sampled_sets=sampled)
    write_json(args.output, {**result, "provenance_sha256": digest(provenance),
                            "checkpoint": str(experiment.checkpoint)}, exclusive=True)


if __name__ == "__main__":
    main()
