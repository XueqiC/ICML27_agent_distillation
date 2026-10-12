"""Offline CRN paired replacements, shared state/gradient plumbing, and merge CLI.

See docs/R0R3_TOOLS.md for the frozen estimand, input schemas and limitations.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import time

from sed.storage import JsonlLedger, digest, file_hash, write_json
from train.optimizer_state import (adamw, assert_trainable_equal, load_checkpoint_state,
                                   load_update_state, restore_state)
from train.sft import exposure_plan, token_mean_backward, turn_loss_sum


def trainables(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def require_no_dropout(model):
    import torch
    if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p != 0
           for m in model.modules()):
        raise ValueError("CRN requires dropout == 0")
    configs = [getattr(model, "config", None)]
    if configs[0] is not None and hasattr(configs[0], "get_text_config"):
        configs.append(configs[0].get_text_config())
    for config in configs:
        for key in ("dropout", "attention_dropout", "hidden_dropout", "hidden_dropout_prob"):
            if config is not None and getattr(config, key, 0):
                raise ValueError("CRN requires dropout == 0")
    if any(getattr(c, "lora_dropout", 0) for c in getattr(model, "peft_config", {}).values()):
        raise ValueError("CRN requires dropout == 0")


def restore(model, optimizer, state):
    restore_state(model, optimizer, state)
    assert_trainable_equal(model, state["trainable"])


def validate_examples(examples, *, training=True):
    if not examples:
        raise ValueError("Empty example set")
    for row in examples:
        tokens = sum(x != -100 for x in row["labels"])
        if (tokens < 1 or tokens != row["supervised_tokens"]
                or len(row["input_ids"]) != len(row["labels"])):
            raise ValueError("Invalid supervised-token accounting")
        if row.get("sealed") or row.get("split") in {"test", "valid_unseen"}:
            raise ValueError("Sealed examples forbidden")
        if training and row.get("label_provenance", {}).get("training_eligible") is False:
            raise ValueError("Diagnostic/surrogate examples cannot enter training")


def batch_for(examples, indices):
    if not indices or any(type(i) is not int or not 0 <= i < len(examples) for i in indices):
        raise ValueError("Batch indices must address a nonempty example set")
    return {"indices": list(indices),
            "supervised_tokens": sum(examples[i]["supervised_tokens"] for i in indices)}


def select_examples(examples, ids):
    """Integers are frozen row positions; strings are unambiguous example/task IDs."""
    selected = []
    for key in ids:
        if type(key) is int:
            if not 0 <= key < len(examples):
                raise ValueError("Example index out of range")
            selected.append(examples[key])
        elif isinstance(key, str):
            matches = [e for e in examples if e.get("example_id", e.get("label_provenance", {}).get("task_id")) == key]
            if len(matches) != 1:
                raise ValueError("Example ID missing or ambiguous; use frozen row positions")
            selected.append(matches[0])
        else:
            raise ValueError("Example IDs must be integers or strings")
    validate_examples(selected)
    return selected


def resolve_message(message, reference, datasets=None):
    """Return ordered examples, per-example multipliers, and one clipping decision.

    A union is ONE token-weighted loss, clipped once after accumulation. Child
    multipliers weight their token sums, never their independently clipped grads.
    """
    datasets = datasets or {}
    if message == "zero" or message == {"zero": True}:
        return [], [], False, []
    if not isinstance(message, dict) or ("union" in message) == ("episodes" in message):
        raise ValueError("Message must be zero, episodes, or union")
    if set(message) - {"episodes", "union", "loss_multiplier", "clip"}:
        raise ValueError("Unknown message field")
    scale = message.get("loss_multiplier", 1.0)
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
        raise ValueError("loss_multiplier must be finite and nonnegative")
    clip = message.get("clip", True)
    if type(clip) is not bool:
        raise ValueError("clip must be boolean")
    if "union" in message:
        children = message["union"]
        if not isinstance(children, list) or len(children) != 2:
            raise ValueError("Union requires two messages")
        parts = [resolve_message(m, reference, datasets) for m in children]
        if any(not p[0] for p in parts):
            raise ValueError("Union requires two nonzero batches")
        if any(p[2] != clip for p in parts):
            raise ValueError("Union/child clipping flags must agree; clipping is applied once")
        rows = parts[0][0] + parts[1][0]
        if {id(e) for e in parts[0][0]} & {id(e) for e in parts[1][0]}:
            raise ValueError("Union batches must be disjoint")
        costs = [sum(e["supervised_tokens"] for e in p[0]) for p in parts]
        return rows, [scale * c for p in parts for c in p[1]], clip, [c / sum(costs) for c in costs]
    spec = message["episodes"]
    if isinstance(spec, dict):
        if set(spec) != {"root", "ids"} or spec["root"] not in datasets:
            raise ValueError("External episodes require a loaded root and ids")
        rows = select_examples(datasets[spec["root"]], spec["ids"])
    elif isinstance(spec, list):
        rows = select_examples(reference, spec)
    else:
        raise ValueError("episodes must be an ID list or {root, ids}")
    return rows, [float(scale)] * len(rows), clip, []


def message_backward(model, message, reference, datasets=None):
    """Native token accumulation; return CPU raw gradients and leave consumed grads."""
    import torch
    rows, multipliers, clip, weights = resolve_message(message, reference, datasets)
    params = trainables(model)
    model.train()
    model.zero_grad(set_to_none=True)
    total = sum(e["supervised_tokens"] for e in rows)
    loss_sum = 0.0
    if not rows:
        for p in params.values():
            p.grad = torch.zeros_like(p)
    elif all(c == 1 for c in multipliers):
        loss_sum, _ = token_mean_backward(model, rows, batch_for(rows, list(range(len(rows)))))
    else:
        device = next(model.parameters()).device
        for row, c in zip(rows, multipliers):
            loss = turn_loss_sum(model, row, device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite message loss")
            loss_sum += float(loss.detach()) * c
            (loss / total * c).backward()
    raw = {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in params.items()}
    # Native norm calculation, also when clipping is disabled (infinity leaves grads unchanged).
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0 if clip else math.inf,
                                         error_if_nonfinite=True)
    return raw, {"pre_clip_grad_norm": float(norm),
                 "clip_active": bool(clip and 1.0 / (norm + 1e-6) < 1), "clip": clip,
                 "supervised_tokens": total, "examples": len(rows),
                 "weighted_token_mean_loss": loss_sum / total if total else None,
                 "token_weights": weights}


def step_message(model, optimizer, message, reference, datasets=None, *, learning_rate=1e-5,
                 before=None):
    import torch
    params = trainables(model)
    if before is None:
        before = {n: p.detach().cpu().clone() for n, p in params.items()}
    raw, stats = message_backward(model, message, reference, datasets)
    del raw
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    squared = 0.0
    with torch.no_grad():
        for n, p in params.items():
            squared += float((p.detach().double() - before[n].to(p.device).double()).square().sum())
    stats["update_norm"] = math.sqrt(squared)
    return stats


def surrogate_score(model, examples):
    """Trajectory-mean geometric token probability plus log-likelihood diagnostics."""
    import torch
    validate_examples(examples, training=False)
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    by_trajectory = defaultdict(lambda: [0.0, 0])
    try:
        with torch.no_grad():
            for i, row in enumerate(examples):
                loss = float(turn_loss_sum(model, row, device))
                if not math.isfinite(loss):
                    raise ValueError("Nonfinite surrogate score")
                key = row.get("trajectory_id", str(i) if row.get("packing", "episode") == "episode"
                              else row.get("label_provenance", {}).get("task_id", str(i)))
                by_trajectory[key][0] -= loss
                by_trajectory[key][1] += row["supervised_tokens"]
    finally:
        model.train(training)
    total = sum(v[0] for v in by_trajectory.values())
    tokens = sum(v[1] for v in by_trajectory.values())
    probabilities = [math.exp(logp / count) for logp, count in by_trajectory.values()]
    if any(not 0 < p <= 1 for p in probabilities):
        raise ValueError("Geometric token probability underflow or outside (0, 1]")
    return {"F_tilde_per_token": total / tokens, "F_tilde_per_traj": total / len(by_trajectory),
            "F_tilde_geo_prob": statistics.mean(probabilities),
            "F_tilde_macro_token_mean": statistics.mean(v[0] / v[1] for v in by_trajectory.values()),
            "surrogate_tokens": tokens, "surrogate_trajectories": len(by_trajectory)}


def surrogate_gradient(model, examples):
    """Gradient of micro token-mean log likelihood, no optimizer updates."""
    import torch
    validate_examples(examples, training=False)
    model.zero_grad(set_to_none=True)
    # Dropout is zero; train mode retains gradient checkpointing for the 12B model.
    model.train()
    tokens = sum(e["supervised_tokens"] for e in examples)
    for row in examples:
        loss = turn_loss_sum(model, row, next(model.parameters()).device)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite surrogate loss")
        (-loss / tokens).backward()
    result = {n: (torch.zeros_like(p, device="cpu") if p.grad is None else p.grad.detach().cpu().clone())
              for n, p in trainables(model).items()}
    model.zero_grad(set_to_none=True)
    return result


def tail_plan(examples, updates, seed, *, token_budget=512, packing="episode"):
    if type(updates) is not int or updates < 0:
        raise ValueError("tail_updates must be a nonnegative integer")
    if not updates:
        return []
    # At least updates complete batches, without an endpoint-induced early flush.
    costs = [e["supervised_tokens"] for e in examples]
    plan = exposure_plan(costs, token_budget=token_budget, seed=seed,
                         supervised_tokens=(updates + 1) * (token_budget + max(costs)),
                         packing=packing, turn_counts=[e.get("turn_count", 1) for e in examples])
    return plan[:updates]


def synchronized_time(model):
    """Read wall time only after this model's CUDA work has finished."""
    import torch
    for device in sorted({p.device for p in model.parameters() if p.device.type == "cuda"}, key=str):
        torch.cuda.synchronize(device)
    return time.monotonic()


def run_item(model, optimizer, state, reference, surrogate, item, *, datasets=None,
             token_budget=512, packing="episode", learning_rate=1e-5):
    require_no_dropout(model)
    validate_examples(reference)
    if (type(item["replica"]) is not int or type(item["tail_seed"]) is not int
            or item["replica"] < 0 or item["tail_seed"] < 0):
        raise ValueError("replica and tail_seed must be nonnegative integers")
    start = synchronized_time(model)
    restore(model, optimizer, state)
    restored = synchronized_time(model)
    try:
        stats = step_message(model, optimizer, item["message"], reference, datasets,
                             learning_rate=learning_rate, before=state["trainable"])
        message_updated = synchronized_time(model)
        tail = tail_plan(reference, item["tail_updates"], item["tail_seed"],
                         token_budget=token_budget, packing=packing)
        import torch
        for batch in tail:
            optimizer.zero_grad(set_to_none=True)
            token_mean_backward(model, reference, batch)
            torch.nn.utils.clip_grad_norm_(list(trainables(model).values()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        tail_updated = synchronized_time(model)
        score = surrogate_score(model, surrogate)
        scored = synchronized_time(model)
        result = {"item_id": item["item_id"], "candidate_id": item.get("candidate_id", item["item_id"]),
                "replica": item["replica"], "tail_seed": item["tail_seed"],
                "tail_updates": item["tail_updates"], "item_sha256": digest(item),
                "tail_batches": [b["indices"] for b in tail], "tail_sha256": digest(tail),
                "tail_supervised_tokens": sum(b["supervised_tokens"] for b in tail),
                "message": stats, **score}
    finally:
        restore_start = synchronized_time(model)
        restore(model, optimizer, state)
        finished = synchronized_time(model)
    result["timing_seconds"] = {"restore": restored - start + finished - restore_start,
                              "message_update": message_updated - restored,
                              "tail_updates": tail_updated - message_updated,
                              "surrogate_scoring": scored - tail_updated}
    result["wall_seconds"] = finished - start
    return result


def build_items(pools, replicas, *, tail_updates=10, seed=0, batch_size=1, token_budget=None):
    """Common U[0,1) stream seeded by replica, independent tail-seed stream.

    pools: {candidate: {ids: [...], costs: [...] (for token_budget), root: optional}}.
    Sampling is WITH replacement. Different pool sizes use floor(u * pool_size).
    """
    if replicas < 2 or not pools or batch_size < 1 or (token_budget is not None and token_budget < 1):
        raise ValueError("Nonempty pools, >=2 replicas and positive batch budget required")
    tails = random.Random(seed)
    items = []
    for replica in range(replicas):
        omega = tails.getrandbits(63)
        for candidate, pool in pools.items():
            ids = pool["ids"]
            if not ids or len(set(ids)) != len(ids):
                raise ValueError("Candidate pool IDs must be nonempty and unique")
            costs = pool.get("costs")
            if token_budget is not None and (costs is None or len(costs) != len(ids)
                                             or any(type(c) is not int or c < 1 for c in costs)):
                raise ValueError("Token-budget sampling requires positive costs for every ID")
            rng, chosen, tokens = random.Random(replica), [], 0
            while (tokens < token_budget) if token_budget is not None else (len(chosen) < batch_size):
                index = int(rng.random() * len(ids))
                chosen.append(ids[index])
                tokens += costs[index] if costs else 0
            episodes = {"root": pool["root"], "ids": chosen} if "root" in pool else chosen
            items.append({"item_id": candidate, "replica": replica, "message": {"episodes": episodes},
                          "tail_updates": tail_updates, "tail_seed": omega})
    return {"schema_version": 1, "items": items, "sampling": {
        "rule": "iid U stream Random(replica); floor(U * pool_size), with replacement",
        "tail_seed_stream": seed, "batch_size": batch_size, "token_budget": token_budget},
        "pools_sha256": digest(pools)}


def bernstein_radius(variance, replicas, candidates, delta, B):
    if replicas < 2 or candidates < 2 or not 0 < delta < 1 or not math.isfinite(B) or B < 0:
        raise ValueError("Invalid Bernstein inputs")
    # In Definition 8.6 the total number of candidates is M+1.
    log = math.log(4 * (candidates - 1) ** 2 / delta)
    return math.sqrt(2 * variance * log / replicas) + 7 * B * log / (3 * (replicas - 1))


def merge_results(rows, *, B=None, delta=0.05, metric="F_tilde_geo_prob", unions=(), expected_items=None):
    bounded = metric == "F_tilde_geo_prob"
    if B is None:
        if not bounded:
            raise ValueError("Log-likelihood metrics require an explicit a-priori B")
        B = 2.0
    if not math.isfinite(B) or B < 0 or (bounded and B < 2):
        raise ValueError("Require finite nonnegative B, and B >= 2 for geometric probability")
    if not rows:
        raise ValueError("No results")
    if len({r.get("experiment_sha256") for r in rows}) != 1:
        raise ValueError("Cannot merge different experiment identities")
    if expected_items is not None:
        expected = {(i["item_id"], i["replica"]): digest(i) for i in expected_items}
        actual = {(r["item_id"], r["replica"]): r["item_sha256"] for r in rows}
        if len(expected) != len(expected_items) or len(actual) != len(rows) or expected != actual:
            raise ValueError("Results do not exactly cover the preregistered items")
    by_candidate = defaultdict(dict)
    for row in rows:
        candidate, replica = row.get("candidate_id", row["item_id"]), row["replica"]
        if replica in by_candidate[candidate] or not math.isfinite(row[metric]):
            raise ValueError("Duplicate candidate/replica or nonfinite score")
        if bounded and not 0 < row[metric] <= 1:
            raise ValueError("Geometric token probability must be in (0, 1]")
        by_candidate[candidate][replica] = row
    candidates = sorted(by_candidate)
    replicas = sorted(by_candidate[candidates[0]])
    if len(candidates) < 2 or len(replicas) < 2:
        raise ValueError("Need >=2 candidates and >=2 replicas")
    for group in by_candidate.values():
        if sorted(group) != replicas:
            raise ValueError("Incomplete paired replicas")
    for replica in replicas:
        if len({(group[replica]["tail_seed"], group[replica]["tail_updates"], group[replica]["tail_sha256"])
                for group in by_candidate.values()}) != 1:
            raise ValueError("CRN tails differ within replica")
    first_group = by_candidate[candidates[0]]
    if len({first_group[r]["tail_updates"] for r in replicas}) != 1:
        raise ValueError("Replicas must use the same fixed horizon")
    if len({first_group[r]["tail_seed"] for r in replicas}) != len(replicas):
        raise ValueError("Replicas must have fresh tail seeds")
    pairs = []
    for j in candidates:
        for k in candidates:
            if j == k:
                continue
            z = [by_candidate[j][r][metric] - by_candidate[k][r][metric] for r in replicas]
            variance, mean = statistics.variance(z), statistics.mean(z)
            observed = max(z) - min(z)
            if observed > B + 1e-12:
                raise ValueError("Observed paired range exceeds declared a-priori B")
            strict = bernstein_radius(variance, len(z), len(candidates), delta, B)
            diagnostic = bernstein_radius(variance, len(z), len(candidates), delta, observed)
            pairs.append({"j": j, "k": k, "Z": z, "S_hat": mean, "sample_variance": variance,
                          "radius_a_priori": strict, "observed_Z_range": observed,
                          "radius_observed_range_DIAGNOSTIC": diagnostic,
                          "surrogate_lower_bound": mean - strict})
    response = []
    for spec in unions:
        values = []
        for r in replicas:
            zero, a, b, ab = [by_candidate[spec[key]][r] for key in ("zero", "A", "B", "union")]
            weights = ab["message"]["token_weights"]
            if len(weights) != 2 or abs(sum(weights) - 1) > 1e-12:
                raise ValueError("Union result lacks token weights")
            da, db, dab = (v[metric] - zero[metric] for v in (a, b, ab))
            residual = dab - weights[0] * da - weights[1] * db
            values.append({"replica": r, "delta_A": da, "delta_B": db, "delta_union": dab,
                           "token_weights": weights, "residual": residual,
                           "nu_resp": abs(residual) / abs(dab) if dab else None})
        valid = [v["nu_resp"] for v in values if v["nu_resp"] is not None]
        mean_union = statistics.mean(v["delta_union"] for v in values)
        mean_residual = statistics.mean(v["residual"] for v in values)
        response.append({**spec, "replicas": values, "mean_delta_union": mean_union,
                         "mean_residual": mean_residual,
                         "nu_resp": abs(mean_residual) / abs(mean_union) if mean_union else None,
                         "median_nu_resp": statistics.median(valid) if valid else None,
                         "undefined_replicas": len(values) - len(valid)})
    return {"schema_version": 1, "metric": metric, "replicas": replicas, "candidates": candidates,
            "M_theorem": len(candidates) - 1, "delta": delta, "B": B,
            "B_definition": "a-priori interval LENGTH for paired Z, not F; F in [a,b] implies B=2*(b-a)",
            "range_status": ("a-priori F in (0,1], paired Z in [-1,1]; B=2 suffices" if bounded else
                             "caller-supplied bound; raw log likelihood has no automatic finite bound"),
            "task_certificate": "not asserted: surrogate alignment alpha/epsilon_sur uncalibrated",
            "pairs": pairs, "response_nonadditivity": response,
            "median_pair_nu_resp": statistics.median([v["nu_resp"] for v in response if v["nu_resp"] is not None])
                if any(v["nu_resp"] is not None for v in response) else None,
            "ranking": sorted(candidates, key=lambda j: -statistics.mean(by_candidate[j][r][metric] for r in replicas))}


def add_state_arguments(parser, *, surrogate=True):
    for name in ("snapshot", "checkpoint", "verified", "output"):
        parser.add_argument("--" + name, required=True)
    if surrogate:
        parser.add_argument("--surrogate", required=True)
    parser.add_argument("--device", default="cuda")


class Experiment:
    """Load the base ONCE; keep one immutable CPU snapshot across branches."""
    def __init__(self, args):
        from sed.manifest import load_frozen
        from train.dataset import load_tokenizer, mask_report
        from train.modeling import attach_lora, load_student, snapshot_metadata
        from train.sft import deterministic_seed
        self.args = args
        self.checkpoint = Path(args.checkpoint).resolve()
        self.run = self.checkpoint.parent
        self.provenance = json.loads((self.run / "provenance.json").read_text())
        self.recipe = recipe = self.provenance["config"]["sft"]
        if (recipe["dropout"] != 0 or recipe["optimizer"] != "AdamW"
                or tuple(recipe["betas"]) != (0.9, 0.999) or recipe["eps"] != 1e-8
                or recipe["weight_decay"] != 0 or recipe["learning_rate"] != 1e-5
                or recipe["gradient_clip"] != 1.0 or recipe.get("lr_warmup_steps", 0)):
            raise ValueError("R0-R3 requires the preregistered constant-LR/dropout-zero AdamW recipe")
        if snapshot_metadata(args.snapshot) != recipe["base_snapshot"]:
            raise ValueError("Base snapshot differs from training")
        self.inputs = [self.run / "provenance.json", self.run / "exposure_plan.json",
                       self.checkpoint / "training_state.json", self.checkpoint / "optimizer.pt"]
        for path, sha in self.provenance["inputs"].items():
            if file_hash(path) != sha:
                raise ValueError("Original training input changed")
            self.inputs.append(Path(path))
        deterministic_seed(recipe["seed"])
        self.tokenizer = load_tokenizer(args.snapshot)
        paths = self.provenance["config"]["paths"]
        self.manifest, self.freeze = load_frozen(paths["manifest"], paths["split_freeze"])
        self.datasets, self.dataset_purposes = {}, {}
        self.reference = self.load_dataset(args.verified)
        self.plan = json.loads((self.run / "exposure_plan.json").read_text())
        if digest(self.plan) != recipe["plan_sha256"] or digest(self.reference) != recipe["examples_sha256"]:
            raise ValueError("Reference examples or plan differ from frozen training")
        if digest(self.tokenizer.chat_template) != recipe["template_sha256"]:
            raise ValueError("Tokenizer template differs")
        mask_report(self.reference, self.tokenizer)
        self.surrogate = self.load_dataset(args.surrogate, purpose="surrogate_only") if getattr(args, "surrogate", None) else None
        self.model = attach_lora(load_student(args.snapshot, device=args.device, dtype=recipe["dtype"]),
                                 rank=recipe["rank"], alpha=recipe["alpha"], dropout=0)
        require_no_dropout(self.model)
        self.optimizer = adamw(trainables(self.model).values(), recipe["learning_rate"])
        if self.checkpoint.name.startswith("state-update-"):
            self.state = load_update_state(self.checkpoint, plan=self.plan, identity_sha256=digest(self.provenance))
        else:
            self.state = load_checkpoint_state(self.model, self.optimizer, self.checkpoint,
                                               plan=self.plan, provenance=self.provenance)
        restore(self.model, self.optimizer, self.state)
        if self.surrogate is not None:
            self.check_surrogate_disjoint(self.reference)

    def load_dataset(self, root, *, purpose="training"):
        from train.dataset import build_verified_dataset
        from sed.seed_evidence import verified_collection_inputs
        key = str(root)
        if key in self.datasets and self.dataset_purposes[key] != purpose:
            raise ValueError("A dataset cannot be both a surrogate and training source")
        if key not in self.datasets:
            path = Path(root)
            options = {"max_length": self.recipe["max_length"], "packing": self.recipe["packing"]}
            if path.is_file():
                if purpose != "surrogate_only":
                    raise ValueError("A listing file is only accepted for the surrogate")
                from data.manifest.build_r03_sets import load_surrogate_examples
                rows = load_surrogate_examples(path, self.tokenizer, self.manifest, self.freeze, **options)
                listing = json.loads(path.read_text())
                self.inputs.append(path)
                path = path.parent / listing["dataset_root"]
                collection = json.loads((path / "manifest.json").read_text())
            else:
                collection = json.loads((path / "manifest.json").read_text())
                if collection.get("format") == "r03_episode_dataset_v1":
                    from data.manifest.build_r03_sets import load_r03_dataset
                    rows = load_r03_dataset(path, self.tokenizer, self.manifest, self.freeze,
                                            purpose=purpose, **options)
                else:
                    rows = build_verified_dataset(path, self.tokenizer, self.manifest, self.freeze, **options)
                    self.inputs.extend(verified_collection_inputs(path, self.manifest, self.freeze))
            if collection.get("format") == "r03_episode_dataset_v1":
                self.inputs.extend([path / "manifest.json", *[path / e["episode_file"] for e in collection["episodes"]]])
                if (path / "provenance.json").is_file():
                    self.inputs.append(path / "provenance.json")
            validate_examples(rows, training=purpose == "training")
            self.datasets[key] = rows
            self.dataset_purposes[key] = purpose
        return self.datasets[key]

    def check_surrogate_disjoint(self, examples):
        from sed.manifest import resolve_task
        def groups(rows):
            return {resolve_task(self.manifest, self.freeze, e["label_provenance"]["task_id"])["group_id"]
                    for e in rows}
        if groups(examples) & groups(self.surrogate):
            raise ValueError("Training/candidate pool overlaps surrogate groups")

    def load_messages(self, messages):
        for message in messages:
            if isinstance(message, dict):
                if "union" in message:
                    self.load_messages(message["union"])
                elif isinstance(message.get("episodes"), dict):
                    rows = self.load_dataset(message["episodes"]["root"])
                    if self.surrogate is not None:
                        self.check_surrogate_disjoint(rows)
            resolve_message(message, self.reference, self.datasets)

    def write_provenance(self, output, command, *, extra_inputs=(), config=None):
        from sed.provenance import write_provenance
        output = Path(output)
        if output.exists() or output.resolve().is_relative_to(self.run):
            raise ValueError("Use a new output outside the immutable source run")
        output.parent.mkdir(parents=True, exist_ok=True)
        directory = output.parent / (output.name + ".provenance")
        directory.mkdir(exist_ok=False)
        return write_provenance(directory, command=command,
            config=config or {**self.provenance["config"], "tool": vars(self.args)},
            inputs=list(dict.fromkeys([*self.inputs, *extra_inputs])),
            details={"training_identity_sha256": digest(self.provenance), "teacher_calls": 0,
                     "checkpoint": str(self.checkpoint),
                     "dataset_sha256": {k: digest(v) for k, v in self.datasets.items()}})


def read_items(path):
    value = json.loads(Path(path).read_text())
    items = value["items"] if isinstance(value, dict) else value
    keys = [(i["item_id"], i["replica"]) for i in items]
    if not items or len(set(keys)) != len(keys):
        raise ValueError("Items must have unique (item_id, replica) keys")
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-items")
    for name in ("pools", "output"):
        build.add_argument("--" + name, required=True)
    build.add_argument("--replicas", type=int, required=True)
    build.add_argument("--tail-updates", type=int, default=10)
    build.add_argument("--seed", type=int, default=0)
    size = build.add_mutually_exclusive_group()
    size.add_argument("--batch-size", type=int, default=1)
    size.add_argument("--tokens-per-update", type=int)
    run = sub.add_parser("run")
    add_state_arguments(run)
    run.add_argument("--items", required=True)
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    merge = sub.add_parser("merge")
    merge.add_argument("--results", nargs="+", required=True)
    merge.add_argument("--items", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--B", type=float, help="A-priori interval LENGTH for paired Z; default 2 for geo_prob, required for log metrics")
    merge.add_argument("--delta", type=float, default=0.05)
    merge.add_argument("--metric", choices=["F_tilde_geo_prob", "F_tilde_per_token", "F_tilde_per_traj",
                                           "F_tilde_macro_token_mean"], default="F_tilde_geo_prob")
    merge.add_argument("--unions", help="JSON list of {zero,A,B,union} candidate IDs")
    args = parser.parse_args(argv)
    if args.command == "build-items":
        value = build_items(json.loads(Path(args.pools).read_text()), args.replicas,
            tail_updates=args.tail_updates, seed=args.seed, batch_size=args.batch_size,
            token_budget=args.tokens_per_update)
        write_json(args.output, value, exclusive=True)
    elif args.command == "merge":
        rows = [json.loads(line) for p in args.results for line in Path(p).read_text().splitlines()]
        summary = merge_results(rows, B=args.B, delta=args.delta, metric=args.metric,
            unions=json.loads(Path(args.unions).read_text()) if args.unions else (),
            expected_items=read_items(args.items))
        summary["input_sha256"] = {p: file_hash(p) for p in [*args.results, args.items, *([args.unions] if args.unions else [])]}
        write_json(args.output, summary, exclusive=True)
    else:
        if not 0 <= args.shard < args.num_shards:
            raise ValueError("Require 0 <= shard < num_shards")
        items = read_items(args.items)
        experiment = Experiment(args)
        experiment.load_messages([i["message"] for i in items])
        provenance = experiment.write_provenance(args.output, "crn_tail", extra_inputs=[args.items])
        identity = digest({"checkpoint": file_hash(experiment.checkpoint / "training_state.json"),
                           "items": file_hash(args.items), "datasets": provenance["details"]["dataset_sha256"],
                           "code": provenance["code_sha256"], "dependencies": provenance["dependencies"]})
        ledger = JsonlLedger(args.output)
        # Even a shard assigned zero items produces a mergeable empty file.
        Path(args.output).touch(exist_ok=False)
        for position, item in enumerate(items):
            if position % args.num_shards != args.shard:
                continue
            result = run_item(experiment.model, experiment.optimizer, experiment.state,
                experiment.reference, experiment.surrogate, item, datasets=experiment.datasets,
                token_budget=experiment.recipe["tokens_per_update"], packing=experiment.recipe["packing"])
            ledger.append({**result, "experiment_sha256": identity, "provenance_sha256": digest(provenance),
                           "shard": args.shard, "num_shards": args.num_shards})


if __name__ == "__main__":
    main()
