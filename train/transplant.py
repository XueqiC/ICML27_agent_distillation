"""T1 preregistered optimizer-state transplant; immutable inputs, native updates.

``block`` emits state-update-U (use this full state for ``compose``/``post``).
Composed directories have the same payload as state-update-U and are read by
load_update_state, independently of their directory name. Post exposure counters
use reference-plan coordinates; post.json also records actual token exposure.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, write_json
from tools.crn_tail import Experiment, add_state_arguments, restore, surrogate_score
from train.continue_from_state import continuation_plan, continue_training, filter_extra_states
from train.optimizer_state import bitwise_equal, learning_rate_at, load_update_state


def new_output(output, inputs=()):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory must not pre-exist")
    if any(Path(p).is_dir() and output.resolve().is_relative_to(Path(p).resolve()) for p in inputs):
        raise ValueError("Output must be outside immutable input directories")
    return output


def provenance_for(output, command, config, inputs, *, details=None):
    """Existing provenance schema plus complete directory content identities.

    Keep inputs file-only so Experiment can revalidate them with file_hash.
    Directory sha256 is digest({relative filename: file sha256}), including base
    weight bytes, not just the snapshot metadata used by the older tools.
    """
    files, directories = set(), {}
    for item in inputs:
        path = Path(item).absolute()
        if path.is_dir():
            members = sorted(p for p in path.rglob("*") if p.is_file())
            directories[str(path)] = members
            files.update(members)
        elif path.is_file():
            files.add(path)
        else:
            raise ValueError(f"Missing provenance input: {path}")
    provenance = write_provenance(output, command=command, config=config,
        inputs=sorted(files), details={"teacher_calls": 0, **(details or {})})
    provenance["details"]["directory_sha256"] = {
        root: digest({str(p.relative_to(root)): provenance["inputs"][str(p)] for p in members})
        for root, members in directories.items()}
    write_json(Path(output) / "provenance.json", provenance)
    return provenance


def block_plan(reference, original_plan, state, extra=(), *, updates, **options):
    examples, full, accounting = continuation_plan(reference, original_plan, state, extra, **options)
    cursor = state["data_cursor"]
    if type(updates) is not int or not 1 <= updates <= len(full) - cursor:
        raise ValueError("Block updates must fit within the remaining full plan")
    plan = deepcopy(full[:cursor + updates])
    plan[-1]["final"] = True
    plan[-1].setdefault("requested_endpoints", [])
    # Compare every field, including indices/pool and all realized extra draws;
    # only the end-of-block save marker is permitted to differ.
    for position, batch in enumerate(plan):
        expected = deepcopy(full[position])
        if position == len(plan) - 1:
            expected["final"] = True
            expected.setdefault("requested_endpoints", [])
        if batch != expected:
            raise AssertionError("Truncation changed the full continuation prefix")
    return examples, full, plan, {**accounting, "block_updates": updates,
        "block_end_update": len(plan), "block_supervised_tokens": plan[-1]["cumulative_supervised_tokens"],
        "block_replacement_updates": sum(b["pool"] == "extra" for b in plan[cursor:]),
        "full_plan_sha256": digest(full), "truncated_plan_sha256": digest(plan)}


def post_plan(reference, original_plan, original_state, state, *, target_tokens=416280, token_budget=512):
    examples, full, accounting = continuation_plan(reference, original_plan, original_state,
        fraction=0, target_tokens=target_tokens, token_budget=token_budget)
    u = state["updates"]
    if (type(u) is not int or state["data_cursor"] != u
            or not original_state["updates"] <= u < len(full)):
        raise ValueError("State updates/cursor must equal a valid reference prefix length")
    for key, default in (("learning_rate", None), ("lr_warmup_steps", 0),
                         ("gradient_clip", 1.0), ("moment_warmup_steps", 0)):
        if state.get(key, default) != original_state.get(key, default):
            raise ValueError(f"State differs from reference learning recipe: {key}")
    # Prefix is an unexecuted placeholder, not the alternative's training history.
    plan = deepcopy(full[:u]) + deepcopy(full[u:])
    if len(plan[:u]) != u or plan[u:] != full[u:]:
        raise AssertionError("Post updates differ from the fixed reference plan")
    reference_tokens = full[u - 1]["cumulative_supervised_tokens"] if u else 0
    post_tokens = sum(b["supervised_tokens"] for b in full[u:])
    return examples, plan, {**accounting, "resume_update": u, "end_update": len(full),
        "state_supervised_tokens": state["supervised_tokens"],
        "reference_supervised_tokens": reference_tokens, "post_supervised_tokens": post_tokens,
        "actual_final_supervised_tokens": state["supervised_tokens"] + post_tokens,
        "reference_final_supervised_tokens": full[-1]["cumulative_supervised_tokens"],
        "saved_counter_semantics": "reference plan coordinates; actual exposure reported separately",
        "reference_plan_sha256": digest(full), "post_plan_sha256": digest(full[u:])}


def _validate_states(states):
    import torch
    first = states[0]
    for state in states:
        for key in ("updates", "data_cursor", "step", "learning_rate", "lr_warmup_steps",
                    "scheduler", "gradient_clip", "moment_warmup_steps"):
            if key not in state or state[key] != first[key]:
                raise ValueError(f"Optimizer/training state settings differ: {key}")
        if state["data_cursor"] != state["updates"] or state["step"] != state["updates"] + state["moment_warmup_steps"]:
            raise ValueError("Optimizer update counters disagree")
        if list(state["trainable"]) != list(first["trainable"]):
            raise ValueError("Trainable parameter keys/order differ")
        optimizer = state["optimizer"]
        if optimizer["param_groups"] != first["optimizer"]["param_groups"]:
            raise ValueError("Optimizer parameter order/learning-rate settings differ")
        ids = [i for group in optimizer["param_groups"] for i in group["params"]]
        if (len(ids) != len(set(ids)) or len(ids) != len(state["trainable"])
                or set(ids) != set(optimizer["state"])
                or list(optimizer["state"]) != list(first["optimizer"]["state"])):
            raise ValueError("Optimizer parameter keys/order differ")
        for group in optimizer["param_groups"]:
            if group["lr"] != learning_rate_at(state["learning_rate"], state["lr_warmup_steps"], state["updates"]):
                raise ValueError("Optimizer learning rate differs from training state")
        for index, (name, parameter) in zip(ids, state["trainable"].items()):
            original = first["trainable"][name]
            if parameter.shape != original.shape or parameter.dtype != original.dtype:
                raise ValueError("Trainable parameter shape/dtype differs")
            entry, baseline = optimizer["state"][index], first["optimizer"]["state"][index]
            if list(entry) != list(baseline):
                raise ValueError("Optimizer parameter field keys/order differ")
            for key, value in entry.items():
                other = baseline[key]
                if isinstance(value, torch.Tensor):
                    if not isinstance(other, torch.Tensor) or value.shape != other.shape or value.dtype != other.dtype:
                        raise ValueError("Optimizer parameter shape/dtype differs")
                elif value != other:
                    raise ValueError("Non-tensor optimizer settings differ")
            step = entry.get("step")
            if (not isinstance(step, torch.Tensor) or step.numel() != 1
                    or float(step) != state["step"] or not bitwise_equal(step, baseline["step"])):
                raise ValueError("Optimizer step values differ")
            for key in ("exp_avg", "exp_avg_sq"):
                value = entry.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("Optimizer moment shape/dtype differs")


def compose(theta, m, v, output):
    """Copy theta's full checkpoint and replace only the two requested moments."""
    import torch
    paths = [Path(p).resolve() for p in (theta, m, v)]
    output = new_output(output, paths)
    states = [load_update_state(p) for p in paths]
    _validate_states(states)  # All compatibility checks precede any output writes.
    result = deepcopy(states[0])
    norms = {}
    for field, donor in (("exp_avg", states[1]), ("exp_avg_sq", states[2])):
        squared = 0.0
        for index, entry in result["optimizer"]["state"].items():
            value = donor["optimizer"]["state"][index][field]
            squared += float((value.double() - entry[field].double()).square().sum())
            entry[field] = value.clone()
        norms[field] = math.sqrt(squared)
    source_hashes = [digest(artifact_hashes(p)) for p in paths]
    record = {"sources": {name: {"path": str(path), "sha256": sha}
                          for name, path, sha in zip(("theta", "m", "v"), paths, source_hashes)},
              "m_difference_norm": norms["exp_avg"], "v_difference_norm": norms["exp_avg_sq"],
              "norm_dtype": "float64", "updates": result["updates"]}
    shutil.copytree(paths[0], output)
    torch.save(result, output / "optimizer.pt")
    # These are T1 audit files, never additional optimizer payload fields.
    write_json(output / "transplant.json", record)
    # Composing an already composed state retains its audit trail as a hashed
    # input; the new directory gets its own provenance rather than appending.
    (output / "provenance.json").unlink(missing_ok=True)
    provenance_for(output, "transplant compose", {"transplant": record}, paths)
    metadata = json.loads((paths[0] / "training_state.json").read_text())
    metadata["artifact_sha256"] = {k: h for k, h in artifact_hashes(output).items() if k != "training_state.json"}
    write_json(output / "training_state.json", metadata)
    load_update_state(output)  # Same integrity-checking loader used by Experiment.
    return record


def _training_output(experiment, args, examples, plan, accounting, inputs):
    output = new_output(args.output, [experiment.run, *inputs])
    recipe = {**experiment.recipe, "plan_sha256": digest(plan), "examples_sha256": digest(examples),
              "output": str(output), "supervised_tokens": plan[-1]["cumulative_supervised_tokens"],
              "transplant": accounting}
    config = {**experiment.provenance["config"], "sft": recipe, "transplant_tool": vars(args)}
    output.mkdir(parents=True, exist_ok=False)
    provenance = provenance_for(output, "transplant " + args.command, config,
        [*experiment.inputs, experiment.checkpoint, args.snapshot, args.verified, *inputs],
        details={"training_identity_sha256": digest(experiment.provenance),
                 "dataset_sha256": {k: digest(v) for k, v in experiment.datasets.items()}})
    write_json(output / "exposure_plan.json", plan, exclusive=True)
    write_json(output / "example_sources.json", [e.get("label_provenance", {}) for e in examples], exclusive=True)
    return output, provenance


def run_block(args):
    new_output(args.output)
    experiment = Experiment(args)
    extra = experiment.load_dataset(args.extra) if args.extra else []
    inputs = [args.extra] if args.extra else []
    selection = None
    if args.extra_states:
        extra, selection = filter_extra_states(extra, args.extra, args.extra_states)
        inputs.append(args.extra_states)
    examples, full, plan, accounting = block_plan(experiment.reference, experiment.plan, experiment.state, extra,
        updates=args.updates, target_tokens=args.target_tokens, fraction=args.fraction, seed=args.seed,
        token_budget=experiment.recipe["tokens_per_update"])
    if selection is not None:
        accounting["extra_states"] = selection
    output, provenance = _training_output(experiment, args, examples, plan, accounting, inputs)
    write_json(output / "full_exposure_plan.json", full, exclusive=True)
    accounting["full_plan_file_sha256"] = file_hash(output / "full_exposure_plan.json")
    write_json(output / "block.json", accounting, exclusive=True)
    report = continue_training(experiment.model, experiment.optimizer, experiment.state, examples, plan, output,
        identity_sha256=digest(provenance), tokenizer=experiment.tokenizer)
    return {**accounting, "state": str(output / f"state-update-{len(plan)}"), "checkpoints": report["checkpoints"]}


def run_post(args):
    new_output(args.output)
    reference_args = argparse.Namespace(**{**vars(args), "checkpoint": args.plan_from})
    experiment = Experiment(reference_args)
    state = load_update_state(args.checkpoint)
    examples, plan, accounting = post_plan(experiment.reference, experiment.plan, experiment.state, state,
        target_tokens=args.target_tokens, token_budget=experiment.recipe["tokens_per_update"])
    # Checks names/dtypes and restores exact theta/m/v/RNG before creating output.
    _validate_states([state])
    restore(experiment.model, experiment.optimizer, state)
    output, provenance = _training_output(experiment, args, examples, plan, accounting, [args.checkpoint])
    write_json(output / "post_plan.json", plan[state["updates"]:], exclusive=True)
    accounting["post_plan_file_sha256"] = file_hash(output / "post_plan.json")
    write_json(output / "post.json", accounting, exclusive=True)
    report = continue_training(experiment.model, experiment.optimizer, state, examples, plan, output,
        identity_sha256=digest(provenance), tokenizer=experiment.tokenizer)
    return {**accounting, "checkpoints": report["checkpoints"]}


def _score_anchor(checkpoint):
    """Find the original reference run for full states with mixed example banks."""
    path, visited, inputs = Path(checkpoint).resolve(), set(), []
    while True:
        if path in visited:
            raise ValueError("Cyclic checkpoint provenance")
        visited.add(path)
        transplant = path / "transplant.json"
        provenance = path.parent / "provenance.json"
        if transplant.is_file():
            inputs.extend([transplant, path / "provenance.json"])
            path = Path(json.loads(transplant.read_text())["sources"]["theta"]["path"])
        elif provenance.is_file():
            inputs.append(provenance)
            tool = json.loads(provenance.read_text())["config"].get("transplant_tool", {})
            if tool.get("command") not in {"block", "post"}:
                return path, inputs
            path = Path(tool.get("plan_from") or tool["checkpoint"]).resolve()
        else:
            raise ValueError("Full-state scoring requires its source run provenance")


def run_score(args):
    output = new_output(args.output, [args.checkpoint, args.snapshot, args.verified, args.surrogate])
    checkpoint = Path(args.checkpoint).resolve()
    metadata = json.loads((checkpoint / "training_state.json").read_text())
    inputs = [checkpoint, args.snapshot, args.verified, args.surrogate]
    if "data_cursor" in metadata:
        anchor, lineage = _score_anchor(checkpoint)
        inputs.extend(lineage)
        experiment = Experiment(argparse.Namespace(**{**vars(args), "checkpoint": str(anchor)}))
        restore(experiment.model, experiment.optimizer, load_update_state(checkpoint))
    else:
        experiment = Experiment(args)
    inputs.append(experiment.checkpoint)
    # Experiment.load_dataset handles both a surrogate listing and a dataset root.
    listing = Path(args.surrogate)
    if listing.is_file():
        inputs.append(listing.parent / json.loads(listing.read_text())["dataset_root"])
    new_output(output, [experiment.run, *inputs])
    score = surrogate_score(experiment.model, experiment.surrogate)
    score.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(artifact_hashes(checkpoint)))
    output.mkdir(parents=True, exist_ok=False)
    provenance_for(output, "transplant score", {**experiment.provenance["config"], "tool": vars(args)},
                   [*experiment.inputs, *inputs], details={"surrogate_sha256": digest(experiment.surrogate)})
    write_json(output / "score.json", score, exclusive=True)
    return score


def summarize_scores(scores):
    values = {arm: scores[arm]["F_tilde_geo_prob"] for arm in ("T00", "T00p", "T10", "T01", "T11", "N1", "A0")}
    if any(not math.isfinite(v) or not 0 < v <= 1 for v in values.values()):
        raise ValueError("Invalid geometric probability score")
    delta = abs(values["T00"] - values["T00p"])
    tau = max(3 * delta, 0.0012)
    M, M_m, M_v = (values[arm] - values["T00"] for arm in ("T11", "T10", "T01"))
    return {"metric": "F_tilde_geo_prob", "scores": values, "M": M, "M_m": M_m, "M_v": M_v,
            "I": M - M_m - M_v, "P": values["N1"] - values["T11"],
            "total_feedback_effect": values["N1"] - values["T00"], "delta_num": delta,
            "A0_minus_T00": values["A0"] - values["T00"], "tau": tau,
            "verdict": "uninterpretable" if delta > 0.0012 else "positive" if abs(M) > tau else "negative"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    block = sub.add_parser("block")
    add_state_arguments(block, surrogate=False)
    block.add_argument("--updates", type=int, required=True)
    block.add_argument("--extra")
    block.add_argument("--extra-states")
    block.add_argument("--fraction", type=float, default=0)
    block.add_argument("--seed", type=int, default=0)
    block.add_argument("--target-tokens", type=int, default=416280)
    composition = sub.add_parser("compose")
    for name in ("theta", "m", "v", "output"):
        composition.add_argument("--" + name, required=True)
    post = sub.add_parser("post")
    add_state_arguments(post, surrogate=False)
    post.add_argument("--plan-from", required=True)
    post.add_argument("--target-tokens", type=int, default=416280)
    score = sub.add_parser("score")
    add_state_arguments(score)
    args = parser.parse_args(argv)
    if args.command == "block" and args.extra_states and not args.extra:
        parser.error("--extra-states requires --extra")
    if args.command == "compose":
        result = compose(args.theta, args.m, args.v, args.output)
    else:
        result = {"block": run_block, "post": run_post, "score": run_score}[args.command](args)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
