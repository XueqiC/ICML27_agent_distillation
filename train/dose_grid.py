"""Prepare exact Seed-SFT exposure endpoints using the training data loader."""

from copy import deepcopy
import json
from pathlib import Path

from sed.seed_evidence import verified_collection_inputs, load_verified_collection
from sed.storage import digest, file_hash
from train.dataset import build_verified_dataset

PASSES = (3, 5, 10, 20, 30, 40)
TOKENS = (100_000, 200_000, 400_000)
REGISTERED_GRID = "configs/seedsft_dose_grid_v2.json"
PROTOCOL_FIELDS = ("rule", "pass_checkpoints", "token_checkpoints",
                   "merge_relative_to", "merge_fraction", "schema_version")


def dose_checkpoints(tokens_per_pass):
    if type(tokens_per_pass) is not int or tokens_per_pass <= 0:
        raise ValueError("Positive measured supervised tokens per pass required")
    checkpoints = [{"tokens": tokens_per_pass * p, "label": f"p{p:02d}", "passes": float(p)}
                   for p in PASSES]
    merged = {}
    for tokens in TOKENS:
        nearest = min(checkpoints[:len(PASSES)], key=lambda c: abs(c["tokens"] - tokens))
        # v1: distance measured relative to the requested token budget.
        if abs(nearest["tokens"] - tokens) * 100 <= 5 * tokens:
            merged[f"t{tokens // 1000}k"] = nearest["label"]
        else:
            checkpoints.append({"tokens": tokens, "label": f"t{tokens // 1000}k",
                                "passes": round(tokens / tokens_per_pass, 2)})
    return sorted(checkpoints, key=lambda c: c["tokens"]), merged


def prepare_grid(template, tokenizer, manifest, freeze, *, data_root="."):
    registered = json.loads((Path(__file__).resolve().parents[1] / REGISTERED_GRID).read_text())
    expected = {(r["K"], r["support_seed"]) for r in registered["runs"]}
    runs = template["runs"]
    pairs = {(r["K"], r["support_seed"]) for r in runs}
    partial = template.get("partial") is True
    if partial:
        if not runs or len(pairs) != len(runs) or not pairs <= expected:
            raise ValueError("Partial v2 requires a nonempty subset of registered runs without duplicates")
    elif len(runs) != len(expected) or pairs != expected:
        raise ValueError("v2 requires exactly K=1/2/50/100 x support seeds 0/1/2")
    if any(type(r["K"]) is not int or type(r["support_seed"]) is not int
           or r.get("name") != f"k{r['K']}s{r['support_seed']}" for r in runs):
        raise ValueError("v2 run names must be k{K}s{S} with integer K/support seed")
    if any(template.get(field) != registered[field] for field in PROTOCOL_FIELDS):
        raise ValueError("v2 dose protocol differs from the registered grid")
    result = deepcopy(template)
    if partial:
        result["partial_of"] = {"path": REGISTERED_GRID, "sha256": digest(registered)}
    for run in result["runs"]:
        root = Path(data_root) / run["verified"]
        collection, selection, _ = load_verified_collection(root, manifest, freeze)
        if (selection["K"], selection["seed"]) != (run["K"], run["support_seed"]):
            raise ValueError("Dose run K/support seed differs from its collection")
        examples = build_verified_dataset(root, tokenizer, manifest, freeze,
                                          max_length=32768, packing="episode")
        count = sum(e["supervised_tokens"] for e in examples)
        # Match the actual whitelist + native assistant labels, including closure.
        if count != sum(sum(label != -100 for label in e["labels"]) for e in examples):
            raise ValueError("Supervised-token accounting differs from training labels")
        checkpoints, merged = dose_checkpoints(count)
        run.update(supervised_tokens_per_pass=count, checkpoints=checkpoints,
                   merged_token_points=merged, total_tokens=max(c["tokens"] for c in checkpoints),
                   collection_sha256=digest(collection), examples_sha256=digest(examples),
                   inputs={str(p): file_hash(p) for p in verified_collection_inputs(root, manifest, freeze)})
    result.update(status="ready", template_sha256=digest(template),
                  manifest_sha256=digest(manifest), split_freeze_sha256=digest(freeze),
                  template_chat_sha256=digest(tokenizer.chat_template))
    return result


def load_ready_run(path, index):
    grid = json.loads(Path(path).read_text())
    if grid.get("schema_version") == 2 and grid.get("status") != "ready":
        raise ValueError("Prepare the v2 dose grid from collections on CPU before submission")
    run = grid["runs"][index]
    if (type(run.get("supervised_tokens_per_pass")) is not int
            or run["supervised_tokens_per_pass"] <= 0 or not run.get("checkpoints")
            or run["total_tokens"] != max(c["tokens"] for c in run["checkpoints"])):
        raise ValueError("Unresolved/invalid dose grid")
    if grid.get("schema_version") == 2:
        checkpoints, merged = dose_checkpoints(run["supervised_tokens_per_pass"])
        if run["checkpoints"] != checkpoints or run.get("merged_token_points") != merged:
            raise ValueError("Final dose grid differs from measured pass/token schedule")
    return run
