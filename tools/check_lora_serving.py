"""Compare student teacher-forced reply logprobs; never query the teacher.

Default: GPU HF+PEFT (bf16/eager), then the existing vLLM server. --hf-only
and --hf-reference split the check so a single GPU need not hold both models.
HF base scores disable the adapter on the same model to measure kernel noise.
Only verified TRAIN turns are eligible. No environment reset or updates occur.
"""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from harness.lora_adapters import (adapter_identity, inspect_adapter, parse_adapters,
                                   validate_served_adapter)
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.storage import digest, file_hash, write_json
from train.modeling import snapshot_metadata

PROMPT_COUNT = 8
SCHEMA_VERSION = 5
MEAN_TOLERANCE = 0.05
BASE_NOISE_MULTIPLIER = 1.5
MIN_EFFECT_NOISE_MULTIPLIER = 2.0
OUTLIER_COUNT_MULTIPLIER = 2
OUTLIER_COUNT_ALLOWANCE = 2
OUTLIER_TOKEN_FRACTION = 0.05


def fixed_prompts(examples):
    """Frozen rule: first eight (task_id, turn_index) TRAIN turns, no selection by loss."""
    rows = sorted(examples, key=lambda e: (e["label_provenance"]["task_id"],
                                          e["label_provenance"]["turn_index"]))
    if len(rows) < PROMPT_COUNT:
        raise ValueError("At least eight verified training turns are required")
    result = []
    for row in rows[:PROMPT_COUNT]:
        source = row["label_provenance"]
        if source.get("training_eligible") is not True or row["packing"] != "turn":
            raise ValueError("Equivalence prompts must be training-eligible turn examples")
        result.append({"input_ids": row["input_ids"], "prompt_tokens": row["prompt_tokens"],
                       "source": source, "input_sha256": digest(row["input_ids"])})
    if len({(r["source"]["task_id"], r["source"]["turn_index"]) for r in result}) != PROMPT_COUNT:
        raise ValueError("Equivalence turns must be distinct")
    return result


def hf_logprobs(model, row):
    import torch
    ids = torch.tensor([row["input_ids"]], device=next(model.parameters()).device)
    start = row["prompt_tokens"]
    positions = torch.arange(start - 1, ids.shape[1] - 1, device=ids.device)
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=positions).logits[0].float()
        values = logits.log_softmax(-1).gather(1, ids[0, start:, None]).squeeze(1)
    return values.cpu().tolist()


def add_hf_references(model, cases):
    """Score the same tokens with/without LoRA on one loaded PEFT model."""
    for row in cases:
        row["hf_logprobs"] = hf_logprobs(model, row)
    with model.disable_adapter():
        for row in cases:
            row["hf_base_logprobs"] = hf_logprobs(model, row)


def validate_cases(cases, *, score_fields=("hf_logprobs", "hf_base_logprobs")):
    if len(cases) != PROMPT_COUNT:
        raise ValueError("Exactly eight reference prompts are required")
    keys = set()
    for row in cases:
        ids, start, source = row["input_ids"], row["prompt_tokens"], row["source"]
        if (source.get("training_eligible") is not True or not source["task_id"].startswith("train/")
                or type(start) is not int or not 0 < start < len(ids) <= 32768
                or any(type(i) is not int or i < 0 for i in ids) or ids[-1] != 106
                or digest(ids) != row["input_sha256"]):
            raise ValueError("Invalid training-eligible fixed prompt reference")
        for field in score_fields:
            values = row.get(field)
            if (not isinstance(values, list) or len(values) != len(ids) - start
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError(f"Missing/invalid {field}; regenerate the HF reference with base noise")
        keys.add((source["task_id"], source["turn_index"]))
    if len(keys) != PROMPT_COUNT:
        raise ValueError("Duplicate fixed prompt references")


def vllm_logprobs(client, endpoint, model, row):
    ids = row["input_ids"]
    # Token IDs bypass server tokenization and preserve the exact Gemma OFF prefix.
    # echo + max_tokens=0 is vLLM's prompt-only response mode (internally one
    # discarded decode token). Scored tokens include the native <turn|> closure.
    response = client.post(endpoint.rstrip("/") + "/completions", json={
        "model": model, "prompt": ids, "prompt_logprobs": 1, "echo": True,
        "max_tokens": 0, "temperature": 0, "stream": False, "add_special_tokens": False,
        "return_token_ids": True, "skip_special_tokens": False})
    response.raise_for_status()
    data = response.json()
    if data.get("model") != model or len(data["choices"]) != 1:
        raise ValueError("Unexpected completion identity or choice count")
    choice = data["choices"][0]
    if choice.get("prompt_token_ids") != ids or len(choice["prompt_logprobs"]) != len(ids):
        raise ValueError("Server prompt token IDs/logprob alignment differs")
    values = []
    for pos in range(row["prompt_tokens"], len(ids)):
        value = choice["prompt_logprobs"][pos][str(ids[pos])]["logprob"]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Missing/nonfinite prompt logprob")
        values.append(value)
    return values


def comparison(hf, adapter, base, hf_base, *, pooled=False):
    """Check pooled equivalence and visible adapter effect against kernel noise."""
    if not hf or any(len(values) != len(hf) for values in (adapter, base, hf_base)):
        raise ValueError("Reply logprob lengths differ")
    if any(type(v) not in (int, float) or not math.isfinite(v)
           for values in (hf, adapter, base, hf_base) for v in values):
        raise ValueError("Nonfinite logprob")
    error = [abs(a - b) for a, b in zip(adapter, hf)]
    noise = [abs(a - b) for a, b in zip(base, hf_base)]
    effect = [abs(a - b) for a, b in zip(adapter, base)]
    mean_error, mean_noise, mean_effect = (sum(values) / len(values) for values in (error, noise, effect))
    tolerance = max(MEAN_TOLERANCE, BASE_NOISE_MULTIPLIER * mean_noise) if pooled else None
    min_effect = (MIN_EFFECT_NOISE_MULTIPLIER if pooled else 1) * mean_noise

    def error_stats(values, mean):
        return {"max_abs_diff": max(values), "mean_abs_diff": mean,
                "tokens_abs_diff_gt_0_25": sum(v > 0.25 for v in values),
                "tokens_abs_diff_gt_0_5": sum(v > 0.5 for v in values)}

    adapter_error, base_noise = error_stats(error, mean_error), error_stats(noise, mean_noise)
    outlier_limit = max(OUTLIER_COUNT_MULTIPLIER * base_noise["tokens_abs_diff_gt_0_5"]
                        + OUTLIER_COUNT_ALLOWANCE,
                        math.ceil(OUTLIER_TOKEN_FRACTION * len(hf))) if pooled else None
    return {"tokens": len(hf),
            "adapter_error": adapter_error,
            "base_noise": base_noise,
            "effect": {"max_abs_diff": max(effect), "mean_abs_diff": mean_effect},
            "mean_abs_diff_tolerance_nats": tolerance,
            "tokens_abs_diff_gt_0_5_tolerance": outlier_limit,
            "min_effect_nats": min_effect,
            "passed": mean_effect >= min_effect and (not pooled or (
                mean_error <= tolerance
                and adapter_error["tokens_abs_diff_gt_0_5"] <= outlier_limit))}


def summarize_prompts(cases):
    """Recompute all statistics from token measurements, including pooled gates."""
    validate_cases(cases)
    prompts = []
    all_hf, all_adapter, all_base, all_hf_base = [], [], [], []
    for row in cases:
        hf, hf_base = row["hf_logprobs"], row["hf_base_logprobs"]
        adapter, base = row["adapter_vllm_logprobs"], row["base_vllm_logprobs"]
        stats = comparison(hf, adapter, base, hf_base)
        prompts.append({**row, **stats})
        all_hf.extend(hf)
        all_adapter.extend(adapter)
        all_base.extend(base)
        all_hf_base.extend(hf_base)
    aggregate = comparison(all_hf, all_adapter, all_base, all_hf_base, pooled=True)
    return {"schema_version": SCHEMA_VERSION, "prompts": prompts, "aggregate": aggregate,
            "thresholds": {"mean_abs_diff_nats": MEAN_TOLERANCE,
                           "base_noise_multiplier": BASE_NOISE_MULTIPLIER,
                           "min_effect_noise_multiplier": MIN_EFFECT_NOISE_MULTIPLIER,
                           "outlier_count_multiplier": OUTLIER_COUNT_MULTIPLIER,
                           "outlier_count_allowance": OUTLIER_COUNT_ALLOWANCE,
                           "outlier_token_fraction": OUTLIER_TOKEN_FRACTION},
            "rule": "Pooled: adapter_error <= max(0.05, 1.5 * base_noise) "
                    "and adapter_count(abs_diff > 0.5) <= "
                    "max(2 * base_count(abs_diff > 0.5) + 2, ceil(0.05 * pooled_tokens)) "
                    "and effect >= 2 * base_noise; every prompt: effect >= base_noise",
            "passed": len(prompts) == PROMPT_COUNT and aggregate["passed"]
                      and all(p["passed"] for p in prompts)}


def compare_prompts(cases, client, endpoint, name, base_name):
    validate_cases(cases)
    return summarize_prompts([
        {**row, "adapter_vllm_logprobs": vllm_logprobs(client, endpoint, name, row),
         "base_vllm_logprobs": vllm_logprobs(client, endpoint, base_name, row)}
        for row in cases])


def source_identity(args, config, adapter):
    return {"adapter": adapter_identity(adapter),
            "verified_manifest_sha256": file_hash(Path(args.verified) / "manifest.json"),
            "manifest_sha256": file_hash(config["paths"]["manifest"]),
            "split_freeze_sha256": file_hash(config["paths"]["split_freeze"]),
            "snapshot_metadata_sha256": snapshot_metadata(args.snapshot)["metadata_sha256"],
            "prompt_count": PROMPT_COUNT,
            "selection": "first_8_sorted_task_id_turn_index_v1", "packing": "turn",
            "enable_thinking": False, "dtype": "bfloat16", "attention": "eager",
            "peft_autocast_adapter_dtype": False,
            "scoring": "teacher reply plus native turn closure, natural-log probabilities"}


def check_server(server, endpoint, model_name, adapter, *, snapshot=None):
    info = server["config"]["server"]
    if info.get("endpoint") != endpoint.rstrip("/") or info.get("merged_model"):
        raise ValueError("Endpoint differs from unmerged server provenance")
    matches = [a for a in info.get("lora_adapters", []) if a["name"] == model_name]
    if len(matches) != 1 or adapter_identity(matches[0]) != adapter_identity(adapter):
        raise ValueError("Requested adapter differs from server provenance")
    validate_served_adapter(matches[0], snapshot or info.get("model_path"))
    return info["served_model_name"]


def validate_report(report, server, model_name, endpoint):
    """Fail closed before evaluation; bind a passing check to this server and weights."""
    if type(report.get("schema_version")) is not int or report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Schema version 5 is required; recompute complete schema 2/3/4 measurements "
                         "under schema 5 or rerun the equivalence check; schema 1 is refused")
    adapters = server["config"]["server"].get("lora_adapters", [])
    matches = [a for a in adapters if a["name"] == model_name]
    if len(matches) != 1:
        raise ValueError("Requested model is not a recorded server adapter")
    check_server(server, endpoint, model_name, matches[0])
    if (report.get("passed") is not True or report.get("model_name") != model_name
            or report.get("server_provenance_sha256") != digest(server)
            or report.get("identity", {}).get("adapter") != adapter_identity(matches[0])
            or len(report.get("prompts", [])) != PROMPT_COUNT):
        raise ValueError("A passing equivalence report for this adapter/server is required")
    if not summarize_prompts(report["prompts"])["passed"]:
        raise ValueError("Equivalence report fails the pooled noise/outlier/effect "
                         "or per-prompt effect thresholds")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--adapter", required=True, help="Original HF/PEFT checkpoint, never the converted copy")
    parser.add_argument("--verified", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--server-provenance")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--hf-only", action="store_true")
    modes.add_argument("--hf-reference", help="reference.json produced by --hf-only on this checkpoint")
    parser.add_argument("--output", required=True, help="New directory for JSON report/provenance")
    args = parser.parse_args(argv)
    if not args.hf_only:
        url = urlsplit(args.endpoint or "")
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}
                or url.username or url.password or url.query or url.fragment
                or not args.server_provenance):
            parser.error("Checking requires a loopback --endpoint and --server-provenance")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = load_config(args.config)
        student = config["models"]["student"]
        parse_adapters([args.model_name + "=" + args.adapter], student["requested_repo"])
        adapter = inspect_adapter(args.model_name, args.adapter, student, args.snapshot)
        identity = source_identity(args, config, adapter)
        inputs = [args.config, Path(args.verified) / "manifest.json", config["paths"]["manifest"],
                  config["paths"]["split_freeze"], Path(args.adapter) / "training_state.json"]
        if args.hf_reference:
            inputs.append(args.hf_reference)
        if args.server_provenance:
            inputs.append(args.server_provenance)
        provenance = write_provenance(output, command="check_lora_serving", config={**config,
            "lora_check": vars(args)}, inputs=inputs, details={"identity": identity})
        if args.hf_reference:
            reference = json.loads(Path(args.hf_reference).read_text())
            if reference["identity"] != identity:
                raise ValueError("HF reference identity differs from current checkpoint/data/template")
            old = json.loads((Path(args.hf_reference).parent / "provenance.json").read_text())
            if (reference["provenance_sha256"] != digest(old)
                    or old["code_sha256"] != provenance["code_sha256"]):
                raise ValueError("HF reference provenance/code mismatch")
            cases = reference["prompts"]
        else:
            from train.dataset import build_verified_dataset, load_tokenizer
            from train.modeling import load_student
            import torch
            from peft import PeftModel
            if not torch.cuda.is_available():
                raise RuntimeError("HF equivalence reference requires a GPU (bf16/eager)")
            tokenizer = load_tokenizer(args.snapshot)
            manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
            cases = fixed_prompts(build_verified_dataset(args.verified, tokenizer, manifest, freeze,
                                                         packing="turn"))
            model = PeftModel.from_pretrained(load_student(args.snapshot), args.adapter,
                                             local_files_only=True, is_trainable=False,
                                             autocast_adapter_dtype=False)
            add_hf_references(model, cases)
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            reference = {"schema_version": SCHEMA_VERSION, "identity": identity, "prompts": cases,
                         "template_sha256": digest(tokenizer.chat_template),
                         "provenance_sha256": digest(provenance)}
            write_json(output / "reference.json", reference, exclusive=True)
        validate_cases(cases)
        if args.hf_only:
            return 0
        server = json.loads(Path(args.server_provenance).read_text())
        base_name = check_server(server, args.endpoint, args.model_name, adapter, snapshot=args.snapshot)
        import httpx
        with httpx.Client(timeout=180, trust_env=False) as client:
            report = compare_prompts(cases, client, args.endpoint, args.model_name, base_name)
        report.update(model_name=args.model_name, base_model_name=base_name,
                      identity=identity, server_provenance_sha256=digest(server),
                      lora_dtype=server["config"]["server"].get("lora_dtype", "bfloat16"),
                      template_sha256=reference["template_sha256"])
        write_json(output / "report.json", report, exclusive=True)
        print(json.dumps({"passed": report["passed"], "aggregate": report["aggregate"]}))
        return 0 if report["passed"] else 1
    except Exception as exc:
        write_json(output / "failure.json", {"passed": False, "error_type": type(exc).__name__}, exclusive=True)
        print("LoRA check failed; inspect sanitized failure.json and provenance.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
