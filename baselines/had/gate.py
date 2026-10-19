"""Split HF/vLLM equivalence gate for standalone HAD checkpoints, TRAIN only."""
import argparse
import json
import math
from pathlib import Path

from baselines.common.fsdp_train import validate_checkpoint
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.storage import digest, file_hash, write_json
from tools.check_lora_serving import fixed_prompts, hf_logprobs, vllm_logprobs


def comparison(cases):
    if len(cases) != 8:
        raise ValueError("HAD gate requires eight verified TRAIN cases")
    errors = []
    for row in cases:
        a, b = row["hf_logprobs"], row["vllm_logprobs"]
        if (not a or len(a) != len(b) or len(a) != len(row["input_ids"]) - row["prompt_tokens"]
                or digest(row["input_ids"]) != row["input_sha256"]
                or row["source"].get("training_eligible") is not True
                or not row["source"]["task_id"].startswith("train/")
                or any(not math.isfinite(v) for v in a + b)):
            raise ValueError("Invalid HAD equivalence measurements")
        errors.extend(abs(x - y) for x, y in zip(a, b))
    mean = sum(errors) / len(errors)
    outliers = sum(v > .5 for v in errors)
    limit = max(2, math.ceil(.05 * len(errors)))
    return {"passed": mean <= .05 and outliers <= limit, "mean_abs_diff": mean,
            "outliers_gt_0_5": outliers, "outlier_limit": limit, "tokens": len(errors)}


def validate_report(report, server, model, endpoint):
    info = server["config"]["server"]
    checkpoint = Path(info["model_path"])
    if (report.get("passed") is not True or report["server_provenance_sha256"] != digest(server)
            or info["endpoint"] != endpoint or info["served_model_name"] != model
            or info["checkpoint_kind"] != "full_parameter" or info["batch_invariant"] is not True
            or report["identity"]["checkpoint_state_sha256"] != file_hash(checkpoint / "training_state.json")
            or report["identity"]["checkpoint"] != str(checkpoint.resolve())
            or not comparison(report["prompts"])["passed"]):
        raise ValueError("Passing HAD equivalence gate for these weights/server required")
    validate_checkpoint(checkpoint, server["config"]["models"]["student"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "snapshot", "checkpoint", "verified", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--hf-only", action="store_true")
    parser.add_argument("--hf-reference")
    parser.add_argument("--endpoint")
    parser.add_argument("--server-provenance")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = load_config(args.config)
        state = validate_checkpoint(args.checkpoint, config["models"]["student"], base_snapshot=args.snapshot)
        if state.get("method") != "had":
            raise ValueError("This gate is for HAD full checkpoints")
        identity = {"checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_state_sha256": file_hash(Path(args.checkpoint) / "training_state.json"),
            "verified_manifest_sha256": file_hash(Path(args.verified) / "manifest.json"),
            "config_sha256": digest(config), "selection": "first_8_sorted_task_id_turn_index_v1"}
        provenance = write_provenance(output, command="had_equivalence", config=config,
            inputs=[args.config, Path(args.checkpoint) / "training_state.json"], details=identity)
        if args.hf_only:
            import torch
            from train.dataset import build_verified_dataset, load_tokenizer
            from train.modeling import load_student
            if not torch.cuda.is_available():
                raise RuntimeError("Production HF gate needs a GPU; CPU tests use stubs")
            manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
            cases = fixed_prompts(build_verified_dataset(args.verified, load_tokenizer(args.checkpoint),
                                                         manifest, freeze, packing="turn"))
            model = load_student(args.checkpoint, dtype="bfloat16")
            for row in cases:
                row["hf_logprobs"] = hf_logprobs(model, row)
            write_json(output / "reference.json", {"identity": identity, "prompts": cases,
                "provenance_sha256": digest(provenance)})
            return 0
        from urllib.parse import urlsplit
        url = urlsplit(args.endpoint or "")
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"} or url.username or url.password:
            raise ValueError("Equivalence scoring requires the local student endpoint")
        reference = json.loads(Path(args.hf_reference).read_text())
        old = json.loads((Path(args.hf_reference).parent / "provenance.json").read_text())
        if (reference["identity"] != identity or reference["provenance_sha256"] != digest(old)
                or old["code_sha256"] != provenance["code_sha256"]):
            raise ValueError("HAD HF reference identity/code changed")
        server = json.loads(Path(args.server_provenance).read_text())
        model_name = config["models"]["student"]["requested_repo"]
        import httpx
        with httpx.Client(timeout=180, trust_env=False) as client:
            cases = [{**r, "vllm_logprobs": vllm_logprobs(client, args.endpoint, model_name, r)}
                     for r in reference["prompts"]]
        report = {**comparison(cases), "prompts": cases, "identity": identity,
                  "server_provenance_sha256": digest(server)}
        write_json(output / "report.json", report)
        validate_report(report, server, model_name, args.endpoint)
        return 0
    except Exception as exc:
        write_json(output / "failure.json", {"error_type": type(exc).__name__, "passed": False})
        print("HAD gate failed; inspect sanitized artifacts")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
