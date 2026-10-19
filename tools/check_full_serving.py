"""Opt-in HF/vLLM full-checkpoint gate for STeCa, using fixed verified TRAIN turns."""
import argparse
import math
from pathlib import Path

from baselines.common.fsdp_train import validate_checkpoint
from baselines.common.method_data import read
from sed.config import load_config
from sed.manifest import load_frozen
from sed.storage import digest, file_hash, write_json
from sed.provenance import write_provenance
from tools.check_lora_serving import fixed_prompts, hf_logprobs, vllm_logprobs, validate_cases


def statistics(cases):
    validate_cases(cases)
    errors = []
    for row in cases:
        values = row["vllm_logprobs"]
        if len(values) != len(row["hf_logprobs"]) or any(not math.isfinite(v) for v in values):
            raise ValueError("Invalid full-checkpoint token scores")
        errors += [abs(x-y) for x, y in zip(values, row["hf_logprobs"])]
    mean = sum(errors) / len(errors)
    outliers = sum(v > .5 for v in errors)
    return {"tokens": len(errors), "mean_abs_diff": mean, "outliers_gt_0_5": outliers,
            "passed": mean <= .05 and outliers <= max(2, math.ceil(.05*len(errors))),
            "mean_tolerance": .05, "outlier_fraction_tolerance": .05}


def validate_report(report, server, endpoint):
    if (not report["passed"] or report["statistics"] != statistics(report["cases"])
            or not report["statistics"]["passed"] or report["server_sha256"] != digest(server)
            or report["endpoint"] != endpoint):
        raise ValueError("Full-checkpoint equivalence gate failed or changed")
    serving = server["config"]["server"]
    if (Path(serving["model_path"]).resolve() != Path(report["checkpoint"]).resolve()
            or serving["training_provenance_sha256"] != report["training_provenance_sha256"]
            or serving.get("endpoint") != endpoint.rstrip("/")
            or serving.get("batch_invariant") is not True):
        raise ValueError("Full-checkpoint report belongs to another server")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "snapshot", "checkpoint", "verified", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--hf-only", action="store_true")
    p.add_argument("--hf-reference")
    p.add_argument("--endpoint")
    p.add_argument("--server-provenance")
    a = p.parse_args(argv)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = load_config(a.config)
        state = validate_checkpoint(a.checkpoint, config["models"]["student"], base_snapshot=a.snapshot)
        identity = {"state": digest(state), "verified_manifest": file_hash(Path(a.verified) / "manifest.json"),
            "checkpoint": str(Path(a.checkpoint).resolve()), "student": config["models"]["student"]}
        inputs = [a.config, Path(a.verified) / "manifest.json", config["paths"]["manifest"],
                  config["paths"]["split_freeze"], Path(a.checkpoint) / "training_state.json"]
        identity.update(manifest_sha256=file_hash(config["paths"]["manifest"]),
                        split_freeze_sha256=file_hash(config["paths"]["split_freeze"]))
        if not a.hf_only:
            inputs += [a.hf_reference, a.server_provenance]
        write_provenance(output, command="full_checkpoint_equivalence", config=config, inputs=inputs, details=identity)
        if a.hf_only:
            from train.dataset import load_tokenizer, build_verified_dataset
            from train.modeling import load_student
            manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
            cases = fixed_prompts(build_verified_dataset(a.verified, load_tokenizer(a.snapshot), manifest, freeze, packing="turn"))
            model = load_student(a.checkpoint)
            for row in cases:
                row["hf_logprobs"] = hf_logprobs(model, row)
                row["hf_base_logprobs"] = row["hf_logprobs"]  # shared reference validator only
            write_json(output / "reference.json", {"identity": identity, "cases": cases})
            return 0
        reference = read(a.hf_reference)
        if reference["identity"] != identity:
            raise ValueError("Full checkpoint/reference identity changed")
        import httpx
        server = read(a.server_provenance)
        with httpx.Client(timeout=180, trust_env=False) as client:
            cases = [{**r, "vllm_logprobs": vllm_logprobs(client, a.endpoint,
                config["models"]["student"]["requested_repo"], r)} for r in reference["cases"]]
        stats = statistics(cases)
        report = {"passed": stats["passed"], "statistics": stats, "cases": cases, "identity": identity,
            "checkpoint": identity["checkpoint"], "server_sha256": digest(server), "endpoint": a.endpoint,
            "training_provenance_sha256": file_hash(Path(a.checkpoint).parent / "provenance.json")}
        write_json(output / "report.json", report)
        validate_report(report, server, a.endpoint)
        return 0
    except Exception as exc:
        write_json(output / "failure.json", {"error_type": type(exc).__name__, "passed": False})
        print("Full checkpoint equivalence failed; inspect sanitized failure.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
