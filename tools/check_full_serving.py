"""HF/vLLM check for standalone full checkpoints using verified TRAIN turns."""
import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from baselines.common.fsdp_train import validate_checkpoint
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.storage import digest, file_hash, write_json
from tools.check_lora_serving import fixed_prompts, hf_logprobs, vllm_logprobs, validate_cases, PROMPT_COUNT


def compare(cases):
    if not cases:
        raise ValueError("Full checkpoint logprobs are empty")
    errors = []
    for case in cases:
        hf, vllm = case["hf_logprobs"], case["vllm_logprobs"]
        if (not hf or len(hf) != len(vllm)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in [*hf, *vllm])):
            raise ValueError("Full checkpoint logprobs are invalid")
        errors.extend(abs(a - b) for a, b in zip(hf, vllm))
    mean = sum(errors) / len(errors)
    outliers = sum(e > .5 for e in errors)
    return {"passed": mean <= .05 and outliers <= math.ceil(.05 * len(errors)),
            "mean_absolute_error": mean, "outliers_over_0_5": outliers, "tokens": len(errors),
            "rule": "pooled mean absolute logprob error <= .05; at most ceil(.05*N) errors > .5"}


def check_server(server, endpoint, checkpoint):
    info = server["config"]["server"]
    if (info.get("endpoint") != endpoint.rstrip("/") or not info.get("merged_model")
            or info.get("model_path") != str(Path(checkpoint).resolve())
            or info.get("training_provenance_sha256") != file_hash(Path(checkpoint).parent / "provenance.json")
            or info.get("lora_adapters") or info.get("checkpoint_kind") != "full_parameter"
            or info.get("batch_invariant") is not True):
        raise ValueError("Full checkpoint differs from serving identity")
    return info["served_model_name"]


def validate_report(report, server, endpoint):
    """Consume the same full-checkpoint report for SAD and STeCa evaluation."""
    cases = report["prompts"]
    validate_cases(cases, score_fields=("hf_logprobs",))
    stats = compare(cases)
    prompts = [{k: v for k, v in row.items() if k not in {"hf_logprobs", "vllm_logprobs"}}
               for row in cases]
    identity = report["identity"]
    if (not stats["passed"] or any(report.get(k) != v for k, v in stats.items())
            or report["server_provenance_sha256"] != digest(server)
            or report["endpoint"] != endpoint.rstrip("/")
            or identity["prompt_count"] != PROMPT_COUNT or identity["prompts_sha256"] != digest(prompts)):
        raise ValueError("Full-checkpoint equivalence gate failed or changed")
    serving = server["config"]["server"]
    if (Path(serving["model_path"]).resolve() != Path(identity["checkpoint"]).resolve()
            or serving.get("training_provenance_sha256") != report["training_provenance_sha256"]
            or serving.get("endpoint") != endpoint.rstrip("/")
            or serving.get("served_model_name") != report["model_name"]
            or serving.get("batch_invariant") is not True or not serving.get("merged_model")
            or serving.get("lora_adapters") or serving.get("checkpoint_kind") != "full_parameter"):
        raise ValueError("Full-checkpoint report belongs to another server")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "snapshot", "checkpoint", "verified", "output"):
        parser.add_argument("--" + name, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--hf-only", action="store_true")
    modes.add_argument("--hf-reference")
    parser.add_argument("--endpoint")
    parser.add_argument("--server-provenance")
    args = parser.parse_args(argv)
    if not args.hf_only:
        url = urlsplit(args.endpoint or "")
        if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"}
                or url.username or url.password or url.query or url.fragment or not args.server_provenance):
            parser.error("A loopback endpoint and server provenance are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = load_config(args.config)
        state = validate_checkpoint(args.checkpoint, config["models"]["student"], base_snapshot=args.snapshot)
        from train.dataset import build_verified_dataset, load_tokenizer
        tokenizer = load_tokenizer(args.snapshot)
        manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
        expected = fixed_prompts(build_verified_dataset(args.verified, tokenizer, manifest, freeze, packing="turn"))
        identity = {"checkpoint": str(Path(args.checkpoint).resolve()), "state_sha256": digest(state),
            "verified_manifest_sha256": file_hash(Path(args.verified) / "manifest.json"),
            "config_sha256": digest(config), "template_sha256": digest(tokenizer.chat_template),
            "prompts_sha256": digest(expected), "prompt_count": PROMPT_COUNT,
            "manifest_sha256": file_hash(config["paths"]["manifest"]),
            "split_freeze_sha256": file_hash(config["paths"]["split_freeze"])}
        inputs = [args.config, Path(args.checkpoint) / "training_state.json", Path(args.verified) / "manifest.json",
                  config["paths"]["manifest"], config["paths"]["split_freeze"]]
        if args.hf_reference:
            inputs += [args.hf_reference, args.server_provenance]
        provenance = write_provenance(output, command="check_full_serving", config=config, inputs=inputs, details=identity)
        if args.hf_only:
            import torch
            from train.modeling import load_student
            if not torch.cuda.is_available():
                raise RuntimeError("HF reference requires allocated GPU; CPU tests use stubs")
            model = load_student(args.checkpoint)
            for row in expected:
                row["hf_logprobs"] = hf_logprobs(model, row)
            validate_cases(expected, score_fields=("hf_logprobs",))
            write_json(output / "reference.json", {"identity": identity, "prompts": expected,
                       "provenance_sha256": digest(provenance)})
            return 0
        reference = json.loads(Path(args.hf_reference).read_text())
        prior = json.loads((Path(args.hf_reference).parent / "provenance.json").read_text())
        if (reference["identity"] != identity or reference["provenance_sha256"] != digest(prior)
                or prior["code_sha256"] != provenance["code_sha256"]):
            raise ValueError("Full HF reference identity/code changed")
        cases = reference["prompts"]
        if [{k: v for k, v in row.items() if k != "hf_logprobs"} for row in cases] != expected:
            raise ValueError("Full HF reference prompts changed")
        validate_cases(cases, score_fields=("hf_logprobs",))
        server = json.loads(Path(args.server_provenance).read_text())
        name = check_server(server, args.endpoint, args.checkpoint)
        import httpx
        with httpx.Client(timeout=180, trust_env=False) as client:
            for row in cases:
                row["vllm_logprobs"] = vllm_logprobs(client, args.endpoint, name, row)
        report = {**compare(cases), "identity": identity, "prompts": cases,
                  "server_provenance_sha256": digest(server), "model_name": name,
                  "endpoint": args.endpoint.rstrip("/"),
                  "training_provenance_sha256": file_hash(Path(args.checkpoint).parent / "provenance.json")}
        write_json(output / "report.json", report)
        validate_report(report, server, args.endpoint)
        return 0 if report["passed"] else 1
    except Exception as exc:
        write_json(output / "failure.json", {"passed": False, "error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
