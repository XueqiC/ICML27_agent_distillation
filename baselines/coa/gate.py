"""CoA-format HF/vLLM equivalence using only fixed verified TRAIN prefixes."""
import argparse
import math
from pathlib import Path
from urllib.parse import urlsplit

from baselines.coa.data import read, support
from baselines.coa.protocol import messages
from baselines.common.fsdp_train import validate_checkpoint
from sed.config import load_config
from sed.provenance import write_provenance
from sed.seed_evidence import checked_episodes, load_verified_collection
from sed.manifest import load_frozen
from sed.storage import digest, file_hash, write_json
from tools.check_lora_serving import fixed_prompts, hf_logprobs, vllm_logprobs, validate_cases
from train.dataset import encode_turn, load_tokenizer


def cases(verified, config, tokenizer):
    support(config, verified)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(verified, manifest, freeze)
    rows = []
    for episode in checked_episodes(root, collection["verified"]):
        page, history = episode["initial"]["visible"], []
        for index, step in enumerate(episode["steps"]):
            # Format conversion solely for a numerical serving check; no
            # fabricated role work is admitted to the SFT dataset.
            reply = "<alfworld>Action: " + step["action_parsed"] + "</alfworld>"
            row = encode_turn(tokenizer, messages(history, page), reply, max_length=32768)
            row.update(packing="turn", label_provenance={"task_id": episode["task_id"], "turn_index": index,
                "source": "verified_seed_coa_gate_only", "training_eligible": True})
            rows.append(row)
            history.append({"visible": page, "reply": reply})
            page = step["visible"]
    return fixed_prompts(rows)


def comparison(rows):
    validate_cases(rows, score_fields=("hf_logprobs", "vllm_logprobs"))
    errors = [abs(a - b) for r in rows for a, b in zip(r["hf_logprobs"], r["vllm_logprobs"])]
    mean = sum(errors) / len(errors)
    outliers = sum(e > .5 for e in errors)
    limit = max(2, math.ceil(.05 * len(errors)))
    return {"passed": mean <= .05 and outliers <= limit, "mean_abs_diff": mean,
            "outliers_gt_0_5": outliers, "outlier_limit": limit, "tokens": len(errors)}


def validate_report(report, server, model, endpoint):
    info = server["config"]["server"]
    checkpoint = Path(info["model_path"])
    if (report.get("passed") is not True or report.get("method") != "coa"
            or report["server_provenance_sha256"] != digest(server)
            or info["endpoint"] != endpoint or info["served_model_name"] != model
            or info["checkpoint_kind"] != "full_parameter" or info["batch_invariant"] is not True
            or report["identity"]["checkpoint"] != str(checkpoint.resolve())
            or report["identity"]["checkpoint_state_sha256"] != file_hash(checkpoint / "training_state.json")
            or not comparison(report["prompts"])["passed"]):
        raise ValueError("Passing CoA-format gate bound to this checkpoint/server required")
    state = validate_checkpoint(checkpoint, server["config"]["models"]["student"])
    if state.get("method") != "coa":
        raise ValueError("CoA checkpoint required")


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
        if state.get("method") != "coa":
            raise ValueError("CoA checkpoint required")
        identity = {"checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_state_sha256": file_hash(Path(args.checkpoint) / "training_state.json"),
            "verified_manifest_sha256": file_hash(Path(args.verified) / "manifest.json"),
            "config_sha256": digest(config), "format": "coa", "selection": "first_8_sorted_task_id_turn_index_v1"}
        provenance = write_provenance(output, command="coa_equivalence", config=config,
            inputs=[args.config, Path(args.checkpoint) / "training_state.json"], details=identity)
        if args.hf_only:
            import torch
            from train.modeling import load_student
            if not torch.cuda.is_available():
                raise RuntimeError("Production HF gate needs a GPU")
            rows = cases(args.verified, config, load_tokenizer(args.checkpoint))
            model = load_student(args.checkpoint, dtype="bfloat16")
            for row in rows:
                row["hf_logprobs"] = hf_logprobs(model, row)
            write_json(output / "reference.json", {"identity": identity, "prompts": rows,
                "provenance_sha256": digest(provenance)})
            return 0
        url = urlsplit(args.endpoint or "")
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"} or url.username or url.password:
            raise ValueError("Gate requires local student endpoint")
        reference = read(args.hf_reference)
        previous = read(Path(args.hf_reference).parent / "provenance.json")
        if (reference["identity"] != identity or reference["provenance_sha256"] != digest(previous)
                or previous["code_sha256"] != provenance["code_sha256"]):
            raise ValueError("HF reference identity/code changed")
        server = read(args.server_provenance)
        import httpx
        name = config["models"]["student"]["requested_repo"]
        with httpx.Client(timeout=180, trust_env=False) as client:
            rows = [{**r, "vllm_logprobs": vllm_logprobs(client, args.endpoint, name, r)} for r in reference["prompts"]]
        report = {**comparison(rows), "method": "coa", "identity": identity, "prompts": rows,
                  "server_provenance_sha256": digest(server)}
        write_json(output / "report.json", report)
        validate_report(report, server, name, args.endpoint)
        return 0
    except Exception as exc:
        write_json(output / "failure.json", {"passed": False, "error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
