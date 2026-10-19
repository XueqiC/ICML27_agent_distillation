"""HF/vLLM equivalence on eight verified TRAIN turns with the entire MCP box."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit

from baselines.agentdistill.box import load_box
from baselines.agentdistill.prompts import messages
from baselines.agentdistill.teacher import read
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes, checked_episode, load_verified_collection
from sed.storage import digest, file_hash, write_json
from sed.visibility import visible_observation
from tools.check_full_serving import compare
from tools.check_lora_serving import fixed_prompts, validate_cases, hf_logprobs, vllm_logprobs


def cases_for(verified, tokenizer, config, tools):
    from train.dataset import encode_turn
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(verified, manifest, freeze)
    examples = []
    for receipt in sorted(collection["verified"], key=lambda r: r["task_id"]):
        episode = checked_episode(root, receipt)
        history, visible = [], visible_observation(episode["initial"]["visible"])
        for index, step in enumerate(episode["steps"]):
            reply = step["action_raw"]
            encoded = encode_turn(tokenizer, messages(history, {}, visible, tools), reply, max_length=32768)
            examples.append({**encoded, "packing": "turn", "label_provenance": {
                "task_id": episode["task_id"], "turn_index": index, "training_eligible": True,
                "source": "verified_support_extended_react_equivalence_only"}})
            if len(examples) == 8:
                return fixed_prompts(examples)
            history.append({"visible": visible, "reply": reply})
            visible = visible_observation(step["visible"])
    return fixed_prompts(examples)


def check_server(server, endpoint, identity):
    info = server["config"]["server"]
    student = server["config"]["models"]["student"]
    if (info["endpoint"] != endpoint.rstrip("/") or info["model_path"] != identity["snapshot"]
            or info["checkpoint_kind"] != "base" or info["merged_model"] or info["lora_adapters"]
            or info["batch_invariant"] is not True or student != identity["student"]
            or info["served_model_name"] != student["requested_repo"]):
        raise ValueError("AgentDistill requires the original frozen base student/server")
    return info["served_model_name"]


def validate_report(report, server, endpoint, box_path, box_hash):
    validate_cases(report["prompts"], score_fields=("hf_logprobs", "vllm_logprobs"))
    stats = compare(report["prompts"])
    identity = report["identity"]
    prompts = [{k: v for k, v in r.items() if k not in {"hf_logprobs", "vllm_logprobs"}}
               for r in report["prompts"]]
    if (not stats["passed"] or any(report.get(k) != v for k, v in stats.items())
            or report["server_provenance_sha256"] != digest(server)
            or identity["box_sha256"] != box_hash or identity["prompts_sha256"] != digest(prompts)
            or report["endpoint"] != endpoint.rstrip("/")
            or artifact_hashes(identity["snapshot"]) != identity["snapshot_artifact_sha256"]):
        raise ValueError("AgentDistill equivalence gate failed or identity changed")
    load_box(box_path, box_hash)
    check_server(server, endpoint, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "snapshot", "verified", "box", "box-sha256", "output"):
        parser.add_argument("--" + key, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--hf-only", action="store_true")
    modes.add_argument("--hf-reference")
    parser.add_argument("--endpoint")
    parser.add_argument("--server-provenance")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        config = load_config(args.config)
        box = load_box(args.box, args.box_sha256)
        from train.dataset import load_tokenizer
        tokenizer = load_tokenizer(args.snapshot)
        cases = cases_for(args.verified, tokenizer, config, box["tools"])
        identity = {"snapshot": str(Path(args.snapshot).resolve()), "snapshot_artifact_sha256": artifact_hashes(args.snapshot),
            "student": config["models"]["student"], "config_sha256": digest(config),
            "box_sha256": args.box_sha256, "prompts_sha256": digest(cases),
            "verified_sha256": file_hash(Path(args.verified) / "manifest.json"),
            "template_sha256": digest(tokenizer.chat_template)}
        provenance = write_provenance(output, command="agentdistill_equivalence", config=config,
            inputs=[args.config, args.box, Path(args.verified) / "manifest.json"], details=identity)
        if args.hf_only:
            import torch
            from train.modeling import load_student
            if not torch.cuda.is_available():
                raise RuntimeError("Production gate requires allocated GPU; tests use stubs")
            model = load_student(args.snapshot)
            for row in cases:
                row["hf_logprobs"] = hf_logprobs(model, row)
            validate_cases(cases, score_fields=("hf_logprobs",))
            write_json(output / "reference.json", {"identity": identity, "prompts": cases,
                "provenance_sha256": digest(provenance)})
            return 0
        url = urlsplit(args.endpoint or "")
        if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"}
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("Loopback vLLM endpoint required")
        reference = read(args.hf_reference)
        prior = read(Path(args.hf_reference).parent / "provenance.json")
        if (reference["identity"] != identity or reference["provenance_sha256"] != digest(prior)
                or prior["code_sha256"] != provenance["code_sha256"]
                or [{k: v for k, v in r.items() if k != "hf_logprobs"} for r in reference["prompts"]] != cases):
            raise ValueError("HF reference or prompt identity changed")
        cases = reference["prompts"]
        validate_cases(cases, score_fields=("hf_logprobs",))
        server = read(args.server_provenance)
        model_name = check_server(server, args.endpoint, identity)
        import httpx
        with httpx.Client(timeout=180, trust_env=False) as client:
            for row in cases:
                row["vllm_logprobs"] = vllm_logprobs(client, args.endpoint, model_name, row)
        report = {**compare(cases), "identity": identity, "prompts": cases,
            "server_provenance_sha256": digest(server), "endpoint": args.endpoint.rstrip("/")}
        write_json(output / "report.json", report)
        validate_report(report, server, args.endpoint, args.box, args.box_sha256)
        return 0
    except Exception as exc:
        write_json(output / "failure.json", {"passed": False, "error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
