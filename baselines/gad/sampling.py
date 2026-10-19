"""vLLM completion groups on exact native OFF prompt IDs; no teacher SDK."""
import argparse
from pathlib import Path

from baselines.common.method_data import read
from baselines.gad import RECIPE
from baselines.gad.data import encode_state, load_rows
from sed.config import load_config
from sed.provenance import write_provenance, utc_now
from sed.storage import digest, JsonlLedger, write_json
from train.dataset import load_tokenizer


def sample_group(client, endpoint, model, prompt_ids, *, seed, ledger):
    if not prompt_ids or len(prompt_ids) + 1024 > 32768:
        raise ValueError("Complete GAD prompt exceeds H0 context; no truncation")
    body = {"model": model, "prompt": prompt_ids, "n": 8, "temperature": 0.8,
        "top_p": 1.0, "top_k": -1, "max_tokens": 1024, "seed": seed,
        "stream": False, "return_token_ids": True, "skip_special_tokens": False}
    ledger.append({"status": "started", "timestamp": utc_now(), "request": body})
    try:
        response = client.post(endpoint.rstrip("/") + "/completions", json=body)
        response.raise_for_status()
        raw = response.json()
        # Keep all returned candidates, including malformed/rejected groups.
        ledger.append({"status": "returned", "timestamp": utc_now(), "response": raw})
        choices = sorted(raw["choices"], key=lambda c: c["index"])
        if raw["model"] != model or [c["index"] for c in choices] != list(range(8)):
            raise ValueError("Incomplete group or changed serving model")
        for choice in choices:
            ids = choice.get("token_ids")
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 1024
                    or any(type(i) is not int or i < 0 for i in ids)
                    or choice["finish_reason"] not in {"stop", "length"}):
                raise ValueError("Server must return exact generated IDs; cannot retokenize GAD actions")
        return [{"input_ids": prompt_ids + c["token_ids"], "prompt_tokens": len(prompt_ids),
                 "response_tokens": len(c["token_ids"]), "raw_text": c["text"],
                 "finish_reason": c["finish_reason"]} for c in choices]
    except Exception as exc:
        ledger.append({"status": "failed", "error_type": type(exc).__name__, "timestamp": utc_now()})
        raise RuntimeError("GAD sampling failed; inspect sanitized ledger") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "cell", "data", "plan", "snapshot", "endpoint", "server-provenance", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    config, identity, plan = load_config(args.config), read(Path(args.cell) / "identity.json"), read(args.plan)
    rows = load_rows(args.data, config, verified=identity["verified"], identity=identity, round_index=plan["round"])
    server = read(args.server_provenance)["config"]["server"]
    if Path(server["model_path"]).resolve() != Path(args.snapshot).resolve() or not server["batch_invariant"]:
        raise ValueError("Sampling server policy differs")
    tokenizer, output = load_tokenizer(args.snapshot), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = write_provenance(output, command="gad_sample_groups", config={**config, "gad": RECIPE},
        inputs=[args.data, args.plan, args.server_provenance],
        details={"snapshot": str(Path(args.snapshot).resolve()), "plan": plan, "identity_sha256": digest(identity)})
    import httpx
    groups = []
    with httpx.Client(timeout=180, trust_env=False) as client:
        for offset, index in enumerate(plan["indices"]):
            teacher = encode_state(rows[index], tokenizer)
            prompt_ids = teacher["input_ids"][:teacher["prompt_tokens"]]
            samples = sample_group(client, args.endpoint, config["models"]["student"]["requested_repo"],
                prompt_ids, seed=plan["seed"] + offset, ledger=JsonlLedger(output / "calls.jsonl"))
            groups.append({"state_id": rows[index]["state_id"], "teacher": teacher, "samples": samples})
    write_json(output / "groups.json", {"groups": groups, "provenance_sha256": digest(provenance),
                                       "groups_sha256": digest(groups)})


if __name__ == "__main__":
    main()
