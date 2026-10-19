"""Opt-in dev140 evaluation, with equivalence gates for full and LoRA models."""
import json
import os
from pathlib import Path

from baselines.eto.pipeline import run_command, served, stage_run
from sed.config import load_config
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash


def evaluate_models(args, *, decodings=("greedy",)):
    config = load_config(args.config)
    models = json.loads((Path(args.output) / "models.json").read_text())
    for model in models:
        checkpoint = Path(model["checkpoint"])
        state = json.loads((checkpoint / "training_state.json").read_text())
        hashes = artifact_hashes(checkpoint)
        hashes.pop("training_state.json")
        if hashes != state["artifact_sha256"]:
            raise ValueError("Evaluation checkpoint changed")
        for decoding in decodings:
            def evaluate(attempt):
                check = [args.train_python, "-m", "tools.check_full_serving", "--config", args.config,
                    "--snapshot", args.snapshot, "--checkpoint", str(checkpoint), "--verified", args.verified]
                if not model.get("adapter"):
                    run_command(check + ["--hf-only", "--output", str(attempt / "hf")], attempt / "hf.log")
                with served(args, checkpoint, attempt) as endpoint:
                    server = attempt / "server/provenance.json"
                    extra = []
                    if model.get("adapter"):
                        name = "dose-policy"
                        extra = ["--lora-check", str(attempt / "check/report.json")]
                    else:
                        name = config["models"]["student"]["requested_repo"]
                        run_command([args.env_python, *check[1:], "--hf-reference", str(attempt / "hf/reference.json"),
                            "--endpoint", endpoint, "--server-provenance", str(server),
                            "--output", str(attempt / "check")], attempt / "check.log")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = ""
                    run_command([args.env_python, "eval/run_suite.py", "--config", args.config,
                        "--policy", "student", "--decoding", decoding, "--endpoint", endpoint,
                        "--model-name", name, "--enable-thinking", "off", "--max-tokens", "1024",
                        "--tasks", "data/manifest/dev140.json", "--workers", str(args.workers),
                        "--output", str(attempt / "output"), "--server-provenance", str(server), *extra],
                        attempt / "eval.log", env=env)
            stage_run(args.output, "eval-" + model["name"] + "-" + decoding,
                digest([state, config, str(checkpoint), decoding,
                        file_hash(Path(args.verified) / "manifest.json")]), evaluate)
