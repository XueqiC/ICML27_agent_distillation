"""CLI plumbing for WKM/STeCa; stage transactions remain the shared implementation."""
import argparse
import os
from pathlib import Path
import signal
import sys

from baselines.common.pipeline import batch_invariant, prepare
from sed.config import load_config
from sed.provenance import versions
from sed.storage import locked


def main(method, recipe, stages, train, disclose, argv=None):
    parser = argparse.ArgumentParser(description=method.upper() + " verified-support pipeline")
    parser.add_argument("--config", default="configs/base.yaml")
    for key in ("snapshot", "verified", "output", "embedding-snapshot"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate"), default="train")
    parser.add_argument("--through", choices=["all", *stages], default="all")
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=1 if method == "wkm" else 4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=900)
    for name in ("env", "train", "vllm"):
        parser.add_argument("--" + name + "-python", default=os.environ.get(name.upper() + "_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if args.workers < 1 or (method == "wkm" and args.gpus != 1):
        parser.error("Positive workers; WKM LoRA requires one training GPU")
    for name in ("snapshot", "verified", "output", "embedding_snapshot"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if Path(args.output).is_relative_to(Path(args.verified)):
        parser.error("Baseline output must be outside the verified support collection")
    batch_invariant()
    config = load_config(args.config)
    args.model_name = config["models"]["student"]["requested_repo"]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with locked(Path(args.output) / ".pipeline.lock"):
        if args.mode == "evaluate":
            from baselines.eto.pipeline import evaluate_pipeline
            os.environ["DECODING"] = "greedy"
            return evaluate_pipeline(args)
        from baselines.common.embeddings import identity as embedding_identity
        effective = {**recipe, "runtime_dependencies": versions(),
            "embedding": embedding_identity(args.embedding_snapshot, "minilm" if method == "wkm" else "bert")}
        identity = prepare(args, config, method=method, recipe=effective, unrestricted_verified_size=True)
        try:
            if args.mode == "train":
                train(args, identity, config)
        finally:
            disclose(args.output, identity)


def require_completed(output, names):
    for name in names:
        if not (Path(output) / "stages" / name / "complete.json").exists():
            raise ValueError("CPU stage requires completed prerequisite: " + name)
