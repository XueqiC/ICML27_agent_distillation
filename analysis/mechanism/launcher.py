"""CPU preflight and owned, resumable workspace allocation for rai_mechanism.sh."""
import argparse
from pathlib import Path

from analysis.mechanism.common import gap_episodes, workspace
from analysis.mechanism.examples import load_seeds
from analysis.mechanism.policies import validate_continuation
from harness.lora_adapters import adapter_identity, inspect_adapter, parse_adapters
from sed.config import load_config
from sed.storage import locked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gap-set", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-teacher-calls", type=int, required=True)
    parser.add_argument("--max-teacher-calls-run", type=int, required=True)
    parser.add_argument("--include-base", choices=["0", "1"], default="0")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if min(args.max_teacher_calls, args.max_teacher_calls_run) <= 0:
        parser.error("Positive explicit teacher caps required")
    _, population = gap_episodes(args.gap_set, config)
    if not population:
        raise ValueError("Launcher requires a nonempty frozen gap set")
    load_seeds(args.verified, config)
    model = config["models"]["student"]
    [(name, path)] = parse_adapters([args.adapter], model["requested_repo"])
    adapter = inspect_adapter(name, path, model, args.snapshot)
    proof = dict(config, lora_equivalence={"identity": {"adapter": adapter_identity(adapter)}})
    for row, _, _ in population:
        validate_continuation(proof, {"model": name}, row["episode_path"], verified=args.verified)
    identity = {"adapter": adapter_identity(adapter), "model_name": name,
                "max_episode": args.max_teacher_calls, "max_run": args.max_teacher_calls_run,
                "include_base": args.include_base}
    output = workspace(args.output, config=config, command="launcher", identity=identity,
                       inputs=[args.config, args.gap_set, Path(args.verified)/"manifest.json"])
    with locked(output/".allocate.lock"):
        index = 0
        while (output/"launches"/f"attempt-{index:03d}").exists():
            index += 1
        launch = output/"launches"/f"attempt-{index:03d}"
        launch.mkdir(parents=True, exist_ok=False)
    print(launch.resolve())


if __name__ == "__main__":
    main()
