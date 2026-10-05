"""CPU preflight for a separate resumable Stage 4f workspace."""
import argparse
import os
from pathlib import Path

from analysis.mechanism.common import gap_episodes, output_guard, read_json, workspace
from analysis.mechanism.coverage import load_training_states
from analysis.mechanism.policies import validate_continuation
from harness.lora_adapters import adapter_identity, inspect_adapter, parse_adapters
from sed.config import load_config
from sed.storage import locked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source-run", "verified", "adapter", "snapshot", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    source = Path(args.source_run).resolve()
    output_guard(args.output, inputs=[source, args.verified], config=config)
    fce_path, arms_path = source/"fce/fce.json", source/"interventions/interventions.json"
    fce, arms = read_json(fce_path), read_json(arms_path)
    gap_path = fce["gap_path"]
    if Path(arms["fce_path"]).resolve() != fce_path or Path(arms["gap_path"]).resolve() != Path(gap_path).resolve():
        raise ValueError("Source mechanism population/state mismatch")
    _, population = gap_episodes(gap_path, config)
    if not population:
        raise ValueError("Launcher requires a nonempty frozen gap set")
    _, _, inputs = load_training_states(args.verified, config)
    model = config["models"]["student"]
    [(name, path)] = parse_adapters([args.adapter], model["requested_repo"])
    adapter = adapter_identity(inspect_adapter(name, path, model, args.snapshot))
    proof = dict(config, lora_equivalence={"identity": {"adapter": adapter}})
    for row, _, _ in population:
        validate_continuation(proof, {"model": name}, row["episode_path"], verified=args.verified)
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError("Stage 4f requires batch-invariant serving")
    output = workspace(args.output, config=config, command="probe_launcher",
        identity={"adapter": adapter, "model_name": name, "batch_invariant": True, "source_run": str(source)},
        inputs=[args.config, gap_path, fce_path, arms_path, *inputs])
    with locked(output/".allocate.lock"):
        index = 0
        while (output/"launches"/f"attempt-{index:03d}").exists():
            index += 1
        launch = output/"launches"/f"attempt-{index:03d}"
        launch.mkdir(parents=True, exist_ok=False)
    print(launch.resolve())


if __name__ == "__main__":
    main()
