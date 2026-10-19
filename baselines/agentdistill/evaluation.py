"""Read-only MCP box evaluation through the shared dev140 suite."""
import argparse
from pathlib import Path

from baselines.agentdistill.box import load_box
from baselines.agentdistill.gate import validate_report
from baselines.agentdistill.policy import BoxStudent, ToolPolicy
from baselines.agentdistill.teacher import read
from eval.run_suite import frozen_tasks, run_suite
from sed.config import load_config
from sed.manifest import load_frozen, verify_task_files
from sed.runner import _environment, run_episode
from sed.storage import write_json


def worker(job):
    task, config, output, _, spec, inputs = job
    output = Path(output)
    try:
        verify_task_files(config["environment"]["data_root"], task)
        box = load_box(spec["box"], spec["box_sha256"])
        client_spec = {k: v for k, v in spec.items() if k not in {"box", "box_sha256", "decoding", "label"}}
        client = BoxStudent(tools=box["tools"], log_dir=output / "student_calls", **client_spec)
        policy = ToolPolicy(client, box["tools"], output / "mcp_calls")
        episode = run_episode(_environment(config, task), policy, task, config, output,
            policy_condition="agentdistill_different_interface", purpose="evaluation", input_paths=inputs)
        policy.finish(episode)
        write_json(output / "episode.json", episode)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", {"task_id": task["task_id"], "status": "error", "error_type": type(exc).__name__})
    return task["task_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "tasks", "endpoint", "server-provenance", "check", "box", "box-sha256", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    server, report = read(args.server_provenance), read(args.check)
    validate_report(report, server, args.endpoint, args.box, args.box_sha256)
    config["server_provenance"] = server
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    listing, _ = frozen_tasks(args.tasks, manifest, freeze)
    if listing.get("name") != "dev140":
        raise ValueError("AgentDistill evaluation requires frozen dev140")
    spec = {"endpoint": args.endpoint, "model": config["models"]["student"]["requested_repo"],
        "revision": config["models"]["student"]["revision"], "enable_thinking": False,
        "max_tokens": 1024, "decoding": "greedy", "box": args.box, "box_sha256": args.box_sha256,
        "label": "different interface"}
    run_suite(config=config, tasks_path=args.tasks, output=args.output, spec=spec,
        workers=args.workers, worker_fn=worker,
        inputs=[args.config, args.server_provenance, args.check, args.box])
    summary = read(Path(args.output) / "summary.json")
    summary.update(method="agentdistill", label="different interface", training_exposure=0,
                   policy_condition="agentdistill_different_interface")
    write_json(Path(args.output) / "summary.json", summary)


if __name__ == "__main__":
    main()
