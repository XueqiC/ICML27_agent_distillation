"""Supervise vLLM on a SLURM allocation; never import torch/vLLM in the launcher."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import build_opener, ProxyHandler

from sed.config import load_config
from sed.provenance import write_provenance
from sed.storage import file_hash, write_json


def job_port(job_id):
    if not str(job_id).isdigit():
        raise ValueError("SLURM_JOB_ID must be numeric, or pass --port")
    return 20000 + int(job_id) % 20000


def inspect_runtime(python):
    # Metadata and source only; safe without CUDA or importing vLLM/transformers.
    code = '''import importlib.metadata as m, json
from pathlib import Path
v = m.distribution("vllm")
p = Path(v.locate_file("vllm/reasoning/__init__.py"))
print(json.dumps({"vllm": v.version, "transformers": m.version("transformers"),
                  "gemma4_parser": '"gemma4"' in p.read_text(),
                  "parser_registry_path": str(p)}))
'''
    result = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Cannot inspect server runtime metadata")
    return json.loads(result.stdout)


def server_command(args, student, runtime):
    model = Path(args.merged_model_path or args.model_path).expanduser().resolve()
    if not model.is_dir():
        raise ValueError("MODEL_PATH must be an existing local checkpoint (no download)")
    command = [args.python, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
               "--host", "127.0.0.1", "--port", str(args.port),
               "--served-model-name", student["requested_repo"],
               "--revision", student["revision"], "--tokenizer-revision", student["revision"],
               "--max-model-len", "32768", "--gpu-memory-utilization", "0.85",
               "--generation-config", "vllm", "--seed", "0"]
    if runtime["gemma4_parser"]:
        command += ["--reasoning-parser", "gemma4"]
    return command


def wait_ready(process, port, timeout, model, *, opener=None):
    opener = opener or build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before readiness; inspect server.log")
        try:
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                data = json.load(response)
            if any(entry.get("id") == model for entry in data["data"]):
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(1)
    raise TimeoutError("vLLM readiness timeout")


def stop_server(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--python", default=os.environ.get("VLLM_PYTHON"), required=not os.environ.get("VLLM_PYTHON"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"), required=not os.environ.get("MODEL_PATH"))
    parser.add_argument("--merged-model-path", help="Explicit already-merged LoRA checkpoint for later work")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.port is None:
        args.port = job_port(os.environ.get("SLURM_JOB_ID", ""))
    if not 1024 <= args.port <= 65535 or args.ready_timeout <= 0:
        parser.error("Invalid port/readiness timeout")
    config = load_config(args.config)
    runtime = inspect_runtime(args.python)
    command = server_command(args, config["models"]["student"], runtime)
    # Do not mistake another job's server for this job's ready child.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", args.port))
    model = Path(args.merged_model_path or args.model_path).expanduser().resolve()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Record local config/template/index hashes; weights remain the user's pinned
    # local snapshot. We do not imply that --revision verifies local weight bytes.
    model_files = {str(p.relative_to(model)): file_hash(p) for p in model.iterdir()
                   if p.is_file() and p.suffix in {".json", ".jinja", ".jinja2"}}
    config["server"] = {"command": command, "runtime": runtime,
                        "model_path": str(model), "merged_model": bool(args.merged_model_path),
                        "model_metadata_sha256": model_files,
                        "model_revision_verification": "local_path_and_metadata; weight_bytes_not_verified",
                        "environment": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                                        "VLLM_USE_FLASHINFER_SAMPLER": "0",
                                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "reasoning_parser_registry_sha256": file_hash(runtime["parser_registry_path"])}
    write_provenance(output, command="serve_student", config=config, inputs=[args.config])
    process = None
    ready_file = output / "ready.json"
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    try:
        env = os.environ.copy()  # Inherited privately; never serialized/logged.
        env.update(config["server"]["environment"])
        with (output / "server.log").open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            wait_ready(process, args.port, args.ready_timeout, config["models"]["student"]["requested_repo"])
            write_json(ready_file, {"endpoint": f"http://127.0.0.1:{args.port}/v1",
                                   "pid": process.pid, "port": args.port}, exclusive=True)
            return process.wait()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        write_json(output / "failure.json", {"status": "error", "error_type": type(exc).__name__})
        print("Student server failed; inspect sanitized provenance and server.log")
        return 1
    finally:
        ready_file.unlink(missing_ok=True)
        stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
