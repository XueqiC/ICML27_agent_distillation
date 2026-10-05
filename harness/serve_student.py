"""Supervise vLLM on a SLURM allocation; never import torch/vLLM in the launcher."""

import argparse
import json
import math
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
from harness.lora_adapters import parse_adapters, inspect_adapter, convert_adapter


def job_port(job_id):
    if not str(job_id).isdigit():
        raise ValueError("SLURM_JOB_ID must be numeric, or pass --port")
    return 20000 + int(job_id) % 20000


def free_port():
    """Ask the kernel, rather than guessing a shared host's free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def inspect_runtime(python):
    # Metadata and source only; safe without CUDA or importing vLLM/transformers.
    code = '''import importlib.metadata as m, json, hashlib
from pathlib import Path
v = m.distribution("vllm")
p = Path(v.locate_file("vllm/reasoning/__init__.py"))
sources = ["model_executor/models/gemma4.py", "model_executor/models/gemma4_mm.py",
           "model_executor/models/gemma4_unified.py", "lora/utils.py",
           "lora/peft_helper.py", "lora/lora_model.py", "lora/worker_manager.py",
           "entrypoints/openai/completion/protocol.py", "entrypoints/openai/completion/serving.py"]
hashes = {name: hashlib.sha256(Path(v.locate_file("vllm/" + name)).read_bytes()).hexdigest()
          for name in sources if Path(v.locate_file("vllm/" + name)).is_file()}
print(json.dumps({"vllm": v.version, "transformers": m.version("transformers"),
                  "gemma4_parser": '"gemma4"' in p.read_text(),
                  "parser_registry_path": str(p), "serving_source_sha256": hashes}))
'''
    result = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Cannot inspect server runtime metadata")
    return json.loads(result.stdout)


def server_command(args, student, runtime, adapters=()):
    model = Path(args.merged_model_path or args.model_path).expanduser().resolve()
    if not model.is_dir():
        raise ValueError("MODEL_PATH must be an existing local checkpoint (no download)")
    memory = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
    if not math.isfinite(float(memory)) or not 0 < float(memory) <= 1:
        raise ValueError("VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    command = [args.python, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
               "--host", "127.0.0.1", "--port", str(args.port),
               "--served-model-name", student["requested_repo"],
               "--revision", student["revision"], "--tokenizer-revision", student["revision"],
               "--max-model-len", "32768", "--gpu-memory-utilization", memory,
               "--generation-config", "vllm", "--seed", "0"]
    if runtime["gemma4_parser"]:
        command += ["--reasoning-parser", "gemma4"]
    if adapters:
        if args.merged_model_path:
            raise ValueError("LoRA adapters require the unmerged base snapshot")
        command += ["--enable-lora", "--max-lora-rank", str(max(a["rank"] for a in adapters)),
                    "--max-loras", str(len(adapters)), "--lora-modules"]
        command += [a["name"] + "=" + a["served_path"] for a in adapters]
    return command


def owns_listener(pid, port, *, proc=Path("/proc")):
    """Linux: a readiness response must come from our child process group.

    Binding port 0 and then closing it has an unavoidable handoff window to
    vLLM. If another job wins that race, never accept its /v1/models response.
    """
    inodes = set()
    for row in (proc / "net/tcp").read_text().splitlines()[1:]:
        fields = row.split()
        if fields[1] == f"0100007F:{port:04X}" and fields[3] == "0A":
            inodes.add("socket:[" + fields[9] + "]")
    if not inodes:
        return False
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if os.getpgid(int(directory.name)) != pid:
                continue
            for fd in (directory / "fd").iterdir():
                try:
                    if os.readlink(fd) in inodes:
                        return True
                except OSError:
                    pass
        except (OSError, ProcessLookupError):
            continue
    return False


def wait_ready(process, port, timeout, model, *, opener=None, require_owner=False):
    opener = opener or build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before readiness; inspect server.log")
        try:
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                data = json.load(response)
            expected = {model} if isinstance(model, str) else set(model)
            if (expected <= {entry.get("id") for entry in data["data"]} and process.poll() is None
                    and (not require_owner or owns_listener(process.pid, port))):
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
    parser.add_argument("--merged-model-path", help="Standalone HF checkpoint (merged LoRA or full fine-tuning)")
    parser.add_argument("--lora-adapter", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--lora-conversion", choices=("none", "targets", "gemma4"), default="none",
                        help="Explicit fallback after a naming rejection; writes only under server output")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT", "0")
    if batch_invariant not in {"0", "1"}:
        parser.error("VLLM_BATCH_INVARIANT must be 0 or 1")
    if args.port is None:
        args.port = job_port(os.environ.get("SLURM_JOB_ID", ""))
    elif args.port == 0:
        args.port = free_port()
    if not 1024 <= args.port <= 65535 or args.ready_timeout <= 0:
        parser.error("Invalid port/readiness timeout")
    config = load_config(args.config)
    student = config["models"]["student"]
    parsed = parse_adapters(args.lora_adapter, student["requested_repo"])
    if parsed and args.merged_model_path:
        parser.error("--lora-adapter cannot be combined with --merged-model-path")
    adapters = [inspect_adapter(name, path, student, args.model_path) for name, path in parsed]
    runtime = inspect_runtime(args.python)
    # Do not mistake another job's server for this job's ready child.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", args.port))
    model = Path(args.merged_model_path or args.model_path).expanduser().resolve()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.lora_conversion != "none":
        adapters = [convert_adapter(a, output / "adapters" / a["name"], args.lora_conversion)
                    for a in adapters]
    command = server_command(args, student, runtime, adapters)
    # Record local config/template/index hashes; weights remain the user's pinned
    # local snapshot. We do not imply that --revision verifies local weight bytes.
    model_files = {str(p.relative_to(model)): file_hash(p) for p in model.iterdir()
                   if p.is_file() and p.suffix in {".json", ".jinja", ".jinja2"}}
    checkpoint_kind = "merged_lora" if args.merged_model_path else "base"
    training_provenance = None
    if args.merged_model_path and (model / "training_state.json").exists():
        state = json.loads((model / "training_state.json").read_text())
        if state.get("text_export_dtype") == "bfloat16":
            from baselines.common.fsdp_train import validate_checkpoint
            validate_checkpoint(model, student)
            checkpoint_kind = "full_parameter"
            training_provenance = file_hash(model.parent / "provenance.json")
    config["server"] = {"command": command, "runtime": runtime,
                        "batch_invariant": batch_invariant == "1",
                        "port": args.port, "endpoint": f"http://127.0.0.1:{args.port}/v1",
                        "readiness_identity": "all model IDs plus listening socket owned by child process group",
                        "served_model_name": student["requested_repo"], "lora_adapters": adapters,
                        "model_path": str(model), "merged_model": bool(args.merged_model_path),
                        "checkpoint_kind": checkpoint_kind,
                        "training_provenance_sha256": training_provenance,
                        "model_metadata_sha256": model_files,
                        "model_revision_verification": (
                            "training_artifact_hashes; original_HF_base_weight_bytes_not_certified"
                            if checkpoint_kind == "full_parameter" else
                            "local_path_and_metadata; weight_bytes_not_verified"),
                        "environment": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                                        "VLLM_BATCH_INVARIANT": batch_invariant,
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
            models = [student["requested_repo"], *[a["name"] for a in adapters]]
            wait_ready(process, args.port, args.ready_timeout, models, require_owner=True)
            write_json(ready_file, {"endpoint": f"http://127.0.0.1:{args.port}/v1",
                                   "pid": process.pid, "port": args.port, "models": models}, exclusive=True)
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
