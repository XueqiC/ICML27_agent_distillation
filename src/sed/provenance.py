"""Explicit run provenance without dumping environment variables or secrets."""

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from sed.storage import file_hash, write_json
from sed.hash_receipts import cache_provenance
from sed.startup_timing import timing_provenance


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def versions():
    result = {}
    for name in ("sufficient-experience-distillation", "alfworld", "textworld",
                 "openai", "httpx", "vllm", "transformers", "torch", "peft", "accelerate",
                 "tokenizers", "PyYAML", "pytest", "numpy", "fast-downward"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def code_hashes(root):
    root = Path(root)
    files = []
    for folder in ("analysis/mechanism", "baselines", "src", "harness", "teachers", "students", "train", "eval", "tools", "tests", "configs", "scripts"):
        files.extend(p for p in (root / folder).rglob("*")
                     if p.suffix in {".py", ".yaml", ".env", ".slurm", ".sh"}
                     or (folder == "baselines" and p.suffix == ".md"))
    files.extend(root / p for p in ("AGENTS.md", "pyproject.toml", "requirements.txt",
                                    "analysis/serving.py", "analysis/k_ladder.py",
                                    "requirements-alfworld.txt", "data/manifest/build.py",
                                    "data/manifest/build_dev40.py", "data/manifest/build_dev140.py",
                                    "data/seeds/select_seeds.py", "docs/MECHANISM_PLAN.md"))
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(files) if p.is_file()}


def safe_config(value):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in {
            "api_key", "password", "secret", "authorization", "access_token"
        } else safe_config(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_config(v) for v in value]
    return value


def write_provenance(directory, *, command, config, inputs=None, details=None, run_id=None):
    root = Path(__file__).resolve().parents[2]
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_head = None
    row = {"run_id": run_id or str(uuid.uuid4()), "timestamp": utc_now(),
           "command": command, "python_executable": sys.executable,
           "python_version": platform.python_version(), "dependencies": versions(),
           "git_head": git_head, "code_sha256": code_hashes(root),
           "hash_cache": cache_provenance(),
           "config": safe_config(config),
           "inputs": {str(p): file_hash(p) for p in (inputs or [])},
           "details": details or {}}
    if timing_provenance() is not None:
        row["startup_timing"] = timing_provenance()
    write_json(Path(directory) / "provenance.json", row, exclusive=True)
    return row
