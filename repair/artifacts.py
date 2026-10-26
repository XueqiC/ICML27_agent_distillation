"""Immutable pilot inputs and provenance; evidence is never a model prompt."""
from copy import deepcopy
import json
from pathlib import Path

from sed.provenance import write_provenance
from sed.storage import digest, file_hash, write_json


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def seal(value):
    return {**value, "content_sha256": digest(value)}


def checked_document(path):
    value = read_json(path)
    if value.get("content_sha256") != digest({k: v for k, v in value.items() if k != "content_sha256"}):
        raise ValueError("Artifact content hash mismatch")
    return value


def provenance(output, config, *, command, inputs=(), details=None):
    sources = sorted(Path(__file__).parent.glob("*.py"))
    return write_provenance(output, command=command, config=deepcopy(config),
                            inputs=[*inputs, *sources], details=details)


def run_manifest(prov, *, checkpoint, tokenizer, tasks, seeds, **extra):
    cfg = prov["config"]
    return {"schema_version": 1, "run_id": prov["run_id"], "code_revision": prov["git_head"],
        "code_sha256": prov["code_sha256"], "inputs": prov["inputs"],
        "environment": cfg["environment"], "models": cfg["models"], "config": cfg,
        "dependency_versions": prov["dependencies"], "checkpoint": checkpoint,
        "tokenizer": tokenizer, "task_manifest_sha256": digest(tasks), "tasks": tasks,
        "seeds": seeds, **extra}


def checkpoint_files(path):
    path = Path(path).resolve()
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}


def tokenizer_identity(snapshot, tokenizer):
    files = {p.name: file_hash(p) for p in sorted(Path(snapshot).iterdir()) if p.is_file() and
             p.name.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'chat_template', 'vocab', 'merges'))}
    return {'id': digest(files), 'files': files, 'chat_template_sha256': digest(tokenizer.chat_template)}


def write_sealed(path, value):
    write_json(path, seal(value), exclusive=True)
