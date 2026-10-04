"""Small, explicit configuration loader; no implicit machine paths."""

import os
from pathlib import Path
import re

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value):
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        def replace(match):
            name, default = match.groups()
            if name not in os.environ and default is None:
                raise ValueError(f"Missing configuration environment variable: {name}")
            return os.environ.get(name) or default or ""
        return os.path.expanduser(_ENV.sub(replace, value))
    return value


def load_config(path="configs/base.yaml"):
    config = _expand(yaml.safe_load(Path(path).read_text()))
    if config["models"]["teacher"]["access"] != "black_box_outputs_only":
        raise ValueError("Only black-box teacher outputs are allowed")
    if config["training"]["mode"] != "post_training_only":
        raise ValueError("Only post-training is allowed")
    if config["models"]["checkpoint_origin_policy"] != "no_chinese_checkpoints":
        raise ValueError("Checkpoint origin policy cannot be relaxed")
    if config["evaluation"]["privileged_fields_allowed"]:
        raise ValueError("Privileged model inputs are forbidden")
    return config
