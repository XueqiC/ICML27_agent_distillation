"""Canonical, opt-in Kang LoRA controls (no torch or model imports)."""
import argparse
import re

from train.modeling import LORA_TARGETS

OPTIONS = ("lora_rank", "lora_alpha", "lora_targets", "lora_layers", "lora_modules", "checkpoint_updates")


def parse_targets(value):
    items = value.split(",") if isinstance(value, str) else list(value)
    if not items or not set(items).issubset(LORA_TARGETS):
        raise ValueError("--lora-targets must be a nonempty subset of " + ",".join(LORA_TARGETS))
    return [target for target in LORA_TARGETS if target in items]


def parse_layers(value):
    if not isinstance(value, str):
        items = list(value)
        if not items or any(type(i) is not int or i < 0 for i in items):
            raise ValueError("layers must be nonnegative integer indices")
        return sorted(set(items))
    layers = set()
    for item in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise ValueError("layers must be comma-separated inclusive ranges, e.g. 0-13,20")
        ends = [int(i) for i in item.split("-")]
        start, stop = ends[0], ends[-1]
        if stop < start:
            raise ValueError("layers range is reversed: " + item)
        layers.update(range(start, stop + 1))
    return sorted(layers)


def parse_modules(value):
    items = value.split(",") if isinstance(value, str) else list(value)
    if not items or any(not re.fullmatch(r"\d+\.\w+", item) or item.split(".")[1] not in LORA_TARGETS
                        for item in items):
        raise ValueError("--lora-modules must contain <layer>.<target>, e.g. 22.down_proj")
    return sorted({f"{int(item.split('.')[0])}.{item.split('.')[1]}" for item in items},
                  key=lambda item: (int(item.split('.')[0]), LORA_TARGETS.index(item.split('.')[1])))


def validate_updates(updates, total=None):
    if updates is not None and (not updates or any(type(n) is not int or n < 1 for n in updates)
                               or any(a >= b for a, b in zip(updates, updates[1:]))):
        raise ValueError("--checkpoint-updates must be strictly increasing positive integers")
    if updates and total is not None and updates[-1] > total:
        raise ValueError(f"--checkpoint-updates requests update {updates[-1]} beyond last update {total}")


def add_arguments(parser):
    # Suppression is essential: vars(args) is sealed and recorded verbatim.
    for name in ("lora-rank", "lora-alpha"):
        parser.add_argument("--" + name, type=int, default=argparse.SUPPRESS)
    for name in ("lora-targets", "lora-layers", "lora-modules"):
        parser.add_argument("--" + name, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-updates", type=int, nargs="+", default=argparse.SUPPRESS)


def normalize(args):
    """Validate once at the CLI boundary; leave absent options absent."""
    rank = getattr(args, "lora_rank", None)
    if rank is not None and not 1 <= rank <= 256:
        raise ValueError("--lora-rank must be in 1..256")
    if rank is not None and getattr(args, "lora_alpha", None) is None:
        args.lora_alpha = 2 * rank
    if getattr(args, "lora_alpha", None) is not None and args.lora_alpha < 1:
        raise ValueError("--lora-alpha must be positive")
    if getattr(args, "lora_modules", None) is not None and any(
            getattr(args, name, None) is not None for name in ("lora_targets", "lora_layers")):
        raise ValueError("--lora-modules is mutually exclusive with --lora-targets and --lora-layers")
    for name, parse in (("lora_targets", parse_targets), ("lora_layers", parse_layers), ("lora_modules", parse_modules)):
        if getattr(args, name, None) is not None:
            setattr(args, name, parse(getattr(args, name)))
    validate_updates(getattr(args, "checkpoint_updates", None))


def shape(args):
    rank = getattr(args, "lora_rank", None)
    alpha = getattr(args, "lora_alpha", None)
    return {"rank": 64 if rank is None else rank,
            "alpha": (128 if rank is None else 2 * rank) if alpha is None else alpha,
            "targets": getattr(args, "lora_targets", None),
            "layers": getattr(args, "lora_layers", None), "modules": getattr(args, "lora_modules", None)}


def recipe_overrides(args):
    if not any(getattr(args, name, None) is not None for name in OPTIONS):
        return {}
    actual = shape(args)
    targets = actual["targets"] or LORA_TARGETS
    if actual["modules"] is not None:
        targets = [t for t in LORA_TARGETS if any(m.endswith("." + t) for m in actual["modules"])]
    result = {"rank": actual["rank"], "alpha": actual["alpha"], "targets": targets}
    for name in ("lora_layers", "lora_modules", "checkpoint_updates"):
        if getattr(args, name, None) is not None:
            result[name] = getattr(args, name)
    return result


def command_options(args):
    result = []
    for name in OPTIONS:
        value = getattr(args, name, None)
        if value is not None:
            result.append("--" + name.replace("_", "-"))
            result.extend(map(str, value) if name == "checkpoint_updates" else
                          [",".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)])
    return result
