"""CPU algebra for PEFT LoRA adapters; never materializes dense updates.

Run with python -m tools.wsp_adapter_ops --help. Computation is float64;
adapter factors are written as float32. A zero update has kept share 1 when
both input and output are zero, and undefined (null) overlap. Top-k subspaces
use min(k, rank(A), rank(B)) nonzero directions, reported as effective_k.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sed.seed_evidence import artifact_hashes
from sed.storage import digest


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def unused(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"Refusing to overwrite existing output: {path}")
    return path


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def module_info(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return (match[1] if match else "unknown"), name.rsplit(".", 1)[-1]


class Adapter:
    def __init__(self, path, *, require_provenance=False):
        self.path = Path(path).resolve()
        self.weights = self.path / "adapter_model.safetensors"
        self.config = json.loads((self.path / "adapter_config.json").read_text())
        if require_provenance:
            self.verify_source()
        for field in ("rank_pattern", "alpha_pattern", "use_dora", "lora_bias"):
            if self.config.get(field):
                raise ValueError(f"Unsupported adapter field: {field}")
        if self.config.get("bias", "none") != "none":
            raise ValueError("Unsupported adapter bias: expected bias='none'")
        if self.config.get("peft_type", "LORA") != "LORA":
            raise ValueError("Expected a PEFT LORA adapter")
        self.rank = self.config.get("r")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("Adapter r must be a positive integer")
        alpha = self.config.get("lora_alpha")
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
            raise ValueError("Adapter lora_alpha must be finite")
        self.scale = alpha / (math.sqrt(self.rank) if self.config.get("use_rslora", False) else self.rank)
        self.keys = {}
        self.shapes = {}
        with safe_open(self.weights, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                match = re.fullmatch(r"(.+)\.lora_([AB])\.weight", key)
                if not match:
                    raise ValueError(f"Unsupported tensor key (only LoRA A/B weights allowed): {key}")
                name, part = match.groups()
                self.keys.setdefault(name, {})[part] = key
            if not self.keys:
                raise ValueError("Adapter has no LoRA modules")
            for name, keys in self.keys.items():
                if set(keys) != {"A", "B"}:
                    raise ValueError(f"Missing LoRA A/B pair: {name}")
                a, b = (handle.get_slice(keys[p]).get_shape() for p in ("A", "B"))
                if len(a) != 2 or len(b) != 2 or a[0] != self.rank or b[1] != self.rank or min(*a, *b) < 1:
                    raise ValueError(f"Invalid factor dimensions/r for module: {name}")
                self.shapes[name] = (b[0], a[1])

    def verify_source(self):
        """Bind derivation to recorded artifacts; never invent training evidence."""
        state_path = self.path / "training_state.json"
        provenance_path = self.path.parent / "provenance.json"
        for path in (state_path, provenance_path):
            if not path.is_file():
                raise ValueError(f"Unverifiable source adapter {self.path}: missing {path.name}")
        try:
            state = json.loads(state_path.read_text())
            self.provenance = json.loads(provenance_path.read_text())
            identity = digest(self.provenance)
            if state["identity_sha256"] != identity:
                raise ValueError(f"Source checkpoint provenance mismatch: {self.path}")
            recorded = state["artifact_sha256"]
            if not isinstance(recorded, dict):
                raise ValueError(f"Invalid source artifact_sha256: {self.path}")
            hashes = artifact_hashes(self.path)
            if (not {"adapter_config.json", "adapter_model.safetensors"} <= recorded.keys()
                    or any(key not in hashes or hashes[key] != value for key, value in recorded.items())):
                raise ValueError(f"Source checkpoint artifacts changed after training: {self.path}")
            metadata = {key: state[key] for key in ("supervised_tokens", "passes", "packing")}
            config = self.provenance["config"]
            student, snapshot = config["models"]["student"], config["sft"]["base_snapshot"]
            self.base_identity = {key: student[key] for key in ("requested_repo", "revision")}
            self.snapshot_hashes = snapshot["metadata_sha256"]
            if not isinstance(self.snapshot_hashes, dict) or not self.snapshot_hashes:
                raise ValueError(f"Source has no base snapshot metadata hashes: {self.path}")
            base = self.config.get("base_model_name_or_path", self.config.get("base_model"))
            if (base not in {student["requested_repo"], snapshot["path"]}
                    or self.config.get("base_model") not in (None, base, student["requested_repo"])
                    or self.config.get("revision") not in (None, student["revision"])):
                raise ValueError(f"Source adapter base_model/revision conflicts with provenance: {self.path}")
            if self.config.get("peft_type") != "LORA" or self.config.get("modules_to_save"):
                raise ValueError(f"Unsupported source adapter configuration: {self.path}")
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid source training_state.json/provenance.json schema: {self.path}") from error
        self.source_metadata = {**self.receipt(), "training_state_sha256": sha256(state_path),
                                "provenance_sha256": identity, "checkpoint_metadata": metadata}

    def factors(self, name):
        with safe_open(self.weights, framework="pt", device="cpu") as handle:
            a, b = (handle.get_tensor(self.keys[name][p]).to(dtype=torch.float64) for p in ("A", "B"))
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Nonfinite adapter factors: {name}")
        return b * self.scale, a

    def receipt(self):
        return {"adapter": str(self.path.resolve()), "adapter_model_sha256": sha256(self.weights),
                "adapter_config_sha256": sha256(self.path / "adapter_config.json"),
                "base_model_name_or_path": self.config.get("base_model_name_or_path")}


def thin_svd(b, a):
    """B A = Qb (Rb Ra^T) Qa^T; the SVD operates only on the rank-sized core."""
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    if not torch.isfinite(s).all():
        raise ValueError("Nonfinite singular values")
    return qb @ u, s, vh @ qa.T


def norm2(s):
    return float(s.square().sum())


def share(kept, total):
    return kept / total if total else 1.0


def compressed_factors(decomposition, rank, keep=True):
    u, s, vh = decomposition
    b = torch.zeros((u.shape[0], rank), dtype=torch.float64, device="cpu")
    a = torch.zeros((rank, vh.shape[1]), dtype=torch.float64, device="cpu")
    n = min(rank, len(s)) if keep else 0
    b[:, :n] = u[:, :n] * s[:n]
    a[:n] = vh[:n]
    return b, a, norm2(s[:n])


def save_adapter(output, adapters, rank, factors, report):
    output = unused(output)
    adapter = adapters[0]
    tensors = {}
    for name, (b, a) in factors.items():
        for part, value in (("A", a), ("B", b)):
            value = value.to(dtype=torch.float32).contiguous()
            if not torch.isfinite(value).all():
                raise ValueError(f"Output factor overflows float32: {name}")
            tensors[adapter.keys[name][part]] = value
    config = {**adapter.config, "r": rank, "lora_alpha": rank, "use_rslora": False}
    sources = [a.source_metadata for a in adapters]
    provenance = deepcopy(adapter.provenance)
    derivation = {"operation": report["operation"],
                  "parameters": {**report["parameters"], "rank": rank, "lora_alpha": rank},
                  "tool_sha256": sha256(__file__), "inputs": sources}
    if "wsp_derived" in provenance:
        derivation["parent"] = provenance["wsp_derived"]
    provenance["wsp_derived"] = derivation
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoint"
    checkpoint.mkdir()
    save_file(tensors, checkpoint / "adapter_model.safetensors", metadata={"format": "pt"})
    write_json(checkpoint / "adapter_config.json", config)
    write_json(checkpoint / "wsp_ops.json", report)
    write_json(output / "provenance.json", provenance)
    state = {"final": True, "wsp_derived": True, "identity_sha256": digest(provenance),
             "artifact_sha256": artifact_hashes(checkpoint),
             **sources[0]["checkpoint_metadata"], "wsp_sources": sources}
    write_json(checkpoint / "training_state.json", state)
    return report


def operation_report(operation, parameters, adapters, modules):
    total = sum(row["input_squared_frobenius_norm"] for row in modules.values())
    kept = sum(row["output_squared_frobenius_norm"] for row in modules.values())
    return {"operation": operation, "parameters": parameters, "inputs": [a.receipt() for a in adapters],
            "modules": modules, "input_squared_frobenius_norm": total,
            "output_squared_frobenius_norm": kept, "total_kept_share": share(kept, total)}


def kept_record(total, kept):
    return {"input_squared_frobenius_norm": total, "output_squared_frobenius_norm": kept,
            "kept_share": share(kept, total)}


def spectrum(adapter, out):
    out = unused(out)
    adapter = Adapter(adapter)
    modules = {}
    layers, targets = defaultdict(float), defaultdict(float)
    for name in sorted(adapter.keys):
        _, s, _ = thin_svd(*adapter.factors(name))
        total = norm2(s)
        layer, target = module_info(name)
        layers[layer] += total
        targets[target] += total
        modules[name] = {"singular_values": s.tolist(), "squared_frobenius_norm": total,
                         "top_k_share": {str(k): share(norm2(s[:k]), total) for k in (1, 2, 4, 8, 16, 32)},
                         "stable_rank": total / float(s[0].square()) if total else 0.0}
    total = sum(layers.values())
    ordered = sorted((v["squared_frobenius_norm"] for v in modules.values()), reverse=True)
    counts = {}
    for percentage in (25, 50, 75, 90):
        accumulated, count = 0., 0
        while accumulated < total * percentage / 100:
            accumulated += ordered[count]
            count += 1
        counts[str(percentage)] = count
    report = {"operation": "spectrum", "inputs": [adapter.receipt()], "modules": modules,
              "summary": {"module_count": len(modules), "squared_frobenius_norm": total,
                          "share_by_layer": {k: v / total if total else 0. for k, v in layers.items()},
                          "share_by_module_type": {k: v / total if total else 0. for k, v in targets.items()},
                          "modules_to_reach_percent": counts}}
    write_json(out, report)
    return report


def positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def parse_layers(value):
    if value is None:
        return None
    layers = set()
    for item in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise ValueError("--layers must be inclusive ranges, e.g. 0-13,20")
        ends = list(map(int, item.split("-")))
        if ends[0] > ends[-1]:
            raise ValueError("--layers contains a reversed range")
        layers.update(range(ends[0], ends[-1] + 1))
    return layers


def truncate(adapter, out, *, rank=None, top_modules=None, targets=None, layers=None):
    unused(out)
    adapter = Adapter(adapter, require_provenance=True)
    rank = adapter.rank if rank is None else rank
    positive(rank, "--rank")
    if top_modules is not None:
        positive(top_modules, "--top-modules")
    target_set = set(targets.split(",")) if targets is not None else None
    layer_set = parse_layers(layers)
    # First pass retains only scalar norms; do not hold all float64 SVD factors.
    norms = {name: norm2(thin_svd(*adapter.factors(name))[1]) for name in sorted(adapter.keys)}
    candidates = [name for name in norms if
                  (target_set is None or module_info(name)[1] in target_set) and
                  (layer_set is None or module_info(name)[0] in {str(i) for i in layer_set})]
    selected = set(sorted(candidates, key=lambda name: (-norms[name], name))[:top_modules])
    factors, records = {}, {}
    for name in norms:
        if name in selected:
            b, a, kept = compressed_factors(thin_svd(*adapter.factors(name)), rank)
        else:
            d_out, d_in = adapter.shapes[name]
            b = torch.zeros(d_out, rank, dtype=torch.float64, device="cpu")
            a, kept = torch.zeros(rank, d_in, dtype=torch.float64, device="cpu"), 0.
        factors[name] = (b, a)
        records[name] = {**kept_record(norms[name], kept), "selected": name in selected}
    report = operation_report("truncate", {"rank": rank, "top_modules": top_modules, "targets": targets,
                                           "layers": layers}, [adapter], records)
    return save_adapter(out, [adapter], rank, factors, report)


def compatible(adapters):
    first = adapters[0]
    for other in adapters[1:]:
        # Cluster-local paths are not model identity; gated operations check provenance.
        if other.shapes != first.shapes:
            raise ValueError("Adapters must have the same module set and dimensions")


def average(adapters, out, *, rank, weights=None):
    unused(out)
    positive(rank, "--rank")
    if not adapters:
        raise ValueError("At least one adapter is required")
    adapters = [Adapter(path, require_provenance=True) for path in adapters]
    for other in adapters[1:]:
        if other.base_identity != adapters[0].base_identity:
            raise ValueError("Average sources must have the same student requested_repo and revision")
        if other.snapshot_hashes != adapters[0].snapshot_hashes:
            raise ValueError("Average sources must have the same base_snapshot.metadata_sha256")
    compatible(adapters)
    weights = [1 / len(adapters)] * len(adapters) if weights is None else weights
    if len(weights) != len(adapters) or any(not math.isfinite(w) for w in weights):
        raise ValueError("--weights needs one finite weight per adapter (used without normalization)")
    factors, records = {}, {}
    for name in sorted(adapters[0].keys):
        parts = [adapter.factors(name) for adapter in adapters]
        decomposition = thin_svd(torch.cat([b * w for (b, a), w in zip(parts, weights)], dim=1),
                                 torch.cat([a for b, a in parts], dim=0))
        b, a, kept = compressed_factors(decomposition, rank)
        factors[name] = (b, a)
        records[name] = kept_record(norm2(decomposition[1]), kept)
    report = operation_report("average", {"rank": rank, "weights": weights}, adapters, records)
    return save_adapter(out, adapters, rank, factors, report)


def scale(adapter, out, *, factor):
    unused(out)
    if not math.isfinite(factor):
        raise ValueError("--factor must be finite")
    adapter = Adapter(adapter, require_provenance=True)
    factors, records = {}, {}
    for name in sorted(adapter.keys):
        b, a = adapter.factors(name)
        total = norm2(thin_svd(b, a)[1])
        factors[name] = (b * factor, a if factor else torch.zeros_like(a))
        records[name] = kept_record(total, total * factor ** 2)
    report = operation_report("scale", {"factor": factor}, [adapter], records)
    return save_adapter(out, [adapter], adapter.rank, factors, report)


def numerical_rank(s, dimensions):
    return int((s > torch.finfo(s.dtype).eps * max(dimensions) * s[0]).sum())


def overlap(adapter_a, adapter_b, out, *, k=(1, 4, 16)):
    out = unused(out)
    if not k:
        raise ValueError("--k needs at least one positive integer")
    for n in k:
        positive(n, "--k")
    adapters = [Adapter(adapter_a), Adapter(adapter_b)]
    compatible(adapters)
    modules = {}
    for name in sorted(adapters[0].keys):
        (b1, a1), (b2, a2) = [adapter.factors(name) for adapter in adapters]
        u1, s1, v1 = thin_svd(b1, a1)
        u2, s2, v2 = thin_svd(b2, a2)
        n1, n2 = norm2(s1), norm2(s2)
        # tr(A1^T B1^T B2 A2), with both contractions rank-sized.
        inner = float(((b1.T @ b2) * (a1 @ a2.T)).sum())
        cosine = max(-1., min(1., inner / math.sqrt(n1 * n2))) if n1 and n2 else None
        d_out, d_in = adapters[0].shapes[name]
        rank = min(numerical_rank(s1, (d_out, d_in)), numerical_rank(s2, (d_out, d_in)))
        subspaces = {}
        for requested in k:
            n = min(requested, rank)
            subspaces[str(requested)] = {"effective_k": n,
                "left": min(1., float((u1[:, :n].T @ u2[:, :n]).square().sum()) / n) if n else None,
                "right": min(1., float((v1[:n] @ v2[:n].T).square().sum()) / n) if n else None,
                "chance_left": n / d_out, "chance_right": n / d_in}
        modules[name] = {"frobenius_cosine": cosine, "subspaces": subspaces,
                         "squared_frobenius_norm_a": n1, "squared_frobenius_norm_b": n2}

    def summary(names, weighted=False):
        def mean(getter):
            pairs = [(getter(modules[name]), (modules[name]["squared_frobenius_norm_a"] +
                      modules[name]["squared_frobenius_norm_b"]) / 2 if weighted else 1.) for name in names]
            pairs = [(v, w) for v, w in pairs if v is not None]
            weight = sum(w for v, w in pairs)
            return sum(v * w for v, w in pairs) / weight if weight else None
        return {"frobenius_cosine": mean(lambda row: row["frobenius_cosine"]),
                "subspaces": {str(n): {side: mean(lambda row: row["subspaces"][str(n)][side])
                              for side in ("left", "right", "chance_left", "chance_right")} for n in k}}

    by_type = defaultdict(list)
    for name in modules:
        by_type[module_info(name)[1]].append(name)
    report = {"operation": "overlap", "parameters": {"k": list(k)},
              "inputs": [a.receipt() for a in adapters], "modules": modules,
              "summary": {"mean": summary(modules), "squared_norm_weighted_mean": summary(modules, True),
                  "weight_definition": "(squared_frobenius_norm_a + squared_frobenius_norm_b) / 2",
                  "by_module_type": {target: {"mean": summary(names), "squared_norm_weighted_mean": summary(names, True)}
                                     for target, names in by_type.items()}}}
    write_json(out, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("spectrum", "truncate", "average", "scale", "overlap"):
        command = commands.add_parser(name)
        command.add_argument("adapters" if name in {"average", "overlap"} else "adapter",
                             **({"nargs": "+"} if name == "average" else {"nargs": 2} if name == "overlap" else {}))
        command.add_argument("--out", required=True)
        if name in {"truncate", "average"}:
            command.add_argument("--rank", type=int, required=name == "average")
        if name == "truncate":
            command.add_argument("--top-modules", type=int)
            command.add_argument("--targets")
            command.add_argument("--layers")
        if name == "average":
            command.add_argument("--weights", type=float, nargs="+")
        if name == "scale":
            command.add_argument("--factor", type=float, required=True)
        if name == "overlap":
            command.add_argument("--k", type=int, nargs="+", default=[1, 4, 16])
    args = vars(parser.parse_args(argv))
    operation = args.pop("operation")
    try:
        if operation == "overlap":
            a, b = args.pop("adapters")
            overlap(a, b, **args)
        else:
            globals()[operation](**args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
