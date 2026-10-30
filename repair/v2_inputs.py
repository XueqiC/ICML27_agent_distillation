"""Read-only path relocation and byte verification for the V2 training reader.

Only filesystem references are relocated. Sealed JSON, evidence and hashes are
never edited, and historical provenance is not recursively interpreted as paths.
"""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

from repair.artifacts import read_json
from sed.storage import file_hash

FORBIDDEN_SETS = ("U_dev", "U_mech", "D_probe", "D_unseen_quick", "dev140")


class ValidationError(ValueError):
    """Safe local validation diagnostics (never raw library exceptions)."""


def require(ok, message):
    if not ok:
        raise ValidationError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class Relocation:
    """One component-prefix substitution, followed by symlink confinement.

    Source names are lexical: resolving them on this host before matching would
    lose the sealed identity. Destinations are checked against resolved NEW.
    """
    def __init__(self, dataset, values=()):
        self.rules = []
        self.seen = defaultdict(set)
        for value in values:
            old, separator, new = value.partition("=")
            require(separator and old and new, "Relocation must be OLD=NEW")
            old, new = Path(old), Path(new)
            require(old.is_absolute() and new.is_absolute(), "Relocation prefixes must be absolute directories")
            require(".." not in old.parts and ".." not in new.parts,
                    "Relocation cannot change a relative tail using '..'")
            require(not old.exists() or old.is_dir(), "OLD must be a directory prefix")
            require(not new.exists() or new.is_dir(), "NEW must be a directory prefix")
            require(not any(old.is_relative_to(a) or a.is_relative_to(old) for a, _ in self.rules),
                    "Duplicate or overlapping OLD relocation prefixes")
            self.rules.append((old, new))
        self.rules.sort(key=lambda pair: (-len(pair[0].parts), str(pair[0])))
        original = Path(dataset).absolute()
        self.dataset = self.path(original)
        for _, new in self.rules:
            require(not new.absolute().is_relative_to(original) and
                    not new.resolve().is_relative_to(self.dataset),
                    "NEW must be outside the sealed dataset")

    def path(self, value, *, root=None):
        """Resolve an original reference, applying at most one mapping."""
        path = Path(value)
        if path.is_absolute():
            for old, new in self.rules:
                if path.is_relative_to(old):
                    tail = path.relative_to(old)
                    require(".." not in tail.parts, "Relocation cannot change a path's relative tail")
                    target = new / tail
                    require(target.relative_to(new) == tail, "Relocation changed a path's relative tail")
                    self.seen[str(old)].add(str(path))
                    resolved = self.local(target)
                    require(resolved.is_relative_to(new.resolve()),
                            f"Relocation symlink escape: {json.dumps(str(path))}")
                    return resolved
        return self.local(path if path.is_absolute() or root is None else Path(root) / path)

    def local(self, value):
        """Check a derived local path without performing a second substitution."""
        path = Path(value).absolute()
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            raise ValidationError(f"Cannot resolve input path: {json.dumps(str(path))}") from None
        for _, new in self.rules:
            if path.is_relative_to(new) or path.is_relative_to(new.resolve()):
                require(resolved.is_relative_to(new.resolve()),
                        f"Relocation symlink escape: {json.dumps(str(path))}")
        return resolved

    def report(self):
        return {"map": [{"old": str(old), "new": str(new)} for old, new in self.rules],
                "relocated_paths_per_prefix": {str(old): len(self.seen[str(old)]) for old, _ in self.rules},
                "counting": "distinct original absolute references, including directories; once across bundles"}


class InputFiles:
    """De-duplicated file requirements, with no persistent/stat-based hash cache."""
    def __init__(self, relocation):
        self.relocation = relocation
        self.files = {}
        self.hashes = {}
        self.all_hashes_verified = False

    def add(self, path, sha=None, *, original=None, reason="Interchange input sha256 mismatch"):
        # Keep local aliases in the transfer list: HF snapshot symlinks and
        # their blobs are both needed. Resolve every one before reading bytes.
        path = Path(path).absolute()
        resolved = self.relocation.local(path)
        entry = self.files.setdefault(path, {"sources": set(), "seals": {}})
        entry["sources"].add(str(original if original is not None else path))
        if sha is not None:
            require(_sha(sha), f"Invalid input sha256: {json.dumps(str(path))}")
            entry["seals"][sha] = reason
        if resolved != path:
            target = self.files.setdefault(resolved, {"sources": set(), "seals": {}})
            target["sources"].update(entry["sources"])
            target["seals"].update(entry["seals"])
        self.all_hashes_verified = False
        return path

    def reference(self, value, sha=None, *, root=None, reason="Interchange input sha256 mismatch"):
        return self.add(self.relocation.path(value, root=root), sha, original=value, reason=reason)

    def verify(self):
        self.all_hashes_verified = False
        failures = []
        for path, entry in sorted(self.files.items()):
            try:
                resolved = self.relocation.local(path)
                if not resolved.is_file():
                    failures.append((path, "missing file"))
                    continue
                if entry["seals"]:
                    if resolved not in self.hashes:
                        self.hashes[resolved] = file_hash(resolved)
                    if self.hashes[resolved] not in entry["seals"] or len(entry["seals"]) != 1:
                        failures.append((path, "; ".join(sorted(set(entry["seals"].values())))))
            except (OSError, RuntimeError):
                failures.append((path, "unreadable file"))
        if failures:
            first = "; ".join(f"{json.dumps(str(p))} ({why})" for p, why in failures[:5])
            raise ValidationError(f"Input verification failed: {len(failures)} offending paths; first paths: {first}")
        self.all_hashes_verified = True

    def report(self):
        return {**self.relocation.report(), "all_hashes_verified": self.all_hashes_verified,
                "verified_hashed_files": len({self.relocation.local(p) for p, e in self.files.items()
                                               if e["seals"]}) if self.all_hashes_verified else 0}

    def listing(self):
        groups = defaultdict(list)
        for path, entry in sorted(self.files.items()):
            prefix = next((new.resolve() for _, new in self.relocation.rules
                           if path.is_relative_to(new.resolve())), None)
            if prefix is None:
                prefix = self.relocation.dataset if path.is_relative_to(self.relocation.dataset) else Path(path.anchor)
            groups[str(prefix)].append({"path": str(path), "relative_path": str(path.relative_to(prefix)),
                "source_paths": sorted(entry["sources"]),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sorted(entry["seals"]), "resolved_path": str(self.relocation.local(path))})
        return {"schema": "v2-training-inputs/1", "file_count": len(self.files),
                "relocation": self.report(),
                "groups": [{"prefix": prefix, "files": files} for prefix, files in sorted(groups.items())]}


def training_view_inputs(files, source):
    """Inventory the original K20 view when membership must be read from it.

    Follow only the evidence reader's explicit filesystem fields. This performs
    no rollout, label interpretation, or scientific validation of receipts.
    """
    original = Path(source["config"]["baseline_training"]["verified"])
    sha = source.get("inputs", {}).get(str(original / "manifest.json"))
    require(_sha(sha), "K20 view differs from theta0's original support manifest")
    files.reference(original / "manifest.json", sha,
                    reason="K20 view differs from theta0's original support manifest")
    view = files.relocation.path(original)
    pending, seen = [view], set()
    while pending:
        root = pending.pop()
        if root in seen:
            continue
        seen.add(root)
        for name in ("manifest.json", "selection.json", "identity.json"):
            files.add(root / name)
        # Reading these documents requires their presence even in list mode.
        missing = [p for p in (root / "manifest.json", root / "identity.json") if not p.is_file()]
        require(not missing, f"Missing K20 metadata: {len(missing)} offending paths: "
                + ", ".join(json.dumps(str(p)) for p in missing[:5]))
        collection, identity = read_json(root / "manifest.json"), read_json(root / "identity.json")
        evidence_root = root.parent.parent if "source_collection" in collection else root
        if evidence_root != root:
            pending.append(evidence_root)
        for field in ("support_view", "extension_source"):
            if field in identity:
                parent = files.relocation.path(identity[field]["path"], root=root)
                pending.append(parent)
                if field == "extension_source":
                    # Its recorded tree seal makes every original source file a
                    # dependency, including inherited smaller-K view anchors.
                    for path in sorted(parent.rglob("*")):
                        if (path.is_file() or path.is_symlink()) and not path.name.endswith(".lock"):
                            files.add(path)
        for receipt in collection.get("attempts", []):
            episode = files.relocation.path(receipt["episode_dir"], root=evidence_root)
            require(episode.is_relative_to(evidence_root), "Episode path escapes collection")
            files.add(episode.parent / "receipt.json")
            for directory, key in ((episode, "episode_hashes"), (episode.parent / "teacher", "teacher_hashes")):
                for name, sha in receipt[key].items():
                    require(_sha(sha), "Invalid K20 evidence sha256")
                    files.reference(name, sha, root=directory)
    return view


def inspect_inputs(dataset, *, theta0=None, relocate=(), snapshot=None, config=None,
                   splits=None, inputs=(), check=True, runtime=True):
    """Inventory and verify bytes only; no tokenizer, model or training imports.

    Snapshot defaults to the sealed tokenizer location, then theta0's recorded
    base. Callers can supply the same explicit snapshot/config as a run.
    Historical theta0 training inputs are followed only when the reader can
    reach its original-K20 membership fallback; other history is not opened.
    """
    relocation = relocate if isinstance(relocate, Relocation) else Relocation(dataset, relocate)
    files = InputFiles(relocation)
    root = relocation.dataset
    bundles = []
    directories = [root, *([root / "replay"] if (root / "replay").exists() else [])]
    for directory in directories:
        files.add(directory / "manifest.json")
        files.add(directory / "examples.sha256")
    files.verify()
    for directory in directories:
        manifest = read_json(directory / "manifest.json")
        seal = (directory / "examples.sha256").read_text().strip().split()
        require(len(seal) in (1, 2) and _sha(seal[0]) and
                (len(seal) == 1 or seal[1] in ("examples.jsonl", "*examples.jsonl")), "Invalid examples.sha256")
        files.add(directory / "examples.jsonl", seal[0], reason="examples.jsonl sha256 mismatch")
        require(isinstance(manifest.get("inputs"), dict) and manifest["inputs"], "Missing hashed interchange inputs")
        for name, sha in manifest["inputs"].items():
            require(_sha(sha), f"Invalid input sha256: {json.dumps(name)}")
            files.reference(name, sha, root=directory)
        bundles.append({"root": directory, "manifest": manifest, "examples_sha256": seal[0]})
    if theta0 is None:
        theta0 = bundles[0]["manifest"]["theta0"]["adapter_path"]
    theta0 = relocation.path(theta0)
    for bundle in bundles:
        files.add(theta0 / "adapter_model.safetensors", bundle["manifest"]["theta0"]["adapter_weights_sha256"],
                  reason="theta0 adapter hash differs from dataset manifest")
    local_config = deepcopy(config)
    if local_config is not None:
        # Only these config paths are opened by the training reader. Never
        # traverse arbitrary config strings (in particular secrets_file).
        for key in ("manifest", "split_freeze"):
            local_config["paths"][key] = str(files.reference(config["paths"][key]))
    for path in inputs:
        files.reference(path)
    if splits is not None:
        splits = files.reference(splits)
        for bundle in bundles:
            files.add(splits, bundle["manifest"]["splits_sha256"], reason="Split freeze sha256 mismatch")
    result = {"files": files, "bundles": bundles, "theta0": theta0,
              "config": local_config, "splits": splits}
    if not runtime:
        if check:
            files.verify()
        return result
    for path in (theta0 / "adapter_config.json", theta0 / "training_state.json", theta0.parent / "provenance.json"):
        files.add(path)
    # Validate sealed files before trusting the dependency metadata they contain.
    if check:
        files.verify()
    else:
        missing = [p for p in (theta0 / "training_state.json", theta0.parent / "provenance.json") if not p.is_file()]
        require(not missing, "Input listing requires local theta0 metadata; list on the source host before copying")
    state = read_json(theta0 / "training_state.json")
    source = read_json(theta0.parent / "provenance.json")
    for name, sha in state.get("artifact_sha256", {}).items():
        require(_sha(sha), f"Invalid theta0 artifact sha256: {json.dumps(name)}")
        files.reference(name, sha, root=theta0)
    for path in sorted(theta0.rglob("*")):
        if path.is_file() or path.is_symlink():
            files.add(path)
    if snapshot is None:
        snapshot = bundles[0]["manifest"].get("supervised_token_estimate", {}).get("tokenizer", {}).get("path")
        if snapshot is None:
            snapshot = source["config"]["sft"]["base_snapshot"]["path"]
    snapshot = relocation.path(snapshot)
    files.add(snapshot / "config.json")
    metadata = source["config"]["sft"]["base_snapshot"]["metadata_sha256"]
    for hashes in (metadata, source["config"].get("baseline_training", {}).get("base_artifact_sha256", {})):
        for name, sha in hashes.items():
            require(_sha(sha), f"Invalid snapshot sha256: {json.dumps(name)}")
            files.reference(name, sha, root=snapshot, reason="Base snapshot sha256 mismatch")
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() or path.is_symlink():
            files.add(path)
    candidates = ([splits] if splits is not None else [relocation.path(name, root=root)
                  for name in bundles[0]["manifest"]["inputs"] if Path(name).name == "method_v1_splits.json"])
    if len(candidates) == 1 and candidates[0].is_file():
        sets = read_json(candidates[0]).get("sets", {})
        # A split lacking required sets is refused before K20 in read_dataset.
        # Hash-only inspection still works for such older producer bundles.
        if {"V2_pool", *FORBIDDEN_SETS} <= sets.keys() and "K20" not in sets:
            training_view_inputs(files, source)
    if check:
        files.verify()
    return {**result, "snapshot": snapshot, "theta0_provenance": source}
