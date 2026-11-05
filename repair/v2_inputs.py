"""Read-only relocation and main-grid collection verification for V2 inputs.

Only filesystem references are relocated. Sealed JSON, evidence and hashes are
never edited, and historical provenance is not recursively interpreted as paths.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import json
import os
from pathlib import Path
import re
import stat

from repair.artifacts import read_json
from sed import hash_receipts
from sed.storage import active_file_hash_cache, digest, file_hash

FORBIDDEN_SETS = ("U_dev", "U_mech", "D_probe", "D_unseen_quick", "dev140")


class ValidationError(ValueError):
    """Safe local validation diagnostics (never raw library exceptions)."""


def require(ok, message):
    if not ok:
        raise ValidationError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _within(path, root):
    """POSIX component-prefix test without allocating every ancestor Path."""
    return path.parts[:len(root.parts)] == root.parts


def input_hash_policy(function):
    """Keep audit mode active through validation AND run provenance, then restore it."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = os.environ.get("SED_HASH_CACHE")
        full = kwargs.get("full_rehash", False)
        if full:
            os.environ["SED_HASH_CACHE"] = "0"
        try:
            return function(*args, **kwargs)
        finally:
            if full:
                if previous is None:
                    os.environ.pop("SED_HASH_CACHE", None)
                else:
                    os.environ["SED_HASH_CACHE"] = previous
    return wrapped


class Relocation:
    """One component-prefix substitution, followed by symlink confinement.

    Source names are lexical: resolving them on this host before matching would
    lose the sealed identity. Destinations are checked against resolved NEW.
    """
    def __init__(self, dataset, values=()):
        self._directory_resolution = None
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
        # Fix each destination boundary for this inspection. Individual inputs
        # still resolve afresh, so retargeted links cannot escape that boundary.
        self.destinations = {new: new.resolve() for _, new in self.rules}
        original = Path(dataset).absolute()
        self.dataset = self.path(original)
        for _, new in self.rules:
            require(not new.absolute().is_relative_to(original) and
                    not self.destinations[new].is_relative_to(self.dataset),
                    "NEW must be outside the sealed dataset")

    def path(self, value, *, root=None):
        """Resolve an original reference, applying at most one mapping."""
        path = Path(value)
        if path.is_absolute():
            for old, new in self.rules:
                if _within(path, old):
                    tail = path.relative_to(old)
                    require(".." not in tail.parts, "Relocation cannot change a path's relative tail")
                    target = new / tail
                    require(target.relative_to(new) == tail, "Relocation changed a path's relative tail")
                    self.seen[str(old)].add(str(path))
                    resolved = self.local(target)
                    require(resolved.is_relative_to(self.destinations[new]),
                            f"Relocation symlink escape: {json.dumps(str(path))}")
                    return resolved
        return self.local(path if path.is_absolute() or root is None else Path(root) / path)

    def local(self, value):
        """Check a derived local path without performing a second substitution."""
        path = Path(value).absolute()
        try:
            resolved = (path.resolve() if self._directory_resolution is None else
                        self._resolve_in_batch(path, self._directory_resolution))
        except (OSError, RuntimeError):
            raise ValidationError(f"Cannot resolve input path: {json.dumps(str(path))}") from None
        for _, new in self.rules:
            boundary = self.destinations[new]
            if _within(path, new) or _within(path, boundary):
                require(_within(resolved, boundary),
                        f"Relocation symlink escape: {json.dumps(str(path))}")
        return resolved

    @staticmethod
    def _resolve_in_batch(path, directories):
        # Resolve shared directory components once in this read-only pass,
        # as a directory walk does. Inspect every leaf; delegate symlinks,
        # '..', missing components and unusual paths to pathlib unchanged.
        if ".." in path.parts or path.parent == path:
            return path.resolve()
        parent = directories.get(path.parent)
        if parent is None:
            parent = Relocation._resolve_in_batch(path.parent, directories)
            directories[path.parent] = parent
        candidate = parent / path.name
        try:
            info = candidate.lstat()
        except OSError:
            return path.resolve()
        return candidate.resolve() if stat.S_ISLNK(info.st_mode) else candidate

    @contextmanager
    def resolution_batch(self):
        previous = self._directory_resolution
        if previous is None and active_file_hash_cache() is not None and hash_receipts.cache_enabled():
            self._directory_resolution = {}
        try:
            yield
        finally:
            self._directory_resolution = previous

    def report(self):
        return {"map": [{"old": str(old), "new": str(new)} for old, new in self.rules],
                "relocated_paths_per_prefix": {str(old): len(self.seen[str(old)]) for old, _ in self.rules},
                "counting": "distinct original absolute references, including directories; once across bundles"}


class InputFiles:
    """Byte-hash inputs except evidence covered by successful collection validation."""
    def __init__(self, relocation):
        self.relocation = relocation
        self.files = {}
        self.hashes = {}
        self.collection_roots = set()
        self.collection_files = set()
        self.collection_anchors = {}
        self.collection_seals = {}
        self.collection_covered = set()
        self.full_rehash = not hash_receipts.cache_enabled()
        self.collection_artifact_inventory = "explicit_files"
        self.hash_cache = hash_receipts.cache_provenance()
        self.all_hashes_verified = False

    def add(self, path, sha=None, *, original=None, reason="Interchange input sha256 mismatch", _resolved=None):
        # Keep local aliases in the transfer list: HF snapshot symlinks and
        # their blobs are both needed. Resolve every one before reading bytes.
        path = Path(path).absolute()
        resolved = self.relocation.local(path) if _resolved is None else _resolved
        entry = self.files.setdefault(path, {"sources": set(), "seals": {}})
        entry["sources"].add(str(original if original is not None else path))
        if sha is not None:
            require(_sha(sha), f"Invalid input sha256: {json.dumps(str(path))}")
            entry["seals"].setdefault(sha, set()).add(reason)
        if resolved != path:
            target = self.files.setdefault(resolved, {"sources": set(), "seals": {}})
            target["sources"].update(entry["sources"])
            for expected, reasons in entry["seals"].items():
                target["seals"].setdefault(expected, set()).update(reasons)
        self.all_hashes_verified = False
        return path

    def reference(self, value, sha=None, *, root=None, reason="Interchange input sha256 mismatch"):
        resolved = self.relocation.path(value, root=root)
        return self.add(resolved, sha, original=value, reason=reason, _resolved=resolved)

    def verify(self, paths=None):
        """Verify all requirements, or only metadata needed for safe discovery."""
        self.all_hashes_verified = False
        failures = []
        entries = self.files.items() if paths is None else ((p, self.files[p]) for p in paths)
        cache = active_file_hash_cache()
        def verify_one(item):
            path, entry = item
            try:
                resolved = self.relocation.local(path)
                if not resolved.is_file():
                    return path, resolved, None, False, "missing file"
                covered = (not self.full_rehash and hash_receipts.cache_enabled() and resolved not in self.hashes
                    and resolved in self.collection_files
                    and (not entry["seals"] or resolved in self.collection_seals))
                if covered:
                    sha = self.collection_seals.get(resolved)
                else:
                    # V0 always stats again, including repeated verify() calls.
                    # Its scoped hash service handles aliases, receipts and audit
                    # bypass; retain the existing V2-only path outside that scope.
                    if cache is not None:
                        sha = cache.hash_resolved(resolved, file_hash)
                    else:
                        sha = file_hash(resolved) if resolved not in self.hashes else self.hashes[resolved]
                if entry["seals"]:
                    if sha not in entry["seals"] or len(entry["seals"]) != 1:
                        reasons = {why for values in entry["seals"].values() for why in values}
                        return path, resolved, sha, covered, "; ".join(sorted(reasons))
                return path, resolved, sha, covered, None
            except (OSError, RuntimeError):
                return path, None, None, False, "unreadable file"
        # Covered evidence still gets the same path/existence/seal checks.
        # Do not create a future for each of its ~million already-checked files.
        results = (cache.map(verify_one, entries, parallel_if=lambda item:
                            item[0] not in self.collection_files and not cache.likely_cached(item[0]))
                   if cache is not None else map(verify_one, entries))
        with self.relocation.resolution_batch():
            for path, resolved, sha, covered, failure in results:
                if covered:
                    self.collection_covered.add(resolved)
                elif sha is not None:
                    self.collection_covered.discard(resolved)
                    self.hashes[resolved] = sha
                if failure is not None:
                    failures.append((path, failure))
        if failures:
            first = "; ".join(f"{json.dumps(str(p))} ({why})" for p, why in sorted(failures)[:5])
            raise ValidationError(f"Input verification failed: {len(failures)} offending paths; first paths: {first}")
        self.all_hashes_verified = paths is None
        self.hash_cache = hash_receipts.cache_provenance()

    def report(self):
        cache = active_file_hash_cache()
        if cache is not None:
            self.hash_cache = hash_receipts.cache_provenance()
        byte_hashed = len(self.hashes) if cache is None else len(set(self.hashes) & cache.byte_hashed_paths)
        return {**self.relocation.report(), "all_hashes_verified": self.all_hashes_verified,
                "verified_hashed_files": len(self.hashes) if self.all_hashes_verified else 0,
                "byte_hashed_files": byte_hashed if self.all_hashes_verified else 0,
                "collection_covered_files": len(self.collection_covered) if self.all_hashes_verified else 0,
                "collection_validation_roots": sorted(map(str, self.collection_roots)),
                "collection_validation_anchors": self.collection_anchors,
                "collection_artifact_inventory": self.collection_artifact_inventory,
                "full_rehash": self.full_rehash, "hash_cache": self.hash_cache,
                "counting_files": "distinct resolved files; collection coverage excludes byte-hashed files"}

    def provenance_inputs(self, extra=()):
        """Retain byte-hashed inputs/anchors without rehashing the evidence bank."""
        require(self.all_hashes_verified, "Provenance requires completed input verification")
        with self.relocation.resolution_batch():
            return sorted(p for p in set(self.files) | {Path(p).absolute() for p in extra}
                          if p not in self.collection_covered
                          and self.relocation.local(p) not in self.collection_covered)

    def listing(self):
        groups = defaultdict(list)
        for path, entry in sorted(self.files.items()):
            prefix = next((boundary for boundary in self.relocation.destinations.values()
                           if path.is_relative_to(boundary)), None)
            if prefix is None:
                prefix = self.relocation.dataset if path.is_relative_to(self.relocation.dataset) else Path(path.anchor)
            groups[str(prefix)].append({"path": str(path), "relative_path": str(path.relative_to(prefix)),
                "source_paths": sorted(entry["sources"]),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sorted(entry["seals"]), "resolved_path": str(self.relocation.local(path))})
        return {"schema": "v2-training-inputs/1", "file_count": len(self.files),
                "relocation": self.report(),
                "groups": [{"prefix": prefix, "files": files} for prefix, files in sorted(groups.items())]}


def training_view_inputs(files, source, *, validate=False, config=None, seed_k=None):
    """Inventory the original support view when membership must be read from it.

    Follow only the evidence reader's explicit filesystem fields. In check mode,
    use its unchanged recursive hash and semantic validation before covering any
    evidence files. No rollout or model runtime is involved.
    """
    from sed.seed_selection import is_stratified_nested
    cache = active_file_hash_cache()
    compact = validate and cache is not None and hash_receipts.cache_enabled()
    if compact:
        # The collection reader compares every declared artifact hash with a
        # fresh enumeration/stat check. Keep that validation and its confinement
        # checks, without constructing and re-resolving a second million-file
        # inventory just to mark those same files as collection-covered later.
        cache.register_relocation(files.relocation)
        files.collection_artifact_inventory = "recursive_collection_validation"
    original = Path(source["config"]["baseline_training"]["verified"])
    if seed_k is not None:
        require(type(seed_k) is int and seed_k in (10, 20, 30), "Unsupported V0 seed K")
        require(source["config"]["baseline_training"].get("K") == seed_k
                and original.parts[-2:] == ("subsets", f"k{seed_k}"),
                "Round K differs from theta0's original support manifest subsets/k{K}")
    sha = source.get("inputs", {}).get(str(original / "manifest.json"))
    require(_sha(sha), "K20 view differs from theta0's original support manifest")
    view_manifest = files.reference(original / "manifest.json", sha,
        reason="K20 view differs from theta0's original support manifest")
    view = files.relocation.path(original)
    if active_file_hash_cache() is not None:
        active_file_hash_cache().protect(view)
    if validate:
        # The original byte seal binds theta0 to this view before its recursive
        # collection/semantic checks are allowed to cover any dependencies.
        files.verify(paths=[view_manifest])
    pending, seen, covered, anchors = [view], set(), set(), {}
    def dependency(value, root):
        path = files.relocation.path(value, root=root)
        # The collection reader consumes sealed internal paths unchanged. Never
        # grant coverage to a relocated file when it validated a different one.
        require(not Path(value).is_absolute() or path == files.relocation.local(root / value),
                "Collection internal reference differs after relocation")
        return path

    while pending:
        root = pending.pop()
        if active_file_hash_cache() is not None:
            active_file_hash_cache().protect(root)
        if root in seen:
            continue
        seen.add(root)
        for name in ("manifest.json", "selection.json", "identity.json"):
            covered.add(files.add(root / name))
        # Reading these documents requires their presence even in list mode.
        missing = [root / name for name in ("manifest.json", "selection.json", "identity.json")
                   if not (root / name).is_file()]
        require(not missing, f"Missing K20 metadata: {len(missing)} offending paths: "
                + ", ".join(json.dumps(str(p)) for p in missing[:5]))
        collection, identity = read_json(root / "manifest.json"), read_json(root / "identity.json")
        selection = read_json(root / "selection.json")
        anchors[str(root)] = {"semantic_sha256": {name: digest(value) for name, value in
            (("manifest.json", collection), ("selection.json", selection), ("identity.json", identity))}}
        evidence_root = root.parent.parent if "source_collection" in collection else root
        if evidence_root != root:
            pending.append(evidence_root)
            # The subset validator recomputes these byte seals as part of its
            # comparison with the expected view, including receipt whitespace.
            for reference in collection.get("receipt_refs", []):
                require(_sha(reference["sha256"]), "Invalid K20 receipt sha256")
                path = dependency(reference["path"], root)
                files.collection_seals[path] = reference["sha256"]
        for field in ("support_view", "extension_source"):
            # Mirror the loader's actual recursion, not arbitrary identity keys.
            if (evidence_root == root and field in identity
                    and (field == "extension_source" or is_stratified_nested(selection))):
                parent = dependency(identity[field]["path"], root)
                pending.append(parent)
                if field == "extension_source" and not compact:
                    # Its recorded tree seal makes every original source file a
                    # dependency, including inherited smaller-K view anchors.
                    for path in sorted(parent.rglob("*")):
                        if (path.is_file() or path.is_symlink()) and not path.name.endswith(".lock"):
                            covered.add(files.add(path))
        for receipt in collection.get("attempts", []):
            episode = dependency(receipt["episode_dir"], evidence_root)
            require(episode.is_relative_to(evidence_root), "Episode path escapes collection")
            covered.add(files.add(episode.parent / "receipt.json"))
            if compact:
                continue  # Unchanged recursive loader validates these dictionaries and files below.
            for directory, key in ((episode, "episode_hashes"), (episode.parent / "teacher", "teacher_hashes")):
                for name, sha in receipt[key].items():
                    require(_sha(sha), "Invalid K20 evidence sha256")
                    path = dependency(name, directory)
                    files.add(path, sha, original=name, _resolved=path)
                    covered.add(path)
                    files.collection_seals[path] = sha
    collection = None
    if validate:
        from sed.manifest import load_frozen
        from sed.seed_evidence import load_verified_collection
        effective = config if config is not None else source["config"]
        frozen, freeze = load_frozen(*(files.relocation.path(effective["paths"][key])
                                       if config is None else effective["paths"][key]
                                       for key in ("manifest", "split_freeze")))
        try:
            collection, _, _ = load_verified_collection(view, frozen, freeze)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            raise ValidationError(f"K20 collection validation failed: {json.dumps(str(view))}") from None
        if seed_k is not None:
            require(collection["K"] == seed_k, "Round K differs from theta0's original support manifest")
        # The loader recursively checks the source, support and extension roots
        # followed above. Register coverage only after all of it succeeds.
        files.collection_roots.update(seen)
        # Only dependencies actually checked by the loader are covered. An
        # unrelated config or interchange file placed under a root still hashes.
        files.collection_files.update(p for p in covered
            if any(p.is_relative_to(root) for root in seen))
        files.collection_anchors.update(anchors)
    return view, collection


def split_source_inputs(files, splits, split_doc, *, repair_splits=None, dev140=None):
    """Follow only the hash-bound diagnostic sources named by the method freeze."""
    sets = split_doc.get("sets", {})
    require(isinstance(sets, dict) and {"V2_pool", "U_dev"} <= sets.keys(),
            "Method split freeze must include V2_pool and U_dev")
    # Preserve the previously supported self-contained frozen split format.
    # Once external sources are recorded, inline sets cannot bypass their seals.
    if (set(FORBIDDEN_SETS) <= sets.keys() and "exclusion_sources" not in split_doc
            and "rebuild_arguments" not in split_doc and repair_splits is None and dev140 is None):
        return {}
    records, arguments = split_doc.get("exclusion_sources"), split_doc.get("rebuild_arguments")
    require(isinstance(records, list) and isinstance(arguments, dict),
            "Missing hash-bound diagnostic split sources")
    result = {}
    for key, override, categories in (("repair_splits", repair_splits, ("U_mech", "D_probe", "D_unseen_quick")),
                                       ("dev140", dev140, ("dev140",))):
        original = arguments.get(key)
        require(isinstance(original, str) and original, f"Missing {key} source path")
        seals = set()
        for category in categories:
            matches = [r for r in records if isinstance(r, dict) and r.get("category") == category]
            require(matches and all(r.get("path") == original and _sha(r.get("sha256")) for r in matches),
                    f"Missing or conflicting {category} source sha256")
            seals.update(r["sha256"] for r in matches)
        require(len(seals) == 1, f"Conflicting {key} source sha256")
        sha = next(iter(seals))
        recorded = split_doc.get("input_sha256", {}).get(original, sha)
        require(recorded == sha, f"Conflicting {key} source sha256")
        if override is not None:
            path = files.relocation.path(override)
        elif Path(original).is_absolute():
            path = files.relocation.path(original)
        else:
            # Rebuild paths are relative to the producer project, not to its
            # data/manifest directory or the consumer's current directory.
            output = arguments.get("output")
            require(isinstance(output, str) and output and not Path(output).is_absolute()
                    and ".." not in Path(output).parts and ".." not in Path(original).parts,
                    "Relative split sources require a relative rebuild output path")
            tail = Path(output).parts
            require(tuple(splits.parts[-len(tail):]) == tail,
                    "Cannot locate split source root; provide --repair-splits and --dev140")
            project = splits.parents[len(tail) - 1]
            # splits is already relocated: do not apply a second substitution.
            path = files.relocation.local(project / original)
        result[key] = files.add(path, sha, original=original, reason=f"{key} source sha256 mismatch")
    return result


def _example_summary(bundles, theta0, files):
    """Use the run reader's sealed bundle validation without loading a model."""
    # Local import: the bundle validator shares this module's safe errors.
    from repair.v2_validation import _read_bundle
    labels = Counter()
    for bundle in bundles:
        data = _read_bundle(bundle, theta0, files)
        labels.update(f'{row["evidence"]["label_source"]}|{row["evidence"]["label_check"]}'
                      for row in data["rows"])
    return {"examples_validated": sum(labels.values()), "label_provenance": dict(sorted(labels.items()))}


@input_hash_policy
def inspect_inputs(dataset, *, theta0=None, relocate=(), snapshot=None, config=None,
                   splits=None, repair_splits=None, dev140=None, inputs=(), check=True, runtime=True,
                   full_rehash=False, seed_k=None):
    """Inventory and verify inputs and examples; no tokenizer/model/training imports.

    Snapshot defaults to the sealed tokenizer location, then theta0's recorded
    base. Callers can supply the same explicit snapshot/config as a run.
    Historical theta0 training inputs are followed only when the reader can
    reach its original-K20 membership fallback; other history is not opened.
    """
    # Only V0 opts into an explicit K; standalone V2 keeps its original gates.
    if seed_k is not None:
        require(type(seed_k) is int and seed_k in (10, 20, 30), "Unsupported V0 seed K")
    seed_set = f"K{20 if seed_k is None else seed_k}"
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
    if splits is None:
        candidates = [name for name in bundles[0]["manifest"]["inputs"]
                      if Path(name).name == "method_v1_splits.json"]
        require(len(candidates) == 1, "Provide --splits or a hashed method_v1_splits.json input")
        splits = files.reference(candidates[0], root=root)
    else:
        splits = files.reference(splits)
    for bundle in bundles:
        require(splits in {relocation.path(name, root=bundle["root"]) for name in bundle["manifest"]["inputs"]},
                "Split freeze must be an interchange input")
        files.add(splits, bundle["manifest"]["splits_sha256"], reason="Split freeze sha256 mismatch")
    if check:
        files.verify(paths=[splits])
    require(splits.is_file(), "Input listing requires local method split metadata")
    split_doc = read_json(splits)
    split_sources = split_source_inputs(files, splits, split_doc, repair_splits=repair_splits, dev140=dev140)
    result = {"files": files, "bundles": bundles, "theta0": theta0,
              "config": local_config, "splits": splits, "split_sources": split_sources}
    if not runtime and seed_set in split_doc["sets"]:
        if check:
            files.verify()
            result.update(_example_summary(bundles, theta0, files))
        return result
    metadata_paths = (theta0 / "adapter_config.json", theta0 / "training_state.json", theta0.parent / "provenance.json")
    for path in metadata_paths:
        files.add(path)
    # Validate sealed files before trusting the dependency metadata they contain.
    if check:
        files.verify(paths=metadata_paths)
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
    if seed_set not in split_doc["sets"]:
        view, collection = training_view_inputs(files, source, validate=check, config=local_config,
                                               seed_k=seed_k)
        result.update(training_view=view, training_collection=collection)
    if check:
        files.verify()
        result.update(_example_summary(bundles, theta0, files))
    return {**result, "snapshot": snapshot, "theta0_provenance": source}
