"""Validate environment evidence separately from model-visible training fields."""
import json
import os
import sys
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from errno import EBADF, ELOOP, ENOENT, ENOTDIR
from itertools import islice
from pathlib import Path

from sed.manifest import authorize_task, resolve_task, frozen_write
from sed.seed_selection import (is_full_train, is_stratified_nested, select_nested_verified,
                                stratified_nested_order, select_seeds, validate_selection,
                                validate_extension_selection)
from sed.storage import BlobStore, file_hash, digest, locked, write_json


def environment_won(directory, episode):
    if not (episode.get("complete") is True and episode.get("status") == "ok"
            and episode.get("success") is True and episode.get("steps")):
        return False
    final = episode["steps"][-1]
    verification = BlobStore(Path(directory) / "blobs").get(final["verification_ref"])
    return (verification.get("fact_source") == "real_environment"
            and verification.get("executable") is True
            and verification.get("audit", {}).get("won") is True
            and final.get("environment_steps") == 1)


def artifact_hashes(directory):
    return artifact_hashes_many([Path(directory)])[0]


def _hash_entry(path):
    return file_hash(path) if path.is_file() and not path.name.endswith(".lock") else None


class _DirectoryListing:
    """One scan task; mirror CPython 3.11/3.12 pathlib's ** then * selectors.

    Keep the wildcard and descent scans separate, as pathlib does: their error
    scopes differ, especially Path.walk's iterator errors on 3.12. DirEntry's
    no-follow directory check in both local versions avoids symlink target stats.
    Workers record errors; the caller later visits them in pathlib's order.
    """
    def __init__(self, directory, *, root=False):
        self.directory, self.root = directory, root
        self.paths, self.children = [], []
        self.error = self.walk_error = None

    def scan(self):
        modern = sys.version_info >= (3, 12)
        try:
            if self.root:
                sys.audit("pathlib.Path.rglob", self.directory, "*")
                if not self.directory.is_dir():
                    return
            try:
                with os.scandir(self.directory) as entries:
                    entries = list(entries)
            except OSError as error:
                if not modern and not isinstance(error, PermissionError):
                    raise
            else:
                self.paths = [self.directory / entry.name for entry in entries]
        except Exception as error:
            self.error = error
            return
        try:
            if modern:
                # Path.walk suppresses open errors, but not iterator/close errors.
                try:
                    entries = os.scandir(self.directory)
                except OSError:
                    return
                with entries:
                    for entry in entries:
                        try:
                            is_dir = entry.is_dir(follow_symlinks=False)
                        except OSError:
                            is_dir = False
                        if is_dir:
                            self.children.append(_DirectoryListing(self.directory / entry.name))
            else:
                try:
                    with os.scandir(self.directory) as entries:
                        entries = list(entries)
                    for entry in entries:
                        try:
                            is_dir = entry.is_dir(follow_symlinks=False)
                        except OSError as error:
                            if (error.errno not in (ENOENT, ENOTDIR, EBADF, ELOOP)
                                    and getattr(error, "winerror", None) not in (21, 123, 1921)):
                                raise
                            continue
                        if is_dir:
                            self.children.append(_DirectoryListing(self.directory / entry.name))
                except PermissionError:
                    pass
        except Exception as error:
            self.walk_error = error
            if modern:
                self.children.clear()  # walk yields only after the scan closes.

    def collect(self):
        paths, pending = [], [self]
        modern = sys.version_info >= (3, 12)
        if modern:
            if self.error is not None:
                raise self.error
            paths.extend(self.paths)
        while pending:
            node = pending.pop()
            if isinstance(node, Exception):
                raise node
            if modern:
                if node.walk_error is not None:
                    raise node.walk_error
                # 3.12 yields every child's wildcard results before walking them.
                for child in node.children:
                    if child.error is not None:
                        raise child.error
                    paths.extend(child.paths)
            else:
                if node.error is not None:
                    raise node.error
                paths.extend(node.paths)
                if node.walk_error is not None:
                    pending.append(node.walk_error)
            pending.extend(reversed(node.children))
        return sorted(paths)


@contextmanager
def artifact_hashes_batch(directories):
    """Yield ordered hash readers; raise I/O errors only when a reader is called.

    Paths may be thunks so receipt/path errors keep their original check order.
    Prefetch at most one tree per worker and one file per worker/tree.
    Scans and files share one pool. The caller schedules at most one scan per
    worker, including separate tasks for subdirectories within a large tree.
    """
    workers = int(os.environ.get("SED_HASH_WORKERS", "16"))
    if workers < 1:
        raise ValueError("SED_HASH_WORKERS must be positive")

    ready, scanning = deque(), {}

    def failed_reader(error):
        failed = Future()
        failed.set_exception(error)
        return failed.result

    def hash_reader(root, pool):
        try:
            paths = iter(root.collect())
            pending = deque((p, pool.submit(_hash_entry, p)) for p in islice(paths, workers))
        except Exception as error:
            return failed_reader(error)
        def collect():
            result = {}
            while pending:
                path, future = pending.popleft()
                sha = future.result()
                if sha is not None:
                    result[str(path.relative_to(root.directory))] = sha
                for path in paths:
                    pending.append((path, pool.submit(_hash_entry, path)))
                    break
            return result
        return collect

    def schedule(pool):
        while ready and len(scanning) < workers:
            tree = ready.popleft()
            node = tree["todo"].popleft()
            scanning[pool.submit(node.scan)] = tree, node
            if tree["todo"]:
                ready.append(tree)

    def advance(pool):
        done, _ = wait(scanning, return_when=FIRST_COMPLETED)
        for future in done:
            tree, node = scanning.pop(future)
            future.result()
            if node.children and not tree["todo"]:
                ready.append(tree)
            tree["todo"].extend(node.children)
            tree["remaining"] += len(node.children) - 1
            if not tree["remaining"]:
                tree["read"] = hash_reader(tree["root"], pool)
        schedule(pool)

    def prepare(value, pool):
        try:
            directory = Path(value() if callable(value) else value)
            if pool is None:
                paths = sorted(directory.rglob("*"))
                return {str(p.relative_to(directory)): file_hash(p) for p in paths
                        if p.is_file() and not p.name.endswith(".lock")}
            root = _DirectoryListing(directory, root=True)
            tree = {"root": root, "todo": deque([root]), "remaining": 1}
            ready.append(tree)
            schedule(pool)
            def collect():
                while tree["remaining"]:
                    advance(pool)
                return tree["read"]()
            return collect
        except Exception as error:
            if pool is None:
                raise
            return failed_reader(error)

    if workers == 1:
        yield (lambda value=value: prepare(value, None) for value in directories)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def readers():
            values = iter(directories)
            pending = deque(prepare(value, pool) for _, value in zip(range(workers), values))
            while pending:
                yield pending.popleft()
                for value in values:
                    pending.append(prepare(value, pool))
                    break
        yield readers()


def artifact_hashes_many(directories):
    with artifact_hashes_batch(directories) as readers:
        return [read() for read in readers]


def _episode_directory(root, receipt):
    directory = (Path(root) / receipt["episode_dir"]).resolve()
    if not directory.is_relative_to(Path(root).resolve()):
        raise ValueError("Episode path escapes collection")
    return directory


def checked_episode(root, receipt, *, _hashes=None):
    directory = _episode_directory(root, receipt)
    if (_hashes() if _hashes is not None else artifact_hashes(directory)) != receipt["episode_hashes"]:
        raise ValueError("Episode evidence changed")
    episode = json.loads((directory / "episode.json").read_text())
    if not environment_won(directory, episode):
        raise ValueError("Seed lacks a real environment win")
    return episode


def checked_episodes(root, receipts):
    with artifact_hashes_batch(lambda receipt=receipt: _episode_directory(root, receipt)
                               for receipt in receipts) as readers:
        for receipt, hashes in zip(receipts, readers):
            yield checked_episode(root, receipt, _hashes=hashes)


def receipt_path(root, receipt):
    path = (Path(root) / receipt["episode_dir"]).resolve().parent / "receipt.json"
    if not path.is_relative_to(Path(root).resolve() / "attempts"):
        raise ValueError("Receipt path escapes collection attempts")
    return path


def _subset_values(root, source, large, small):
    nested = is_stratified_nested(small)
    if nested:
        expected = select_nested_verified(source, large, k=small["K"], seed=small["seed"])
        if small != expected:
            raise ValueError("Nested selection differs from the full verified prefix")
    else:
        validate_extension_selection(small, large)
    parents = {r["task_id"] for rows in small["selected"].values() for r in rows}
    verified = [r for r in source["verified"] if r["selection_parent_task_id"] in parents]
    if nested:
        by_task = {r["task_id"]: r for r in verified}
        verified = [by_task[tid] for tid in small["task_order"]]
    attempts = [r for r in source["attempts"] if r["selection_parent_task_id"] in parents]
    if len(verified) != small["K"] or {r["selection_parent_task_id"] for r in verified} != parents:
        raise ValueError("Source lacks verified evidence for every subset slot")
    reference = {"path": "../..", "manifest_sha256": digest(source),
                 "selection_sha256": digest(large), "K": source["K"] if nested else large["K"]}
    identity = {"selection": small, "mode": "reuse_verified_nested_v2" if nested else "reuse_verified_parent_slots_v1",
                "source_collection": reference}
    value = {"schema_version": 1, "status": "complete", "K": small["K"],
             "K_counts": "trusted_executed_trajectories", "identity_sha256": digest(identity),
             "selection_sha256": digest(small), "trusted_trajectories": len(verified),
             "unique_instances": len({r["task_id"] for r in verified}),
             "task_groups": len({r["group_id"] for r in verified}),
             "attempted_unique_instances": len({r["task_id"] for r in attempts}),
             "attempted_task_groups": len({r["group_id"] for r in attempts}),
             "attempts": attempts, "verified": verified, "parent_seed_ids": [],
             "source": "reused_verified_nested_collection", "source_collection": reference,
             "receipt_refs": [{"path": "../../" + str(receipt_path(root, r).relative_to(Path(root).resolve())),
                               "sha256": file_hash(receipt_path(root, r))} for r in attempts],
             "reused_trajectories": len(verified), "new_teacher_calls": 0,
             "new_environment_steps": 0, "training_exposure": 0,
             "accounting": "Slot attempts are a view of source costs, not additional calls. "
                           "The source manifest retains ALL largest-K attempts and costs; "
                           "do not sum these views as independent collections."}
    return identity, value


def load_verified_collection(root, manifest, freeze, *, allow_subset=True):
    """Validate evidence first; callers separately whitelist model-visible fields."""
    root = Path(root).resolve()
    collection = json.loads((root / "manifest.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    nested = is_stratified_nested(selection)
    if not nested:
        validate_selection(selection, manifest, freeze)
    full_train = is_full_train(selection)
    if (collection["status"] != "complete" or collection["selection_sha256"] != digest(selection)
            or (not full_train and len(collection["verified"]) != selection["K"])
            or collection["identity_sha256"] != digest(identity) or identity["selection"] != selection):
        raise ValueError("Collection must contain exactly K verified trajectories")
    if "source_collection" in collection:
        if not allow_subset or root.parent.name != "subsets" or root.name != f"k{selection['K']}":
            raise ValueError("Invalid nested collection location or chained source")
        evidence_root = root.parent.parent
        source, large, _ = load_verified_collection(evidence_root, manifest, freeze, allow_subset=False)
        expected_identity, expected = _subset_values(evidence_root, source, large, selection)
        if identity != expected_identity or collection != expected:
            raise ValueError("Nested collection differs from shared source receipts")
    else:
        evidence_root = root
        if nested:
            # FTP/SmartAD publish new executions on exactly the original view's
            # support. Keep that view hash-bound, with its full-source validation.
            reference = identity.get("support_view")
            if not reference:
                raise ValueError("Nested v2 publication requires a bound support view")
            support_root = (root / reference["path"]).resolve()
            support_manifest = json.loads((support_root / "manifest.json").read_text())
            if "source_collection" not in support_manifest or support_root == root:
                raise ValueError("Support reference must point directly to a nested view")
            support, original, _ = load_verified_collection(support_root, manifest, freeze)
            if digest(support) != reference["manifest_sha256"] or selection != original:
                raise ValueError("Publication differs from its bound support view")
        from sed.seed_extension import check_inherited_views
        check_inherited_views(root, collection, selection, manifest, freeze, _subset_values)
    candidates = {r["task_id"] for kind in selection["type_order"]
                  for r in selection["selected"][kind] + selection["replacements"][kind]}
    parents = {r["task_id"]: kind for kind in selection["type_order"] for r in selection["selected"][kind]}
    # Every failed/interrupted attempt remains referenced and checked too.
    seen_attempts = {}
    directories = (lambda receipt=receipt, teacher=teacher:
                   (evidence_root / receipt["episode_dir"]).resolve().parent / "teacher" if teacher
                   else (evidence_root / receipt["episode_dir"]).resolve()
                   for receipt in collection["attempts"] for teacher in (False, True))
    with artifact_hashes_batch(directories) as hashes:
        for receipt in collection["attempts"]:
            if json.loads(receipt_path(evidence_root, receipt).read_text()) != receipt:
                raise ValueError("Stored receipt differs from collection")
            directory = (evidence_root / receipt["episode_dir"]).resolve()
            if (next(hashes)() != receipt["episode_hashes"]
                    or next(hashes)() != receipt["teacher_hashes"]):
                raise ValueError("Attempt evidence changed")
            prior = seen_attempts.setdefault(receipt["task_id"], [])
            episode_file = directory / "episode.json"
            episode = json.loads(episode_file.read_text()) if episode_file.exists() else {}
            if (type(receipt["attempt"]) is not int or receipt["attempt"] != len(prior) + 1
                    or receipt["attempt"] > 3 or any(r["won"] for r in prior)
                    or receipt["won"] != environment_won(directory, episode)):
                raise ValueError("Attempt sequence/win disagrees with environment evidence")
            prior.append(receipt)
            task = resolve_task(manifest, freeze, receipt["task_id"])
            authorize_task(task, "generation")
            if (task["source_split"] != "train" or receipt["task_id"] not in candidates
                    or receipt["group_id"] != task["group_id"] or receipt["task_type"] != task["task_type"]
                    or parents.get(receipt["selection_parent_task_id"]) != task["task_type"]
                    or ((full_train or nested) and receipt["selection_parent_task_id"] != task["task_id"])):
                raise ValueError("Receipt differs from frozen TRAIN selection")
    groups, slots, counts = set(), set(), Counter()
    instances = set()
    with artifact_hashes_batch(lambda receipt=receipt: _episode_directory(evidence_root, receipt)
                               for receipt in collection["verified"]) as readers:
        for receipt, hashes in zip(collection["verified"], readers):
            task = resolve_task(manifest, freeze, receipt["task_id"])
            if (receipt not in collection["attempts"] or receipt["won"] is not True
                    or (not (full_train or nested) and task["group_id"] in groups)
                    or receipt["selection_parent_task_id"] in slots):
                raise ValueError("Seeds must be distinct TRAIN groups and selection slots")
            episode = checked_episode(evidence_root, receipt, _hashes=hashes)
            if (episode["task_id"] != task["task_id"] or episode["game_sha256"] != task["game_sha256"]
                    or episode["split"] != "train"):
                raise ValueError("Episode differs from frozen TRAIN task")
            groups.add(task["group_id"])
            instances.add(task["task_id"])
            slots.add(receipt["selection_parent_task_id"])
            counts[task["task_type"]] += 1
    if full_train:
        failed = [tid for tid, rows in seen_attempts.items() if not rows[-1]["won"]]
        if (list(seen_attempts) != selection["candidate_order"] or identity.get("attempts_per_candidate") != 3
                or any(len(seen_attempts[tid]) != 3 for tid in failed)
                or collection.get("failed_candidates") != failed
                or collection["verified"] != [r for r in collection["attempts"] if r["won"]]
                or collection.get("selection_rule") != selection["selection_rule"]
                or collection.get("candidates") != selection["candidates"]
                or collection.get("K_counts") != "trusted_executed_trajectories"
                or collection["K"] != len(slots) or collection["trusted_trajectories"] != len(slots)
                or collection["unique_instances"] != len(slots) or collection["task_groups"] != len(groups)
                or collection["attempted_unique_instances"] != len(candidates)
                or collection["attempted_task_groups"] != selection["selected_task_groups"]):
            raise ValueError("Full TRAIN requires every candidate's first success or all three failed attempts")
    elif nested:
        if (list(r["task_id"] for r in collection["verified"]) != selection["task_order"]
                or any(counts[kind] != quota for kind, quota in selection["quotas"].items())
                or collection["K"] != selection["K"]
                or collection["trusted_trajectories"] != len(instances)
                or collection["unique_instances"] != len(instances)
                or collection["task_groups"] != len(groups)):
            raise ValueError("Verified trajectories differ from nested prefix/counts")
    elif (any(counts[kind] != quota for kind, quota in selection["quotas"].items())
            or collection["K"] != selection["K"] or collection["trusted_trajectories"] != len(groups)
            or collection["unique_instances"] != len(groups) or collection["task_groups"] != len(groups)):
        raise ValueError("Verified trajectories differ from stratified quotas/counts")
    return collection, selection, evidence_root


def verified_collection_inputs(root, manifest, freeze):
    """Return provenance inputs for a validated collection or nested subset."""
    root = Path(root)
    collection, selection, evidence_root = load_verified_collection(root, manifest, freeze)
    if "source_collection" in collection:
        inputs = [root / "manifest.json", root / "identity.json", root / "selection.json",
                  evidence_root / "manifest.json", evidence_root / "provenance.json"]
    else:
        inputs = [root / "manifest.json", root / "provenance.json"]
    identity = json.loads((evidence_root / "identity.json").read_text())
    if is_stratified_nested(selection):
        inputs.extend(p for p in (evidence_root / "identity.json", evidence_root / "selection.json")
                      if p not in inputs)
        if "support_view" in identity:
            support_root = (evidence_root / identity["support_view"]["path"]).resolve()
            inputs.extend(verified_collection_inputs(support_root, manifest, freeze))
    if "extension_source" in identity:
        origin = (evidence_root / identity["extension_source"]["path"]).resolve()
        inputs.extend([evidence_root / "identity.json", evidence_root / "selection.json",
                       origin / "manifest.json", origin / "identity.json", origin / "selection.json",
                       origin / "provenance.json"])
        if identity["extension_source"]["K"] == 20:
            inputs.extend(origin / "subsets" / f"k{k}" / name
                          for k in (5, 10) for name in ("manifest.json", "identity.json", "selection.json"))
    return inputs


def materialize_subsets(root, manifest, freeze, ks):
    """Write immutable views under largest-K/subsets/kK; never copy trajectories."""
    root = Path(root).resolve()
    with locked(root / ".collection.lock"):
        source, large, _ = load_verified_collection(root, manifest, freeze, allow_subset=False)
        if is_full_train(large):
            raise ValueError("Full TRAIN does not define nested few-shot subsets")
        if is_stratified_nested(large):
            raise ValueError("Create nested v2 views directly from the full TRAIN source")
        values = []
        for k in sorted(set(ks)):
            small = select_seeds(manifest, freeze, k=k, seed=large["seed"], rule=large["selection_rule"])
            identity, value = _subset_values(root, source, large, small)
            values.append((root / "subsets" / f"k{k}", small, identity, value))
        for output, small, identity, value in values:
            frozen_write(output / "selection.json", small)
            frozen_write(output / "identity.json", identity)
            frozen_write(output / "manifest.json", value)
        return [value for _, _, _, value in values]


def support_view_reference(verified, output, source, selection):
    """Extra identity binding for new FTP/SmartAD publications only."""
    if not is_stratified_nested(selection):
        return {}
    if "source_collection" not in source:
        raise ValueError("Nested v2 teacher collection requires an original support view")
    return {"support_view": {"path": os.path.relpath(Path(verified).resolve(), Path(output).resolve()),
                             "manifest_sha256": digest(source)}}


def materialize_full_train_views(root, manifest, freeze, ks, *, seed=0, dry_run=False):
    """New v2 views only. Dry run is read-only, including no lock-file creation."""
    root = Path(root).resolve()
    ks = list(ks)
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("View Ks must be distinct positive integers")
    outputs = [root / "subsets" / f"k{k}" for k in sorted(ks)]
    if (root / "subsets").is_symlink():
        raise ValueError("Subsets directory must not be a symlink")
    if any(os.path.lexists(output) for output in outputs):
        raise FileExistsError("Target view already exists; refusing to overwrite")
    source, large, _ = load_verified_collection(root, manifest, freeze, allow_subset=False)
    if not is_full_train(large):
        raise ValueError("Nested v2 requires a complete all_train_v1 source")
    order = stratified_nested_order(source, seed=seed)
    values = []
    for output, k in zip(outputs, sorted(ks)):
        small = select_nested_verified(source, large, k=k, seed=seed, order=order)
        identity, value = _subset_values(root, source, large, small)
        values.append((output, small, identity, value))
    if not dry_run:
        # Reserve new directories exclusively; never open an existing source file
        # for writing (including the collector's existing lock file).
        for output, _, _, _ in values:
            output.mkdir(parents=True, exist_ok=False)
        for output, small, identity, value in values:
            for name, record in (("selection", small), ("identity", identity), ("manifest", value)):
                write_json(output / (name + ".json"), record, exclusive=True)
    return [small for _, small, _, _ in values]
