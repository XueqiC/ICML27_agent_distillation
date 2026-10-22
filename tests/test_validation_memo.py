"""Whole-collection memoization preserves validation and transitive bindings."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from baselines.common.collection_views import bind_support, check_support
from sed import seed_evidence
from sed.validation_cache import capture_collections, clear_validation_cache
from test_full_train import full_bank
from test_full_train_views import source_bank


def test_views_support_share_one_validation_and_same_object(source_bank, monkeypatch):
    root, _, manifest, freeze, _, _ = source_bank
    seed_evidence.materialize_full_train_views(root, manifest, freeze, [1, 2])
    clear_validation_cache()
    calls = Counter()
    original = seed_evidence._load_verified_collection
    def counted(path, *a, **kw):
        calls[Path(path).resolve()] += 1
        return original(path, *a, **kw)
    monkeypatch.setattr(seed_evidence, "_load_verified_collection", counted)
    one = root / "subsets/k1"
    first = seed_evidence.load_verified_collection(one, manifest, freeze)
    assert seed_evidence.load_verified_collection(one / ".." / "k1", manifest, freeze) is first
    seed_evidence.load_verified_collection(root / "subsets/k2", manifest, freeze)
    full = seed_evidence.load_verified_collection(root, manifest, freeze, allow_subset=False)
    assert seed_evidence.load_verified_collection(root, manifest, freeze) is full
    binding = bind_support(one, *first[:2])
    support = check_support(binding, manifest, freeze)
    with capture_collections() as bound:
        assert check_support(binding, manifest, freeze) is support
    assert set(bound) == {str(one.resolve()), str(root.resolve())}
    assert calls == {root.resolve(): 1, one.resolve(): 1, (root / "subsets/k2").resolve(): 1}
    with pytest.raises(ValueError, match="Invalid nested"):
        seed_evidence.load_verified_collection(one, manifest, freeze, allow_subset=False)
    wrong = {**binding, "K": 999}
    with pytest.raises(ValueError, match="binding changed"):
        check_support(wrong, manifest, freeze)
    # Receipt cache modes are separate semantic-validation sessions.
    monkeypatch.setenv("SED_HASH_CACHE", "0")
    assert seed_evidence.load_verified_collection(one, manifest, freeze) is not first
    assert calls[root.resolve()] == calls[one.resolve()] == 2
    # Fresh top-level hashes invalidate a transitive parent's cache, too.
    path = root / "identity.json"
    path.write_text(path.read_text() + " ")
    second = seed_evidence.load_verified_collection(one, manifest, freeze)
    assert second == first and second is not first
    assert calls[root.resolve()] == calls[one.resolve()] == 3


def test_manifest_freeze_and_failed_loads_are_not_reused(source_bank, monkeypatch):
    root, _, manifest, freeze, _, _ = source_bank
    clear_validation_cache()
    first = seed_evidence.load_verified_collection(root, manifest, freeze)
    calls = []
    original = seed_evidence._load_verified_collection
    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(seed_evidence, "_load_verified_collection", counted)
    changed = deepcopy(freeze)
    changed["manifest_sha256"] = "wrong"
    for _ in range(2):
        with pytest.raises(ValueError):
            seed_evidence.load_verified_collection(root, manifest, changed)
    assert len(calls) == 2
    assert seed_evidence.load_verified_collection(root, manifest, freeze) is first
