"""Both baseline preparation entry points enforce the registered seed ladder."""
from types import SimpleNamespace

import pytest

from baselines.common import REGISTERED_K, pipeline as common
from baselines.eto import pipeline as eto


@pytest.mark.parametrize("pipeline", [common, eto], ids=["shared", "eto"])
@pytest.mark.parametrize("k", [1, 2, 5, 10, 20, 50, 100, 3, 200])
def test_prepare_registered_k(pipeline, k, tmp_path, monkeypatch):
    assert REGISTERED_K == (1, 2, 5, 10, 20, 50, 100)
    args = SimpleNamespace(output=tmp_path / "output", verified=tmp_path / "verified",
                           snapshot=tmp_path / "snapshot", config=tmp_path / "config.yaml", gpus=4)
    config = {"paths": {name: str(tmp_path / name)
                        for name in ("manifest", "split_freeze", "raw", "verified")}}
    collection = {"K": k, "unique_instances": k, "task_groups": 1,
                  "verified": [{"task_id": f"task-{i}"} for i in range(k)]}
    monkeypatch.setattr(pipeline, "load_frozen", lambda *a: ({}, {}))
    monkeypatch.setattr(pipeline, "load_verified_collection",
                        lambda *a: (collection, {"K": k, "seed": 0}, args.verified))
    checked = []

    def checked_episode(root, receipt):
        checked.append(receipt["task_id"])
        return {"episode_id": "seed-" + receipt["task_id"]}

    monkeypatch.setattr(pipeline, "checked_episode", checked_episode)
    monkeypatch.setattr(pipeline, "training_tasks", lambda listing, *a: listing["task_ids"])
    monkeypatch.setattr(pipeline, "artifact_hashes", lambda *a: {})
    monkeypatch.setattr(pipeline, "code_hashes", lambda *a: {})
    monkeypatch.setattr(pipeline, "write_provenance", lambda *a, **kw: None)
    kwargs = {"method": "kang", "recipe": {}} if pipeline is common else {}
    if k in (3, 200):
        prefix = "Registered few-shot K is" if pipeline is common else "ETO few-shot K must be"
        with pytest.raises(ValueError, match=f"^{prefix} 1/2/5/10/20/50/100$"):
            pipeline.prepare(args, config, **kwargs)
        assert checked == []
        assert not args.output.exists()
    else:
        identity = pipeline.prepare(args, config, **kwargs)
        assert identity["K"] == identity["unique_instances"] == k
        assert len(checked) == k
        assert (args.output / "identity.json").is_file()
        assert (args.output / "tasks.json").is_file()
