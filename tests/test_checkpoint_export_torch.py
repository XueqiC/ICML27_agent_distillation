"""CPU regressions for base-file naming, tied aliases, and frozen tower export."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from safetensors.torch import load_file, save_file
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

from baselines.common.checkpoint import checkpoint_key_map, checkpoint_layout
from baselines.common.fsdp_train import export_checkpoint, text_parameter, validate_checkpoint
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, write_json
from test_stage2b_torch import tiny_model


def test_real_snapshot_meta_export_key_set():
    snapshot = Path("/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/"
                    "snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        pytest.skip("Pinned real Gemma4 snapshot is absent")
    config = Gemma4UnifiedConfig.from_pretrained(snapshot, local_files_only=True)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Gemma4UnifiedForConditionalGeneration(config)
    state = model.state_dict()
    base = checkpoint_layout(snapshot)
    mapping = checkpoint_key_map(model, base, state_dict=state)
    assert len(state) == 678 and len(base) == 677
    assert set(mapping.values()) == set(base)
    assert state.keys() - mapping.keys() == {"lm_head.weight"}
    assert model.lm_head.weight is model.get_input_embeddings().weight
    assert sum(name != key for name, key in mapping.items()) == 10
    for name, key in mapping.items():
        assert state[name].device.type == "meta"
        assert list(state[name].shape) == base[key]["shape"]
        if name.startswith("model.language_model."):
            assert name == key


def synthetic_model(*, tied=True):
    config = tiny_model().config
    config.tie_word_embeddings = config.text_config.tie_word_embeddings = tied
    model = Gemma4UnifiedForConditionalGeneration(config)
    # Tiny synthetic tower uses real Transformers class conversion rules.
    tower = torch.nn.Module()
    tower.patch_dense = torch.nn.Linear(3, 4, bias=False)
    tower.multimodal_embedder = torch.nn.Module()
    tower.multimodal_embedder.embedding_projection = torch.nn.Linear(4, 2, bias=False)
    model.model.embed_vision = tower
    return model


def read_tensors(path):
    result = {}
    for shard in sorted(path.glob("*.safetensors")):
        result.update(load_file(shard))
    return result


def finalize_checkpoint(path, report):
    student = {"requested_repo": "google/gemma-4-12B-it", "revision": "test-only"}
    provenance = {"config": {"models": {"student": student}}}
    write_json(path.parent / "provenance.json", provenance)
    state = {**report, "final": True, "provenance_sha256": digest(provenance)}
    write_json(path / "training_state.json", state)
    return student


@pytest.mark.parametrize("tied,store_alias", [(True, False), (True, True), (False, True)])
def test_renamed_tied_export_keeps_base_bytes_without_gather(tmp_path, monkeypatch, tied, store_alias):
    torch.set_num_threads(1)
    model = synthetic_model(tied=tied)
    base, output = tmp_path / "base", tmp_path / "checkpoint"
    model.save_pretrained(base)
    base_tensors = read_tensors(base)
    assert "model.vision_embedder.patch_dense.weight" in base_tensors
    assert "model.embed_vision.embedding_projection.weight" in base_tensors
    if tied and store_alias:
        base_tensors["lm_head.weight"] = model.lm_head.weight.detach().clone()
        save_file(base_tensors, base / "model.safetensors")
    assert ("lm_head.weight" in base_tensors) is store_alias

    # A gather or a copy from any in-memory non-text value must fail. The
    # persisted base, not these placeholders (nor changed tower weights), wins.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.add_(10)
    state = model.state_dict()
    gathered = []
    class Sharded:
        def __init__(self, name, tensor):
            self.name, self.tensor, self.shape = name, tensor, tensor.shape

        def full_tensor(self):
            assert text_parameter(self.name), "Frozen tensors must never be gathered"
            assert self.name != "lm_head.weight" or store_alias
            gathered.append(self.name)
            return self.tensor

    monkeypatch.setattr(model, "state_dict", lambda: {n: Sharded(n, t) for n, t in state.items()})
    report = export_checkpoint(model, output, base, max_shard_bytes=2000)
    saved = read_tensors(output)
    assert saved.keys() == base_tensors.keys()
    mapping = checkpoint_key_map(model, base_tensors)
    assert set(gathered) == {n for n in mapping if text_parameter(n)}
    assert set(report["non_text_tensor_sha256"]) == {k for n, k in mapping.items() if not text_parameter(n)}
    for name, key in mapping.items():
        if text_parameter(name):
            assert saved[key].dtype == torch.bfloat16
            torch.testing.assert_close(saved[key], state[name].to(torch.bfloat16), rtol=0, atol=0)
        else:
            assert saved[key].shape == base_tensors[key].shape
            assert saved[key].dtype == base_tensors[key].dtype
            assert saved[key].view(torch.uint8).numpy().tobytes() == base_tensors[key].view(torch.uint8).numpy().tobytes()
    student = finalize_checkpoint(output, report)
    validate_checkpoint(output, student, base_snapshot=base)

    # All ranks gather only stored text tensors; nonzero ranks write nothing.
    gathered.clear()
    assert export_checkpoint(model, tmp_path / "rank1", base, rank=1) is None
    assert set(gathered) == {n for n in mapping if text_parameter(n)}
    assert not (tmp_path / "rank1").exists()


def test_key_map_rejects_missing_untied_and_colliding_names(tmp_path):
    model = synthetic_model(tied=False)
    base = tmp_path / "base"
    model.save_pretrained(base)
    keys = set(checkpoint_layout(base))
    with pytest.raises(ValueError, match="Untied model tensor"):
        checkpoint_key_map(model, keys - {"lm_head.weight"})
    with pytest.raises(ValueError, match="missing from model"):
        checkpoint_key_map(model, keys | {"model.unknown.weight"})
    with pytest.raises(ValueError, match="conversion collision"):
        checkpoint_key_map(model, keys | {"model.embed_vision.patch_dense.weight"})

    model = synthetic_model(tied=True)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="Untied model tensor"):
        checkpoint_key_map(model, keys - {"lm_head.weight"})


@pytest.mark.parametrize("change", ["missing", "extra_tied", "renamed", "shape", "dtype", "index"])
def test_validate_checkpoint_rejects_bad_layout_even_with_updated_hashes(tmp_path, change):
    model = synthetic_model()
    base, output = tmp_path / "base", tmp_path / "checkpoint"
    model.save_pretrained(base)
    report = export_checkpoint(model, output, base)
    student = finalize_checkpoint(output, report)
    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "model.vision_embedder.patch_dense.weight"
    shard = output / index["weight_map"][key]
    tensors = load_file(shard)
    if change == "missing":
        del tensors[key]
    elif change == "extra_tied":
        tensors["lm_head.weight"] = tensors["model.language_model.embed_tokens.weight"].clone()
    elif change == "renamed":
        tensors["model.embed_vision.patch_dense.weight"] = tensors.pop(key)
    elif change == "shape":
        tensors[key] = tensors[key].reshape(-1)
    elif change == "dtype":
        tensors[key] = tensors[key].to(torch.float16)
    else:
        index["weight_map"][key] = "missing-shard.safetensors"
    save_file(tensors, shard)
    if change != "index":
        index["weight_map"] = {k: shard.name for k in tensors}
    write_json(index_path, index)
    state = json.loads((output / "training_state.json").read_text())
    state["artifact_sha256"] = artifact_hashes(output)
    del state["artifact_sha256"]["training_state.json"]
    write_json(output / "training_state.json", state)
    error = "shape/dtype" if change in {"shape", "dtype"} else "index differs" if change == "index" else "key set differs"
    with pytest.raises(ValueError, match=error):
        validate_checkpoint(output, student)


@pytest.mark.parametrize("merged", [False, True])
def test_serving_rejects_full_checkpoint_key_mismatch_before_launch(tmp_path, monkeypatch, merged):
    import harness.serve_student as server
    model = synthetic_model()
    base, output = tmp_path / "base", tmp_path / "checkpoint"
    model.save_pretrained(base)
    report = export_checkpoint(model, output, base)
    student = finalize_checkpoint(output, report)
    # The hashes still certify the exported artifacts, but this base key set
    # cannot be the export's schema. Both launcher path forms must reject it.
    save_file({"unexpected.weight": torch.zeros(1)}, base / "model.safetensors")
    monkeypatch.setattr(server, "load_config", lambda _: {"models": {"student": student}})
    monkeypatch.setattr(server, "inspect_runtime", lambda _: {"gemma4_parser": False})
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def bind(self, _): pass
    monkeypatch.setattr(server.socket, "socket", lambda *_: Socket())
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid checkpoint must never launch a server")
    monkeypatch.setattr(server.subprocess, "Popen", forbidden)
    args = ["--python", "unused", "--model-path", str(base if merged else output),
            "--port", "23456", "--output", str(tmp_path / "server")]
    if merged:
        args += ["--merged-model-path", str(output)]
    with pytest.raises(ValueError, match="key set differs"):
        server.main(args)
