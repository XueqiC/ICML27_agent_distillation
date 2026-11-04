"""Parent-bound V0 choices and expansion weights, with sealed CPU fakes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from repair import v2_train as v2
from repair.artifacts import read_json
from sed.storage import digest, file_hash, write_json
from tools import train_v0_arm as trainer, v0_data, v0_groups as groups
from test_v0_training import corpus, seed_corpus, read, seal


def seal_inputs(c, paths):
    path = c["root"] / "manifest.json"
    doc = read_json(path)
    doc["inputs"].update({str(p): file_hash(p) for p in paths})
    doc["synthetic_manifest"] = str(c["synthetic"])
    write_json(path, doc)


@pytest.fixture
def expansion(corpus, tmp_path):
    base = corpus["rows"][0]
    for eid in ("repair-b", "repair-zero", "success-a"):
        row = deepcopy(base)
        row["example_id"] = eid
        row["v0"]["episode_id"] = eid
        if eid == "success-a":
            row["v0"].update(bucket="S", edit_decision=False)
            row["evidence"]["executor"] = "student"
        corpus["rows"].append(row)
    seal(corpus)
    seal_inputs(corpus, [])
    parent = read(corpus)
    inherited = v2.continuation_recipe(corpus["theta"], corpus["config"], corpus["snapshot"], 1)
    identity = {"dataset_sha256": parent["dataset_sha256"], "adapter_sha256": parent["adapter_sha256"],
        "parameter_names_sha256": digest(sorted(inherited["parameters"])), "features_index_sha256": digest("features")}
    group_doc = {"schema": "v0-groups/1", **identity, "assignment": {"repair-a": "g-a", "repair-b": "g-b"},
        "groups": [{"group_id": "g-a", "consistency": 2}, {"group_id": "g-b", "consistency": 4}],
        "excluded": {"repair-zero": "ZERO_GRADIENT"}}
    group_path, weight_path, module_path = [tmp_path / n for n in ("groups.json", "weights.json", "modules.json")]
    write_json(group_path, group_doc)
    write_json(weight_path, {"schema": "v0-episode-weights/1", **identity, "status": "valid",
        "weights": groups.episode_weights(group_doc, trainer.episode_entries(parent))})
    selected = [n for n in inherited["parameters"] if ".lora_B." in n][:1]
    write_json(module_path, {"schema": "v0-trainable-modules/1", **identity, "selected": selected})

    directory = tmp_path / "expand-D-data"
    synth = directory / "synth"
    synth.mkdir(parents=True)
    expanded = dict(corpus, root=directory / "dataset", synthetic=synth / "manifest.json", rows=deepcopy(corpus["rows"]))
    shutil.copytree(corpus["root"] / "replay", expanded["root"] / "replay")
    synthetic = read_json(corpus["synthetic"])
    generated, execution, units = [], [], []
    for i, gid in enumerate(("g-a", "g-b")):
        unit = {"unit": i, "group": gid, "source_episode_id": "repair-" + gid[-1]}
        units.append(unit)
        task = dict(synthetic["tasks"][0], task_id=f"train/synth/new-{i}", group_id=f"synthetic/new-{i}",
                    item_fingerprint=digest(["new", i]), expansion_unit=unit)
        generated.append(task)
        execution.append({"unit": unit, "generated_task_id": task["task_id"], "status": "filled"})
        for j in range(2 if i == 0 else 1):
            row = deepcopy(base)
            row.update(task_id=task["task_id"], group_id=task["group_id"])
            row["v0"].update(bucket="G", episode_id=f"G-{i}-{j}")
            # Deliberately unequal episode lengths: normalization is per episode.
            for step in range(1 if i == 0 else 3):
                current = deepcopy(row)
                current["example_id"] = f"new-{i}-{j}-{step}"
                current["evidence"]["step"] = step
                expanded["rows"].append(current)
    synthetic["tasks"].extend(generated)
    write_json(expanded["synthetic"], synthetic)
    for row in expanded["rows"]:
        row["v0"]["synthetic_manifest_sha256"] = file_hash(expanded["synthetic"])
    allocation_path = directory / "allocation.json"
    generated_path, execution_path = synth / "generated.json", synth / "allocation_execution.json"
    write_json(allocation_path, {"variant": "D-data", "units": units})
    write_json(generated_path, generated)
    write_json(execution_path, execution)
    seal(expanded)
    seal_inputs(expanded, [corpus["root"] / "manifest.json", corpus["root"] / "examples.jsonl",
                           group_path, weight_path, generated_path, execution_path])
    return dict(c=expanded, parent_c=corpus, parent=parent, inherited=inherited,
                weights=weight_path, modules=module_path, groups=group_path, selected=selected,
                allocation=allocation_path, generated=generated_path, execution=execution_path)


def load(e, **kwargs):
    data = read(e["c"])
    choices = trainer.load_choices(data, e["inherited"], **kwargs)
    return data, choices


def test_parent_weights_inherit_unit_groups_and_normalize_over_episodes(expansion):
    e = expansion
    data, (weights, selected, paths) = load(e, weights=e["weights"], modules=e["modules"])
    # Raw R+G masses: 2, 4, 1 (excluded), 2, 2, 4 -> mean 2.5.
    assert {eid: weights[eid] for eid in ("repair-a", "repair-b", "repair-zero", "G-0-0", "G-0-1", "G-1-0")} == {
        "repair-a": .8, "repair-b": 1.6, "repair-zero": .4, "G-0-0": .8, "G-0-1": .8, "G-1-0": 1.6}
    assert all(weights[e["episode_id"]] == 1 for e in trainer.episode_entries(data) if e["bucket"] in ("S", "D"))
    assert selected == e["selected"]
    assert data["parent_binding"]["dataset_sha256"] == e["parent"]["dataset_sha256"]
    mapping = data["weight_inheritance"]["G_episode_mapping"]
    assert mapping["G-1-0"] == dict(task_id="train/synth/new-1", unit=1, group="g-b",
        source_episode_id="repair-b", consistency=4, weight=1.6)
    assert {e["groups"], e["execution"], e["generated"], e["allocation"],
            e["parent_c"]["root"] / "manifest.json"} <= set(paths)


def test_parent_modules_do_not_require_g_mapping(expansion):
    e = expansion
    allocation = read_json(e["allocation"])
    allocation["variant"] = "D-param"
    write_json(e["allocation"], allocation)
    data, (weights, selected, _) = load(e, modules=e["modules"])
    assert weights == {} and selected == e["selected"]
    assert data["parent_binding"]["dataset_sha256"] == e["parent"]["dataset_sha256"]
    assert "weight_inheritance" not in data


def test_parent_binding_honors_read_only_relocation(expansion, tmp_path):
    e, c = expansion, expansion["c"]
    original = e["parent_c"]["root"]
    sealed_prefix = tmp_path / "producer-host"
    sealed_prefix.mkdir()
    sealed_parent = sealed_prefix / "dataset"
    shutil.move(original, sealed_parent)
    manifest_path = c["root"] / "manifest.json"
    manifest = read_json(manifest_path)
    for name in ("manifest.json", "examples.jsonl"):
        manifest["inputs"][str(sealed_parent / name)] = manifest["inputs"].pop(str(original / name))
    write_json(manifest_path, manifest)
    moved_prefix = tmp_path / "relocated-host"
    shutil.move(sealed_prefix, moved_prefix)
    moved = moved_prefix / "dataset"
    data = v0_data.read_dataset(c["root"], theta0=c["theta"], snapshot=c["snapshot"], config=c["config"],
        synthetic_manifest=c["synthetic"], relocate=[f"{sealed_prefix}={moved_prefix}"])
    weights, _, _ = trainer.load_choices(data, e["inherited"], weights=e["weights"])
    assert data["parent_binding"]["path"] == str(moved)
    assert data["parent_binding"]["dataset_sha256"] == e["parent"]["dataset_sha256"]
    assert weights["G-0-0"] == .8 and not original.exists()


@pytest.mark.parametrize("artifact", ["weights", "modules"])
@pytest.mark.parametrize("field", ["dataset_sha256", "adapter_sha256", "parameter_names_sha256", "smoke_subset"])
def test_parent_choices_reject_unrelated_identity(expansion, artifact, field):
    e = expansion
    doc = read_json(e[artifact])
    doc[field] = True if field == "smoke_subset" else digest("unrelated")
    write_json(e[artifact], doc)
    seal_inputs(e["c"], [e[artifact]])
    with pytest.raises(ValueError, match="mismatch|Smoke-subset"):
        load(e, **{artifact: e[artifact]})


@pytest.mark.parametrize("mutation", ["missing_parent", "parent_manifest", "parent_examples", "changed_inheritance"])
def test_parent_binding_requires_existing_sealed_unchanged_dataset(expansion, mutation):
    e = expansion
    parent = e["parent_c"]["root"]
    if mutation == "missing_parent":
        (parent / "manifest.json").unlink()
    elif mutation.startswith("parent_"):
        path = parent / ("manifest.json" if mutation == "parent_manifest" else "examples.jsonl")
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        e["c"]["rows"][0]["source"] = "changed"
        path = e["c"]["root"] / "examples.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in e["c"]["rows"]))
        doc = read_json(e["c"]["root"] / "manifest.json")
        doc["examples_sha256"] = file_hash(path)
        write_json(e["c"]["root"] / "manifest.json", doc)
    with pytest.raises(ValueError, match="missing file|sha256 mismatch|changed parent rows"):
        load(e, weights=e["weights"])


@pytest.mark.parametrize("mutation", ["missing_task", "unknown_group", "wrong_source", "conflicting_unit", "duplicate_task"])
def test_unresolved_or_conflicting_g_mapping_refused(expansion, mutation):
    e = expansion
    execution = read_json(e["execution"])
    if mutation == "missing_task":
        execution.pop(0)
    elif mutation == "duplicate_task":
        execution[1]["generated_task_id"] = execution[0]["generated_task_id"]
    elif mutation == "conflicting_unit":
        execution[0]["unit"]["group"] = "conflict"
    else:
        # All provenance agrees, but the group/source cannot resolve to the parent.
        allocation, generated = read_json(e["allocation"]), read_json(e["generated"])
        synthetic = read_json(e["c"]["synthetic"])
        field, value = ("group", "unknown") if mutation == "unknown_group" else ("source_episode_id", "repair-b")
        for unit in (execution[0]["unit"], allocation["units"][0], generated[0]["expansion_unit"],
                     synthetic["tasks"][1]["expansion_unit"]):
            unit[field] = value
        write_json(e["allocation"], allocation)
        write_json(e["generated"], generated)
        write_json(e["c"]["synthetic"], synthetic)
        for row in e["c"]["rows"]:
            row["v0"]["synthetic_manifest_sha256"] = file_hash(e["c"]["synthetic"])
        manifest = read_json(e["c"]["root"] / "manifest.json")
        seal(e["c"])
        seal_inputs(e["c"], list(map(Path, manifest["inputs"])))
    write_json(e["execution"], execution)
    seal_inputs(e["c"], [e["execution"]])
    with pytest.raises(ValueError, match="Unresolved G|differs from allocation|Duplicate.*expansion task"):
        load(e, weights=e["weights"])


def test_inherited_mapping_is_written_to_check_and_run_provenance(expansion, tmp_path):
    e, c = expansion, expansion["c"]
    args = ["--dataset", str(c["root"]), "--theta0", str(c["theta"]), "--snapshot", str(c["snapshot"]),
            "--config", str(c["config_path"]), "--synthetic-manifest", str(c["synthetic"]),
            "--weights", str(e["weights"])]
    checked = tmp_path / "checked.json"
    assert trainer.main(["inputs", "--check", *args, "--output", str(checked)]) == 0
    check = read_json(checked)
    assert check["buckets"]["G"] == dict(rows=5, episodes=3, weight_min=.8, weight_max=1.6, weight_mean=3.2 / 3)
    output = tmp_path / "trained"
    assert trainer.main(["run", *args, "--output", str(output), "--updates", "1", "--checkpoints", "1", "--dry-run"]) == 0
    manifest, provenance = [read_json(output / name) for name in ("manifest.json", "provenance.json")]
    assert manifest["weight_inheritance"] == check["weight_inheritance"]
    assert provenance["details"]["weight_inheritance"] == check["weight_inheritance"]
    assert manifest["teacher_calls_this_run"] == manifest["training_exposure"] == 0
