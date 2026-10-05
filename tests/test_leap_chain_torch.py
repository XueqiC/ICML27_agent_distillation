"""Real CPU train/export/merge/retrain chain; only teacher/environment are fake."""
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import Gemma4UnifiedForConditionalGeneration, Gemma4UnifiedVisionConfig, PreTrainedTokenizerFast

from baselines.common.explore import explore_one, run_exploration
from baselines.common.fsdp_train import text_parameter, validate_checkpoint
from baselines.leap import merge, pipeline, train
from baselines.leap.data import prepare_set, seeds
from baselines.leap.protocol import render, trajectory
from baselines.leap.teacher import TeacherBudget, read
from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.dev_subset import build_dev40
from sed.seed_evidence import artifact_hashes
from sed.seed_selection import select_seeds
from sed.storage import digest, file_hash, write_json
from students.vllm_student import Student
from teachers.collect_seeds import collect
from train.dataset import load_tokenizer
from train.merge_lora import LEAP_MERGE_POLICY
from train.modeling import load_student
from test_stage2b_seeds import seed_bank, factories
from test_stage2b_torch import tiny_model
from test_stage3d import STATE, correction, factory
from test_teacher import FakeClient, response


def tiny_snapshot(path):
    # A real, round-trippable byte tokenizer with Gemma's required turn ID.
    # This diagnostic fixture needs no downloaded checkpoint/tokenizer assets.
    specials = ["[PAD]", "[BOS]", "[UNK]", "<|turn>", "<|channel>", "<channel|>"]
    vocabulary = specials + sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary.insert(106, "<turn|>")
    backend = Tokenizer(models.BPE({s: i for i, s in enumerate(vocabulary)}, [], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
        unk_token="[UNK]", eos_token="<turn|>", additional_special_tokens=specials[3:])
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}{{ '<|turn>' + "
        "('model' if message['role'] == 'assistant' else message['role']) + '\n' + "
        "message['content'] | trim + '<turn|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|turn>model\n<|channel>thought\n<channel|>' }}{% endif %}")
    config = tiny_model().config
    config.text_config.vocab_size = len(tokenizer)
    config.text_config.max_position_embeddings = 4096
    config.vision_config = Gemma4UnifiedVisionConfig(patch_size=2, pooling_kernel_size=1,
        mm_embed_dim=32, mm_posemb_size=8, output_proj_dims=32)
    model = Gemma4UnifiedForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    # Serving assets that model/tokenizer.save_pretrained do not themselves emit.
    write_json(path / "processor_config.json", {"processor_class": "Gemma4UnifiedProcessor"})
    write_json(path / "preprocessor_config.json", {"image_processor_type": "Gemma4UnifiedImageProcessor"})
    (path / "alternate_template.jinja2").write_text(tokenizer.chat_template)
    (path / "tokenizer.model").write_bytes(b"auxiliary tokenizer asset\x00")
    (path / "vocab.txt").write_text("auxiliary vocabulary\n")
    return tokenizer


def test_real_leap_cpu_chained_iterations(seed_bank, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    monkeypatch.setenv("WORLD_SIZE", "1")
    root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 40
    config["paths"].update(manifest=str(tmp_path / "manifest.json"), split_freeze=str(tmp_path / "freeze.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    base = tmp_path / "base"
    tiny_snapshot(base)
    base_tensors = load_file(base / "model.safetensors")
    towers = {k: v for k, v in base_tensors.items() if not text_parameter(k)}
    assert towers  # Do not accidentally test a text-only layout.

    env, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collect(select_seeds(manifest, freeze, k=2), manifest, freeze, config, verified,
        env_factory=env, teacher_factory=teacher)
    bank = seeds(verified, config)
    tasks = [task for task, _ in bank]
    ids = [task["task_id"] for task in tasks]
    listing = tmp_path / "tasks.json"
    write_json(listing, {"task_ids": ids, "list_sha256": digest(ids), "manifest_sha256": digest(manifest),
        "split_freeze_sha256": digest(freeze), "parent_seed_ids": {t["task_id"]: t["parent_seed_ids"] for t in tasks}})
    dev_tasks = tmp_path / "dev40.json"
    write_json(dev_tasks, build_dev40(manifest, freeze))

    # Execute the launcher's actual preflight Python, including its base argument.
    launcher = Path("scripts/eval_adapters.sh").read_text()
    preflight = launcher.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    args = SimpleNamespace(output=str(tmp_path / "run"), variant="privileged", correction_tokens=4096)
    identity = {"K": 2, "recipe": {"variant": "privileged"}}
    extractor = TeacherBudget(tmp_path / "extraction", tasks=tasks, config=config, kind="extract",
        iteration=0, run_identity=digest(identity), max_output_tokens=4096,
        teacher_factory=factory(FakeClient([response(STATE) for _ in tasks])))
    states = tmp_path / "states"
    extracted = []
    for task, ep in bank:
        outcome = extractor.request(task, render("extract", trajectory(ep, expert=True)))
        extracted.append({"task_id": task["task_id"], "state": outcome["labels"], "status": outcome["status"],
            "request_directory": outcome["directory"],
            "outcome_sha256": file_hash(Path(outcome["directory"]) / "outcome.json")})
    write_json(states / "states.json", extracted)

    class FailedBackend(FakeBackend):
        def step(self, action):
            self.position += 1
            state = self.state()
            state.update(won=False, lost=self.position == 2)
            return state, 0., state["lost"]

    def worker(job):
        return explore_one(job, env_factory=lambda c, t: Environment(root, max_episode_steps=40,
            backend_factory=FailedBackend), policy_factory=lambda **kw: Student(**kw,
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
                    "model": config["models"]["student"]["requested_repo"],
                    "choices": [{"message": {"content": "Thought: inspect\nAction: look"},
                                 "finish_reason": "stop"}]}))))

    client = FakeClient([response(correction(action="look")) for _ in tasks])
    monkeypatch.setattr(pipeline, "TeacherBudget", lambda *a, **kw:
        TeacherBudget(*a, teacher_factory=factory(client), **kw))
    current, sets = base, []
    for iteration in range(2):
        evidence = {}
        if iteration:
            exploration = tmp_path / "exploration"
            run_exploration(config=config, tasks_path=listing, output=exploration,
                spec={"endpoint": "http://fake/v1", "model": config["models"]["student"]["requested_repo"],
                      "revision": config["models"]["student"]["revision"], "enable_thinking": False,
                      "max_tokens": 1024}, decoding="greedy", worker_fn=worker)
            corrections = tmp_path / "corrections"
            pipeline.correct_rollouts(args, config, identity, tasks, exploration, states, iteration, corrections)
            evidence.update(exploration=exploration, corrections=corrections)
        dataset = tmp_path / f"data-{iteration}/output"
        prepare_set(verified, dataset, config, iteration=iteration, **evidence)
        sets.append(str(dataset / "examples.json"))
        trained = tmp_path / f"train-{iteration}/output"
        # Real entry point: dataset validation/tokenization, model load, fresh
        # rank-128 q/v LoRA, optimizer updates, adapter and provenance publication.
        train.main(["--config", str(config_path), "--snapshot", str(current), "--base-snapshot", str(base),
            "--verified", str(verified), "--output", str(trained), "--iteration", str(iteration),
            "--variant", "privileged", "--sets", *sets], device="cpu")
        training_state = read(trained / "adapter/training_state.json")
        assert training_state["updates"] >= 2
        assert training_state["examples_seen"] == 2 * (iteration + 1)
        weights = load_file(trained / "adapter/adapter_model.safetensors")
        assert any(torch.count_nonzero(v) for k, v in weights.items() if "lora_B" in k)

        merged = tmp_path / f"merge-{iteration}/output"
        merge.main(["--config", str(config_path), "--snapshot", str(current),
            "--adapter", str(trained / "adapter"), "--output", str(merged)], device="cpu")
        checkpoint = merged / "checkpoint"
        state = read(checkpoint / "training_state.json")
        assert "non_text_source" not in state  # Reproduce the real merge schema.
        assert state == read(merged / "result.json")
        assert state["merge_passed"] and state["policy"] == LEAP_MERGE_POLICY
        assert state["leap_iteration"] == iteration and state["fresh_lora"]
        assert state["provenance_sha256"] == digest(read(merged / "provenance.json"))
        hashes = artifact_hashes(checkpoint)
        hashes.pop("training_state.json")
        assert hashes == state["artifact_sha256"]
        consistency = read(checkpoint / "consistency.json")
        assert consistency["passed"] and consistency["reload_consistency"]["passed"]
        assert consistency["base"]["path"] == str(current.resolve())
        assert consistency["merged_sha256"] == {k: v for k, v in hashes.items() if k != "consistency.json"}
        own_provenance = read(checkpoint / "provenance.json")
        assert own_provenance["command"] == "merge_lora"
        assert own_provenance["config"]["merge"]["snapshot"] == str(current)
        assert read(merged / "provenance.json")["command"] == "leap_checked_merge"
        for name in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja", "processor_config.json",
                     "preprocessor_config.json", "alternate_template.jinja2", "tokenizer.model", "vocab.txt"):
            assert (checkpoint / name).is_file(), name
        for asset in consistency["copied_base_assets"]:
            assert (checkpoint / asset).read_bytes() == (base / asset).read_bytes()
        assert "processor_config.json" in consistency["copied_base_assets"]
        assert not {"training_state.json", "provenance.json", "consistency.json"} & set(consistency["copied_base_assets"])
        load_tokenizer(checkpoint)
        exported = load_file(checkpoint / "model.safetensors")
        assert exported.keys() == base_tensors.keys()
        for key, tensor in towers.items():
            torch.testing.assert_close(exported[key], tensor, rtol=0, atol=0)
        train.validate_initialization(checkpoint, base, iteration=iteration + 1,
            variant="privileged", student=config["models"]["student"])
        # This is the same validator and explicit base used by serve_student
        # for --merged-model-path; no vLLM process or HTTP server is launched.
        assert validate_checkpoint(checkpoint, config["models"]["student"], base_snapshot=base) == state
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", ["-", str(config_path), str(base), str(dev_tasks), str(checkpoint)])
            exec(compile(preflight, "scripts/eval_adapters.sh:preflight", "exec"), {"__name__": "__main__"})
        fresh = train.prepare_model(checkpoint, device="cpu")
        assert all(torch.count_nonzero(p) == 0 for n, p in fresh.named_parameters() if "lora_B" in n)
        standalone = load_student(checkpoint, device="cpu", dtype="float32")
        torch.testing.assert_close(fresh.get_base_model().model.language_model.embed_tokens.weight,
                                  standalone.model.language_model.embed_tokens.weight, rtol=0, atol=0)
        del fresh, standalone
        current = checkpoint

    # Both provenance locations and every serving asset remain hash-bound.
    for path in (current.parent / "provenance.json", current / "provenance.json", current / "processor_config.json"):
        original = path.read_bytes()
        write_json(path, {"tampered": True})
        with pytest.raises(ValueError, match="artifacts/provenance"):
            train.validate_initialization(current, base, iteration=2, variant="privileged", student=config["models"]["student"])
        path.write_bytes(original)
    # Even an updated artifact receipt cannot bless a mismatched non-text tower.
    tensors = load_file(current / "model.safetensors")
    key = next(iter(towers))
    tensors[key] = tensors[key].to(torch.float32)
    save_file(tensors, current / "model.safetensors")
    state = read(current / "training_state.json")
    state["artifact_sha256"] = artifact_hashes(current)
    state["artifact_sha256"].pop("training_state.json")
    write_json(current / "training_state.json", state)
    with pytest.raises(ValueError, match="Non-text checkpoint shape/dtype"):
        train.validate_initialization(current, base, iteration=2, variant="privileged", student=config["models"]["student"])
