"""CPU-only IPR summed chosen NLL, gradients and Gemma continuation masks."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from test_stage2b_torch import tokenizer, trajectory, tiny_model
from test_stage3a_torch import rows
from baselines.common.conversation import episode_messages
from baselines.common.fsdp_train import batch_logps, dpo_objective, train_loop
from baselines.ipr.pairs import build_pairs, preference_conversation


def official_loss():
    source = Path("third_party/IPR/fastchat/train/dpo_trainer.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name == "DPOMultiTrainer")
    cls.body = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "dpo_with_sft_loss"]
    cls.bases, cls.decorator_list = [], []
    scope = {"torch":torch,"F":torch.nn.functional,"Tuple":Tuple}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),"isolated_IPR_loss","exec"),scope)
    return scope["DPOMultiTrainer"]


def test_ipr_nll_matches_official_and_sum_gradient(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = tiny_model()
    model.eval()
    reference = deepcopy(model).eval().requires_grad_(False)
    # A distinct reference, not merely equal logits, catches ratio mistakes.
    with torch.no_grad():
        next(reference.parameters()).add_(.03)
    pair = dict(zip(("chosen","rejected"),rows()))
    loss,metrics = dpo_objective(model,reference,pair,"cpu",beta=.1,nll_weight=1)
    policy = batch_logps(model,rows(),"cpu")
    with torch.no_grad():
        ref = batch_logps(reference,rows(),"cpu")
    setting = SimpleNamespace(beta=.1,label_smoothing=0.,loss_type="sigmoid")
    expected = official_loss().dpo_with_sft_loss(setting,policy[:1],policy[1:],ref[:1],ref[1:])[0].mean()
    torch.testing.assert_close(loss,expected)
    pure,_ = dpo_objective(model,reference,pair,"cpu",beta=.1)
    torch.testing.assert_close(loss-pure,-policy[0])
    torch.testing.assert_close(metrics["loss/chosen_nll"],-policy[0].detach())
    parameter = model.model.language_model.layers[0].self_attn.q_proj.weight
    grad_actual, = torch.autograd.grad(loss,parameter,retain_graph=True)
    grad_expected, = torch.autograd.grad(expected,parameter)
    torch.testing.assert_close(grad_actual,grad_expected)
    assert all(p.grad is None for p in reference.parameters())
    frozen = {n:p.detach().clone() for n,p in reference.named_parameters()}
    state = train_loop(model,[pair],tmp_path / "train",objective="ipr",reference=reference,
        learning_rate=3e-6,scheduler="constant_with_warmup",warmup_ratio=.1,original_batch=48,epochs=3,seed=0)
    assert state["examples_seen"] == 3 and state["b_prime"] == 1
    assert state["loss_reduction"] == "sigmoid_plus_chosen_nll_summed_logps"
    assert state["physical_reference_forward_tokens"] > 0
    assert all(torch.equal(frozen[n],p) for n,p in reference.named_parameters())


def test_ipr_whole_continuation_masks_prefix_and_tracks_reversed_sources(tokenizer):
    expert = trajectory()
    expert["steps"] = [dict(expert["steps"][0],action_raw=f"Thought: step {i}\nAction: look") for i in range(4)]
    agent = deepcopy(expert)
    agent["steps"] = agent["steps"][:2]
    agent["steps"][-1]["action_raw"] = "Thought: shorter\nAction: look"
    pairs,report = build_pairs([{"task_id":expert["task_id"],"messages":episode_messages(expert),
        "parent_seed_ids":["seed"],"final_reward":1,"mc":{1:.2}}],
        [{"task_id":expert["task_id"],"messages":episode_messages(agent),"final_reward":1,"mc":.8,"j":1,
          "episode_id":"student"}])
    assert report["agent_chosen"] == 1
    encoded = preference_conversation(pairs[0],tokenizer)
    n = len(encoded["prompt_input_ids"])
    for side,turns in (("chosen",1),("rejected",3)):
        group = encoded[side]
        assert len(group["turns"]) == group["turn_count"] == turns
        assert group["supervised_tokens"] == sum(t["supervised_tokens"] for t in group["turns"])
        for i, row in enumerate(group["turns"]):
            assert row["labels"][:n] == [-100]*n
            assert row["input_ids"][:n] == encoded["prompt_input_ids"]
            prompt = pairs[0]["prompt"] + pairs[0][side][:2*i]
            text = tokenizer.apply_chat_template(prompt, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert row["input_ids"][:row["prompt_tokens"]] == ids
            assert row["labels"][:len(ids)] == [-100] * len(ids)
            assert row["assistant_spans"][0]["turn_index"] == i + pairs[0]["j"]
            assert row["supervised_tokens"] == sum(v != -100 for v in row["labels"])
            for span in row["assistant_spans"]:
                assert row["labels"][span["stop"]-1] == 106
                assert span["start"] >= n
            text = tokenizer.decode([v for v in row["labels"] if v != -100])
            assert "step 0" not in text and "UNIQUE_OBSERVATION_SENTINEL" not in text
            assert "PRIVATE_PDDL" not in tokenizer.decode(row["input_ids"])
    assert encoded["chosen"]["label_provenance"]["source"] == "student_exploration"
    assert encoded["rejected"]["label_provenance"]["source"] == "trusted_teacher_seed"
    with pytest.raises(ValueError,match="no truncation"):
        preference_conversation(pairs[0],tokenizer,max_length=1)
