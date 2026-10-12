import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
torch = pytest.importorskip("torch")

from test_crn_tail import warm_toy, assert_state_equal
from tools.crn_tail import message_backward, surrogate_gradient
from tools.grad_stats import (gradient_statistics, load_rollout_examples, rollout_examples,
                              sampled_reply_example, squared_norm)
from train.optimizer_state import adamw, capture_state
from test_r03_sets import TinyTokenizer
from test_rollout_pool import sampled_rollout


def test_example_second_moment_and_rankers(warm_toy):
    model, optimizer, state, rows = warm_toy
    result = gradient_statistics(model, optimizer, state, rows, rows,
        candidates={"same": {"episodes": [0, 1]}, "large": {"episodes": [0], "loss_multiplier": 100}},
        example_sets={"cluster": {"episodes": [0, 1]}})
    same = result["candidates"]["same"]
    assert same["cosine_loss_gradient_with_surrogate"]["unclipped"] == pytest.approx(-1)
    assert same["parameter_first_order_score"] > 0
    large = result["candidates"]["large"]
    assert large["gradient_norm"]["clipped"] == pytest.approx(1, abs=1e-6)
    assert large["stats"]["clip_active"]
    moments = result["example_sets"]["cluster"]
    expected = []
    for i in range(2):
        grad, _ = message_backward(model, {"episodes": [i]}, rows)
        expected.append(squared_norm(grad) * rows[i]["supervised_tokens"] ** 2)
    assert moments["sigma_squared_unclipped"] == pytest.approx(sum(expected) / 2)
    assert result["clusters"]["cluster"]["sigma2"] == moments["sigma_squared_unclipped"]
    batch_gradient, _ = message_backward(model, {"episodes": [0, 1]}, rows)
    assert moments["sigma_squared_unclipped"] > squared_norm(batch_gradient)
    assert_state_equal(capture_state(model, optimizer), state)
    json.dumps(result, allow_nan=False)


def test_surrogate_gradient_direction_and_finite_difference(warm_toy):
    model, _, _, rows = warm_toy
    dF = surrogate_gradient(model, rows)["theta"]
    from tools.crn_tail import surrogate_score
    original = model.theta.detach().clone()
    estimates = []
    with torch.no_grad():
        for i in range(4):
            model.theta.copy_(original)
            model.theta[i] += .005
            plus = surrogate_score(model, rows)["F_tilde_per_token"]
            model.theta[i] -= .01
            minus = surrogate_score(model, rows)["F_tilde_per_token"]
            estimates.append((plus - minus) / .01)
        model.theta.copy_(original)
    torch.testing.assert_close(dF, torch.tensor(estimates, dtype=torch.float64), atol=2e-5, rtol=2e-4)


def test_external_examples_and_diagnostic_exclusion(warm_toy):
    model, optimizer, state, rows = warm_toy
    result = gradient_statistics(model, optimizer, state, rows, rows,
        candidates={"external": {"episodes": {"root": "pool", "ids": [1]}}}, datasets={"pool": rows})
    assert result["candidates"]["external"]["stats"]["supervised_tokens"] == 3
    with pytest.raises(ValueError, match="Diagnostic"):
        gradient_statistics(model, optimizer, state, rows, rows,
            candidates={"bad": {"episodes": {"root": "surrogate", "ids": [0]}}},
            datasets={"surrogate": [{**rows[0], "label_provenance": {"training_eligible": False}}]})
    assert_state_equal(capture_state(model, optimizer), state)


def test_rollout_encoding_masks_history_and_scores_raw_samples(tmp_path, data_root):
    _, _, _, _, records, _, _ = sampled_rollout(tmp_path, data_root)
    record = deepcopy(records[0][0])
    for state in record["states"]:
        state["messages"] = [{"role": "user", "content": "SECRET_SAVED_MESSAGES"}]
        state["visible_context"]["extra.expert_plan"] = "SECRET_ORACLE"
        state["success"] = "SECRET_LABEL"
    tokenizer = TinyTokenizer()
    sets = rollout_examples([record], tokenizer)
    rows = next(iter(sets.values()))
    assert len(rows) == 4
    for row in rows:
        assert all(t == -100 for t in row["labels"][:row["prompt_tokens"]])
        text = tokenizer.decode(row["input_ids"], skip_special_tokens=False)
        target = tokenizer.decode([i for i in row["labels"] if i != -100], skip_special_tokens=False)
        expected = " Action: go north \n<turn|>" if row["label_provenance"]["sample_index"] == 0 else "malformed"
        assert target == expected  # Preserve whitespace and don't invent EOS for a cap finish.
        assert "SECRET" not in text and not row["label_provenance"]["training_eligible"]
        assert row["supervised_tokens"] == len(row["input_ids"]) - row["prompt_tokens"]
    one = rollout_examples([record], tokenizer, states_per_cluster=1)
    assert len(next(iter(one.values()))) == 2
    assert one == rollout_examples([record], tokenizer, states_per_cluster=1)


@pytest.mark.parametrize("fault", ["sealed", "missing", "failed", "greedy", "thinking", "duplicate"])
def test_rollout_label_guards(tmp_path, data_root, fault):
    _, _, _, _, records, _, _ = sampled_rollout(tmp_path, data_root)
    row = deepcopy(records[0][0])
    first = row["states"][0]
    if fault == "sealed":
        row["sealed"] = True
    elif fault == "missing":
        first["label_samples"] = []
    elif fault == "failed":
        first["label_samples"][0]["status"] = "failed"
    elif fault == "greedy":
        first["label_samples"][0]["metadata"]["temperature"] = 0
    elif fault == "thinking":
        first["label_samples"][0]["metadata"]["thinking_chars"] = 10
    else:
        row["states"].append(first)
    with pytest.raises(ValueError):
        rollout_examples([row], TinyTokenizer())


def test_fisher_trace_equals_exact_expectation_for_tiny_categorical(warm_toy):
    model, optimizer, state, _ = warm_toy
    # Uniform categorical student: exhaust all four one-token labels. Averaging
    # their score norms is exactly the Fisher trace, 1 - sum_a p(a)^2 = 3/4.
    with torch.no_grad():
        model.theta.zero_()
    state = capture_state(model, optimizer)
    rows = [{"input_ids": [0, action], "labels": [-100, action], "supervised_tokens": 1,
             "label_provenance": {"state_hash": "same", "training_eligible": False}}
            for action in range(4)]
    result = gradient_statistics(model, optimizer, state, [], None, sampled_sets={"cluster": rows})
    assert result["clusters"]["cluster"]["sigma2"] == pytest.approx(.75)
    assert result["clusters"]["cluster"]["sample_count"] == 4
    assert result["clusters"]["cluster"]["state_count"] == 1
    assert_state_equal(capture_state(model, optimizer), state)


def test_sampled_state_mean_is_not_label_count_weighted(warm_toy):
    model, optimizer, state, rows = warm_toy
    a = {**rows[0], "label_provenance": {"state_hash": "a"}}
    b = {**rows[1], "label_provenance": {"state_hash": "b"}}
    out = gradient_statistics(model, optimizer, state, rows, None,
                              sampled_sets={"cluster": [a, a, a, b]})
    entries = out["example_sets"]["cluster"]["per_example"]
    expected = (entries[0]["unclipped_squared_norm"] + entries[-1]["unclipped_squared_norm"]) / 2
    assert out["clusters"]["cluster"]["sigma2"] == pytest.approx(expected)
    assert expected != pytest.approx(sum(e["unclipped_squared_norm"] for e in entries) / 4)


def test_rollout_cli_with_mock_server_and_tiny_model(tmp_path, data_root, monkeypatch):
    import tools.grad_stats as stats
    from tools.r2_allocate import read_sigma
    m, f, output, pool, records, _, checkpoint = sampled_rollout(tmp_path, data_root)
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.linspace(-.2, .2, 1200))
        def forward(self, *, input_ids, logits_to_keep, **kwargs):
            return SimpleNamespace(logits=self.theta[None, None, :].expand(1, len(logits_to_keep), -1))
    model = TinyModel()
    optimizer = adamw(model.parameters(), 1e-5)
    state = capture_state(model, optimizer)
    class TinyExperiment:
        def __init__(self, args):
            assert args.device == "cpu" and args.surrogate is None
            self.model, self.optimizer, self.state = model, optimizer, state
            self.reference, self.surrogate, self.datasets = [], None, {}
            self.checkpoint, self.manifest, self.freeze = checkpoint, m, f
            self.recipe, self.tokenizer = {"max_length": 32768}, TinyTokenizer()
        def load_messages(self, messages):
            assert not messages
        def write_provenance(self, *args, **kwargs):
            assert kwargs["extra_inputs"] == [str(output / "rollouts.jsonl"), str(pool)]
            return {"unit": True}
    monkeypatch.setattr(stats, "Experiment", TinyExperiment)
    destination = tmp_path / "sigma.json"
    stats.main(["--snapshot", "unused", "--checkpoint", str(checkpoint), "--verified", "unused",
        "--rollouts", str(output / "rollouts.jsonl"), "--pool", str(pool), "--device", "cpu",
        "--output", str(destination)])
    result = json.loads(destination.read_text())
    cid = next(iter(result["clusters"]))
    encoded = rollout_examples([records[0][0]], TinyTokenizer())[cid]
    p = model.theta.detach().softmax(0)
    expected = []
    for row in encoded:
        ids = torch.tensor([t for t in row["labels"] if t != -100])
        score = len(ids) * p - torch.bincount(ids, minlength=len(p))
        expected.append(float(score.double().square().sum()))
    assert result["clusters"][cid]["sigma2"] == pytest.approx(sum(expected) / 4, rel=2e-6)
    assert result["clusters"][cid]["sigma2"] > result["clusters"][cid]["sigma2_clipped_DIAGNOSTIC"]
    assert read_sigma(result, [cid])[cid] == result["clusters"][cid]["sigma2"]
    assert result["sampled_examples_sha256"] and result["teacher_calls"] == 0
    assert_state_equal(capture_state(model, optimizer), state)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"different checkpoint")
    args = SimpleNamespace(rollouts=[str(output / "rollouts.jsonl")], pool=str(pool),
                           states_per_cluster=10, seed=0, device="cpu", surrogate=None)
    with pytest.raises(ValueError, match="restored checkpoint"):
        load_rollout_examples(TinyExperiment(args), args)
