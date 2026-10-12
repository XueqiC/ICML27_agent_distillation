import json
import pytest
torch = pytest.importorskip("torch")

from test_crn_tail import warm_toy, assert_state_equal
from tools.crn_tail import message_backward, surrogate_gradient, restore
from tools.warmth_dump import coordinate_warmth, warmth_report, parameter_groups
from train.optimizer_state import capture_state


def test_bias_corrected_coordinate_bound_including_cold_coordinates():
    g = torch.tensor([.3, 2., 0., -4.], dtype=torch.float64)
    m = torch.tensor([.1, -.3, 0., 1.], dtype=torch.float64)
    v = torch.tensor([.2, 0., 0., 2.], dtype=torch.float64)
    b1, b2, eps, step = .7, .8, .02, 7
    kappa, u = coordinate_warmth(g, m, v, step, beta1=b1, beta2=b2, eps=eps)
    n = (b1 * m + (1 - b1) * g) / (1 - b1 ** (step + 1))
    a = (b2 * v / (1 - b2 ** (step + 1))).sqrt()
    d = ((b2 * v + (1 - b2) * g.square()) / (1 - b2 ** (step + 1))).sqrt()
    error = (n / (a + eps) - n / (d + eps)).abs()
    bound = u * torch.clamp(kappa / 2, max=1)
    assert bool((error <= bound + 1e-15).all())
    assert torch.isinf(kappa[1:3]).all() and u[2] == 0
    torch.testing.assert_close(u, n.abs() / (a + eps))


def test_weighted_report_exact_scalar_sum_and_opt_in_arrays(warm_toy, tmp_path):
    model, optimizer, state, rows = warm_toy
    result = warmth_report(model, optimizer, state, rows, rows, {"episodes": [0]})
    assert not list(tmp_path.iterdir())
    dF = surrogate_gradient(model, rows)["theta"]
    _, stats = message_backward(model, {"episodes": [0]}, rows)
    g = model.theta.grad.detach().clone()
    entry = state["optimizer"]["state"][0]
    kappa, u = coordinate_warmth(g, entry["exp_avg"], entry["exp_avg_sq"], float(entry["step"]))
    term = 1e-5 * dF.abs() * u * torch.clamp(kappa / 2, max=1)
    assert result["E_fd"] == pytest.approx(float(term.sum()), rel=1e-12)
    assert result["max_term"] == pytest.approx(float(term.max()), rel=1e-12)
    assert result["top20"][0]["flat_index"] == int(term.argmax())
    assert result["breakdown"]["block:other"]["E_fd"] == result["E_fd"]
    assert result["preregistered_warm_criterion"] == (result["E_fd"] <= .1 * result["first_order_L_abs"])
    assert_state_equal(capture_state(model, optimizer), state)
    path = tmp_path / "coordinates.npz"
    again = warmth_report(model, optimizer, state, rows, rows, {"episodes": [0]}, arrays_path=path)
    import numpy as np
    with np.load(path) as arrays:
        assert set(arrays.files) == {"theta::kappa", "theta::u", "theta::L", "theta::delta_theta"}
        assert all(arrays[name].dtype == np.float16 for name in arrays.files)
    assert again["E_fd"] == result["E_fd"]
    json.dumps(again, allow_nan=False)


def test_lora_block_and_layer_mapping():
    assert parameter_groups("base.model.language_model.layers.12.mlp.up_proj.lora_A.default.weight") == ("A", "12")
    assert parameter_groups("base.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight") == ("B", "0")
