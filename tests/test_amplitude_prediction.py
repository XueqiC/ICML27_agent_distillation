import json
import math

import pytest
torch = pytest.importorskip("torch")

from test_crn_tail import warm_toy, assert_state_equal
from tools.amplitude_prediction import (amplitude_prediction, curve_summary, mean_field_response,
                                        predict_from_gradients)
from train.optimizer_state import capture_state


@pytest.mark.parametrize("T", [0, 10, 300])
@pytest.mark.parametrize("amplitude", [-8., 0., .5, 1., 4., 100.])
def test_exact_law_matches_direct_float64_stationary_simulation(T, amplitude):
    # Deterministic stationary tail: gbar=-.7, vbar=.49, rho=1.
    mean, variance, dF, eta, b1, b2 = -.7, .49, 1.3, .002, .9, .999
    g = amplitude * mean
    def simulate(message):
        m = torch.tensor(mean, dtype=torch.float64)
        v = torch.tensor(variance, dtype=torch.float64)
        theta = torch.zeros((), dtype=torch.float64)
        for gradient in [message] + [mean] * T:
            m = b1 * m + (1 - b1) * gradient
            v = b2 * v + (1 - b2) * gradient**2
            theta -= eta * m / v.sqrt()
        return dF * theta
    predicted = mean_field_response(*[torch.tensor(x, dtype=torch.float64) for x in (g, mean, variance, dF)],
                                    beta1=b1, beta2=b2, eta=eta, T=T)
    torch.testing.assert_close(predicted, simulate(g) - simulate(0), atol=5e-14, rtol=5e-11)


def test_bias_corrected_moments_and_global_clipping(warm_toy):
    model, optimizer, state, rows = warm_toy
    result = amplitude_prediction(model, optimizer, state, rows, rows, {"episodes": [0]}, [0, 1, 4, 8])
    assert result["prediction_complete"]
    assert result["predictions"]["unclipped"]["curve"][0]["predicted_delta_F"] == 0
    assert result["predictions"]["clipped"]["curve"][0]["predicted_delta_F"] == 0
    assert_state_equal(capture_state(model, optimizer), state)
    json.dumps(result, allow_nan=False)


def test_sign_rho_formula_and_degenerate_coordinates():
    m, v, g, dF = (torch.tensor([x], dtype=torch.float64) for x in (.3, .8, 2., -.2))
    rho, lam, alpha_eta = abs(m) / v.sqrt(), g * m.sign() / v.sqrt(), -dF * m.sign() * .1
    expected = 0
    for j in range(11):
        b, c = .1 * .9**j, .001 * .999**j
        expected += alpha_eta * ((rho + b * (lam - rho)) / (1 + c * (lam**2 - 1)).sqrt()
                                 - rho * (1 - b) / math.sqrt(1 - c))
    torch.testing.assert_close(mean_field_response(g, m, v, dF, eta=.1), expected)
    with pytest.raises(ValueError, match="positive v"):
        mean_field_response(g, m, v * 0, dF)
    # The continuous m=0 extension keeps a message's nonzero first-moment response.
    assert float(mean_field_response(g, m * 0, v, dF)) != 0


def test_peak_and_nontrivial_zero_crossing_are_labeled_grid_estimates():
    summary = curve_summary([0, .5, 1, 2, 4], [0, .5, 1, .5, -.5])
    assert summary["predicted_peak_c_grid"] == 1
    assert summary["zero_crossings"] == [{"bracket": [2, 4], "linear_interpolation": 3.0}]
    assert summary["first_negative_c_grid"] == 4


def test_corrected_plug_in_and_missing_warm_coordinates(warm_toy):
    _, _, state, _ = warm_toy
    g = {"theta": torch.tensor([3., -2., 1., -4.], dtype=torch.float64)}
    dF = {"theta": torch.tensor([.2, .1, -.1, .4], dtype=torch.float64)}
    norm = float(g["theta"].norm())
    clipped = {"theta": g["theta"] / (norm + 1e-6)}
    result = predict_from_gradients(state, g, clipped, dF, [0, 1, 4])
    entry = state["optimizer"]["state"][0]
    t = float(entry["step"])
    expected = mean_field_response(g["theta"], entry["exp_avg"] / (1 - .9**t),
                                  entry["exp_avg_sq"] / (1 - .999**t), dF["theta"])
    assert result["predictions"]["unclipped"]["curve"][1]["predicted_delta_F"] == pytest.approx(float(expected.sum()))
    ys = [row["predicted_delta_F"] for row in result["predictions"]["clipped"]["curve"]]
    assert ys[1] == pytest.approx(ys[2], rel=2e-6, abs=1e-12)
    entry["exp_avg_sq"][0] = 0
    partial = predict_from_gradients(state, g, clipped, dF, [1])
    assert partial["counts"]["sensitive_v_zero"] == 1 and not partial["prediction_complete"]
