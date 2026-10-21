"""Known synthetic delays validate log analysis; not hardware measurements."""

import numpy as np
import pytest

from tools.trajectory_lab.timing_audit import fit_delay


@pytest.mark.parametrize("lag", [0.0, 0.07, 0.24, -0.03])
def test_fit_recovers_known_delay_without_turning_stationary_axes_into_zero(lag):
    t = np.arange(0, 3, 0.01)

    def signal(t):
        return np.stack([0.1 * np.sin(t * 2.3) + 0.03 * t, np.zeros_like(t)], axis=1)

    result = fit_delay(t, signal(t), t, signal(t - lag))
    assert result["delay_ms"] == pytest.approx(lag * 1000)
    assert result["moving_joint_count"] == 1
    assert result["rmse_rad"] < 1e-10
    assert not result["search_boundary"]


def test_stationary_and_too_short_data_are_not_latency_measurements():
    t = np.arange(0, 3, 0.01)
    assert fit_delay(t, np.zeros((len(t), 2)), t, np.zeros((len(t), 2))) is None
    assert fit_delay(t[:2], np.ones((2, 2)), t[:2], np.ones((2, 2))) is None


def test_boundary_fit_is_explicit_not_hidden_as_an_exact_latency():
    t = np.arange(0, 3, 0.01)
    source = (t * 0.1)[:, None]
    result = fit_delay(t, source, t, source - 0.09)
    assert result["delay_ms"] == 600
    assert result["search_boundary"]
    assert result["rmse_rad"] > 0.02


def test_unordered_data_rejected():
    t = np.array([0, 2, 1])
    with pytest.raises(ValueError):
        fit_delay(t, np.zeros((3, 1)), t, np.zeros((3, 1)))
