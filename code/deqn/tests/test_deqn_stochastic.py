"""Five machine-checkable acceptance tests for the stochastic DEQN (Part VII of the lecture).

The solver under test is deqn/stochastic.py: solve(seed, steps) -> {"s": callable
(log_z, log_K) -> savings rate, "net", "wall"}. Accuracy is measured on the path the
network's own policy generates and against the reference solver's policy; the
training loss is never consulted. Thresholds come from reference/baseline.md: each is
4-25x the number the reference solution reaches, so a correct solver passes on any
laptop and the silent failure of Part VIII does not.
"""
import numpy as np
import pytest

from conftest import LOG_K_SS, REFERENCE_CSV, euler_residual, simulate_own_path
from deqn.stochastic import solve

MEAN_ABS_RESIDUAL = 1e-3      # reference solution: 1.4e-4
MAX_POLICY_GAP = 1e-2         # reference solution: 2.3e-3 (max |s - s_egm| on 2,000 EGM states)
MEAN_SAVINGS_TOL = 5e-3       # |E[s] - mean s_egm of the reference file|; reference solution: 2e-4
WINDOW = 0.5                  # training states: log K in [log K* - 0.5, log K* + 0.5]


@pytest.fixture(scope="module")
def trained():
    out = solve(seed=42, steps=2000)
    yield out
    # test_run_is_reproducible calls solve() with 200 steps, and every call saves
    # results/deqn_stochastic.pt; put the 2000-step policy back so that the file the
    # /deqn-accuracy skill audits after pytest is the trained one.
    from deqn import common
    common.save_policy(out["net"], "results/deqn_stochastic.pt",
                       dict(model="stochastic", params=common.PARAMS["stochastic"], seed=42, steps=2000,
                            wall=round(out["wall"], 2), window=common.WINDOW_HALF_WIDTH, note="re-saved by the test fixture"))


@pytest.fixture(scope="module")
def own_path(trained):
    return simulate_own_path(trained["s"])


def test_euler_residual_on_own_path(trained, own_path):
    LZ, LK = own_path
    assert np.abs(euler_residual(trained["s"], LZ, LK)).mean() < MEAN_ABS_RESIDUAL


def test_policy_matches_reference(trained):
    ref = np.loadtxt(REFERENCE_CSV, delimiter=",", skiprows=1)
    s_net = trained["s"](ref[:, 0], ref[:, 1])
    assert np.abs(s_net - ref[:, 2]).max() < MAX_POLICY_GAP


def test_mean_savings_rate(trained, own_path):
    LZ, LK = own_path
    ref = np.loadtxt(REFERENCE_CSV, delimiter=",", skiprows=1)
    assert abs(trained["s"](LZ, LK).mean() - ref[:, 2].mean()) < MEAN_SAVINGS_TOL


def test_path_stays_in_training_window(own_path):
    _, LK = own_path
    assert LK.min() > LOG_K_SS - WINDOW and LK.max() < LOG_K_SS + WINDOW


def test_run_is_reproducible():
    x = np.zeros(50)
    a = solve(seed=42, steps=200)["s"](x, np.full(50, LOG_K_SS))
    b = solve(seed=42, steps=200)["s"](x, np.full(50, LOG_K_SS))
    assert np.array_equal(a, b)
