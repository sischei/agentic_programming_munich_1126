"""Three machine-checkable acceptance tests for the PINN loop (Part VII of the lecture).

The target is a one-dimensional boundary-value problem with a known solution:
u'' + u = 0 on [0, pi/2], u(0) = 0, u(pi/2) = 1, exact solution sin(x), with the
trial solution u(x) = 2x/pi + x (pi/2 - x) N(x) so that both boundary conditions hold
by construction and the loss is the residual alone.

`solve(seed=42)` must return a dict with keys
    "max_err"  : max |u - sin x| on 1,001 equispaced points of [0, pi/2]
    "u"        : callable, x (numpy array) -> u(x) (numpy array)
"""
import math
import numpy as np
from pinn_hard_bc import solve

# Produced by reference/soft_bc_baseline.py (same recipe, boundary conditions as a penalty,
# seed 42): 5.536e-4. The hard-BC trial solution must beat it; the reference solution
# kept outside this folder reaches 6.70e-5.
STORED_SOFT_BC_ERROR = 5.54e-4


def test_beats_the_stored_soft_bc_error():
    out = solve(seed=42)
    assert out["max_err"] < STORED_SOFT_BC_ERROR


def test_boundary_conditions_hold_by_construction():
    out = solve(seed=42)
    u = out["u"]
    assert abs(float(u(np.array([0.0]))[0]) - 0.0) < 1e-12
    assert abs(float(u(np.array([math.pi / 2]))[0]) - 1.0) < 1e-12


def test_run_is_reproducible():
    a = solve(seed=42)["max_err"]
    b = solve(seed=42)["max_err"]
    assert a == b
