"""Step 4 of the lecture: the stochastic Brock-Mirman DEQN that the /goal builds.

The tests in tests/test_deqn_stochastic.py are the specification. solve(seed, steps)
must return a dict with
    "s"     : callable (log_z, log_K) -> savings rate, numpy arrays in and out
    "net"   : the trained torch network
    "wall"  : training time in seconds
and save the policy with deqn.common.save_policy to results/deqn_stochastic.pt.
"""


def solve(seed: int = 42, steps: int = 2000) -> dict:
    raise NotImplementedError("the /goal of Part VII fills this in")
