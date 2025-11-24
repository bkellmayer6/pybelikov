import numpy as np
import pytest
from pybelikov import compute_fnALF


@pytest.mark.parametrize("method", ["jit", "npv"])
def test_sum_of_squares_conservation(method):
    """
    Verifies the computed fnALFs satisfy the theoretical sum of squares property.
    Sum(P_nm^2) over all n,m should equal (N+1)^2 approximately.
    """
    N = 2190
    theta = np.radians(45.0)

    # Run computation
    P_matrix = compute_fnALF(n=N, theta=theta, method=method)

    # Calculate Sum of Squares
    computed_sum = np.sum(np.square(P_matrix))
    theoretical_sum = (N + 1)**2

    # Check error is within acceptable bounds
    error = abs(computed_sum - theoretical_sum) / theoretical_sum

    # We expect high precision, so tolerance is very low (e.g. 1e-12)
    assert error < 1e-12, f"Method {method} failed sum-of-squares check. Error: {error}"


def test_jit_vs_numpy_consistency():
    """Ensure both methods return the exact same matrix."""
    N = 2190
    theta = 0.5

    res_jit = compute_fnALF(N, theta, method='jit')
    res_np = compute_fnALF(N, theta, method='npv')

    np.testing.assert_allclose(
        res_jit, res_np, atol=1e-15, err_msg="JIT and NumPy results diverge")
