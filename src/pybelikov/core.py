import numpy as np
from numba import jit

# -----------------------------------------------------------------------------
# 1. JIT Compiled Implementation (Fastest ~38x speedup)
# -----------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def _compute_N_bar_jit(N):
    """Internal JIT compiled N_bar calculation."""
    N_bar = np.zeros((N + 1, N + 1))

    if N >= 0:
        N_bar[0, 0] = 1.0
    if N >= 1:
        N_bar[1, 0] = 1.0
        N_bar[1, 1] = 1.0

    for n in range(2, N + 1):
        for k in range(n):
            N_bar[n, k] = np.sqrt(1.0 - (k**2 / n**2)) * N_bar[n - 1, k]
        N_bar[n, n] = np.sqrt(1.0 - 0.5 / n) * N_bar[n - 1, n - 1]

    return N_bar


@jit(nopython=True, cache=True)
def _compute_belikov_jit(N, theta, N_bar):
    """Internal JIT compiled Belikov recurrence."""
    cx = np.cos(theta)
    sx = np.sin(theta)

    P_tilde = np.zeros((N + 2, N + 2))
    P_tilde[0, 0] = 1.0

    for n in range(1, N + 1):
        P_tilde[n, 0] = cx * P_tilde[n - 1, 0] - (sx / 2.0) * P_tilde[n - 1, 1]
        for k in range(1, n + 1):
            term1 = cx * P_tilde[n - 1, k]
            term2 = (P_tilde[n - 1, k + 1] / 4.0) - P_tilde[n - 1, k - 1]
            P_tilde[n, k] = term1 - sx * term2

    P = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        factor = np.sqrt(2 * n + 1)
        for k in range(n + 1):
            P[n, k] = factor * N_bar[n, k] * P_tilde[n, k]
    return P

# -----------------------------------------------------------------------------
# 2. Vectorized npv Implementation (~30x speedup)
# -----------------------------------------------------------------------------


def _compute_N_bar_npv(N):
    """Internal Vectorized N_bar calculation."""
    N_bar = np.zeros((N + 1, N + 1))

    if N >= 0:
        N_bar[0, 0] = 1.0
    if N >= 1:
        N_bar[1, 0] = 1.0
        N_bar[1, 1] = 1.0

    for n in range(2, N + 1):
        k = np.arange(n)
        factors = np.sqrt(1.0 - (k**2 / n**2))
        N_bar[n, :n] = factors * N_bar[n - 1, :n]
        N_bar[n, n] = np.sqrt(1.0 - 0.5 / n) * N_bar[n - 1, n - 1]

    return N_bar


def _compute_belikov_npv(N, theta, N_bar):
    """Internal Vectorized Belikov recurrence."""
    cx = np.cos(theta)
    sx = np.sin(theta)

    P_tilde = np.zeros((N + 2, N + 2))
    P_tilde[0, 0] = 1.0

    for n in range(1, N + 1):
        P_tilde[n, 0] = cx * P_tilde[n - 1, 0] - (sx / 2.0) * P_tilde[n - 1, 1]

        # Vectorized inner row calculation
        prev = P_tilde[n - 1]
        term1 = cx * prev[1:n+1]
        term2 = (prev[2:n+2] / 4.0) - prev[0:n]
        P_tilde[n, 1:n+1] = term1 - sx * term2

    idx = np.arange(N + 1)
    conversion_factors = np.sqrt(2 * idx + 1).reshape(-1, 1)

    # Broadcasting
    P = conversion_factors * N_bar * P_tilde[:N+1, :N+1]
    return P

# -----------------------------------------------------------------------------
# 3. Main Public Interface
# -----------------------------------------------------------------------------


def compute_fnALF(N, theta, method='jit'):
    """
    Compute Fully Normalized Associated Legendre Functions up to degree N.

    This function implements the Belikov recurrence algorithm as evaluated and
    formulated by Lei & Li (2016). It provides options for JIT-compiled
    execution (fastest) or vectorized NumPy execution.

    Parameters
    ----------
    N : int
        Maximum degree.
    theta : float
        Colatitude in radians.
    method : str, optional
        'jit' (default) for Numba JIT compilation (fastest),
        'npv' for vectorized numpy (fast).

    Returns
    -------
    np.ndarray
        (N+1, N+1) matrix containing P_nm values.

    Raises
    ------
    ValueError
        If an invalid method is selected.

    References
    ----------
    .. [1] Lei, W., & Li, K. (2016). Evaluating Applicability of Four Recursive
       Algorithms for Computation of the Fully Normalized Associated Legendre
       Functions. Journal of Applied Geodesy, 10(4).
       https://doi.org/10.1515/jag-2016-0032
    """
    if method == 'jit':
        N_bar = _compute_N_bar_jit(N)
        return _compute_belikov_jit(N, theta, N_bar)
    elif method == 'npv':
        N_bar = _compute_N_bar_npv(N)
        return _compute_belikov_npv(N, theta, N_bar)
    else:
        raise ValueError("Method must be 'jit' or 'npv'")
