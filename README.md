# pybelikov

A Python module for the high-precision computation of **Fully Normalized Associated Legendre Functions (fnALFs)** using the Belikov recurrence algorithm.

## Why pybelikov?

Standard recursive algorithms for Legendre functions often suffer from numerical instability (underflow/overflow) at high degrees ($n > 2000$) and can be computationally slow.

`pybelikov` implements the stable recurrence methods originally derived by Belikov, using the formulation and evaluation presented by **Lei & Li (2016)**. This implementation is optimized for modern Python environments and offers two execution modes:

1. **JIT Mode (Default):** Uses `numba` to compile the recursion into machine code.
    * **~38x faster** than standard iterative loop implementations.
    * **~25% faster** than optimized vectorized NumPy code.
2. **Vectorized Mode:** A pure NumPy implementation that utilizes array broadcasting.
    * **~30x faster** than standard iterative loop implementations.

*Benchmarks performed on N=2190 (EGM2008 standard).*

## Installation

```bash
pip install pybelikov
```

## Usage

```Python
import numpy as np
from pybelikov import compute_fnALF

# Define parameters
degree = 2190
colat_rad = np.deg2rad(45.0)  # Colatitude in radians

# Compute using the fastest method (JIT)
P_nm = compute_fnALF(N=degree, theta=colat_rad, method='jit')

# Access a specific value P(n=2, m=2)
print(f"P_2,2 = {P_nm[2, 2]}")
```

## References

The algorithm implemented in this package is based on the evaluation and formulation described in:

Lei, W., & Li, K. (2016). Evaluating Applicability of Four Recursive
Algorithms for Computation of the Fully Normalized Associated Legendre
Functions. Journal of Applied Geodesy, 10(4).
<https://doi.org/10.1515/jag-2016-0032>

## Citing pybelikov

If you use `pybelikov` in your research, please credit the software.
A `CITATION.cff` file is included in this repository for compatibility with reference managers.

**Software Citation:**
> Kellmayer, B. (2025). pybelikov: High-precision computation of fully
> normalized associated Legendre functions.
> <https://github.com/bkellmayer6/pybelikov>

**Algorithm Citation:**
> Lei, W., & Li, K. (2016). Evaluating Applicability of Four Recursive
> Algorithms for Computation of the Fully Normalized Associated Legendre
> Functions. Journal of Applied Geodesy, 10(4).
