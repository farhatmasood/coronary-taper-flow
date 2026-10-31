"""Small shims so the package runs unchanged on NumPy 1.x and 2.x."""

from __future__ import annotations

import numpy as np

# NumPy 2.0 renamed ``trapz`` to ``trapezoid`` and removed the old spelling.
if hasattr(np, "trapezoid"):
    trapezoid = np.trapezoid
else:                                   # NumPy < 2.0
    trapezoid = np.trapz

__all__ = ["trapezoid"]
