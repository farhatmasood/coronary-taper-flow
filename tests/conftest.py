"""Shared fixtures. The package is imported from src/ without installation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coronary import model, pulsatile_solver  # noqa: E402


@pytest.fixture(scope="session")
def m():
    """The default calibrated model (taper-resolved closure)."""
    return model.build()


@pytest.fixture(scope="session")
def puls(m):
    """Resting pulsatile solution of the default model."""
    return pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
