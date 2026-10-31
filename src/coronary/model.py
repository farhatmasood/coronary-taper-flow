"""One entry point that assembles a calibrated coronary model.

Every script, test and experiment builds the model through :func:`build`, so
there is exactly one place where geometry, outflow closure and calibration
are combined.

    >>> from coronary import model
    >>> m = model.build()
    >>> round(m.rest.total_inflow_ml_min, 1)
    250.0
"""

from __future__ import annotations

from dataclasses import dataclass

from . import constants as C
from . import coronary_bc, geometry, steady_solver
from .coronary_bc import CoronaryOutlet
from .geometry import CoronaryTree
from .steady_solver import SteadySolution


@dataclass
class Model:
    """A calibrated coronary model and its resting steady solution."""

    tree: CoronaryTree
    outlets: dict[str, CoronaryOutlet]
    rest: SteadySolution
    closure: str
    murray_exponent: float

    def hyperaemic(self, **kwargs) -> SteadySolution:
        """Lesion-free steady solve at maximal hyperaemia."""
        return steady_solver.solve_steady(self.tree, self.outlets, hyperemic=True, **kwargs)


def build(closure: str = "taper", murray_exponent: float | None = None,
          tree: CoronaryTree | None = None, reference_fit: bool = True) -> Model:
    """Load the tree, distribute the outflow, calibrate and solve at rest.

    With ``reference_fit`` (the default) the radii are first replaced by
    their monotone reference fit (:func:`geometry.monotone_reference`), as
    quantitative angiography reports a reference diameter function rather
    than raw edge radii; pass ``False`` to use the radii exactly as given.
    Parameters are resolved at call time, so a sensitivity sweep that
    perturbs a value in :mod:`coronary.constants` reaches the model.
    """
    k = C.MURRAY_EXPONENT if murray_exponent is None else murray_exponent
    tree = geometry.load_default_tree() if tree is None else tree
    if reference_fit:
        tree = geometry.monotone_reference(tree)
    outlets = coronary_bc.build_outlets(tree, k, closure)
    steady_solver.calibrate(tree, outlets)
    rest = steady_solver.solve_steady(tree, outlets, hyperemic=False)
    return Model(tree=tree, outlets=outlets, rest=rest, closure=closure,
                 murray_exponent=k)
