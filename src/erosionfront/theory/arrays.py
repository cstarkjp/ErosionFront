"""
XXXX
---------------------------------------------------------------------

"""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sympy import Rational  #type: ignore

from erosionfront.theory.numerical import RichterSlopeModel

warnings.filterwarnings("ignore")

__all__ = ["Arrays"]

@dataclass
class Arrays:
    """
    XXXX

    Parameters
    ----------
    XXX

    Attributes
    ----------
    TBD

    Returns
    -------
    XXX
    """
    n_pts: int
    β: NDArray
    px: NDArray
    pz: NDArray
    vx: NDArray
    vz: NDArray
    p: NDArray
    v: NDArray
    ξx: NDArray
    ξz: NDArray
    ξrt: NDArray
    ξup: NDArray
    def __init__(
            self, 
            rsm: RichterSlopeModel,
            beta_max_frac: Rational | float,
            n_pts: int=1001,
        ) -> None:
        """
        XXXX

        Parameters
        ----------
        XXX

        Attributes
        ----------
        TBD

        Returns
        -------
        XXX
        """
        beta = np.linspace(0.0, (np.pi/2)*float(beta_max_frac), n_pts)
        if (
            rsm.px_lambda is None 
            or rsm.pz_lambda is None
            or rsm.vx_lambda is None
            or rsm.vz_lambda is None
            or rsm.p_lambda is None
            ): 
            return
        px = rsm.px_lambda(beta) 
        pz = rsm.pz_lambda(beta)
        vx = rsm.vx_lambda(beta)
        vz = rsm.vz_lambda(beta)
        p = rsm.p_lambda(beta)
        v = np.sqrt(vx**2+vz**2)

        self.n_pts = n_pts
        self.β = beta[np.isfinite(vx) & np.isfinite(p)]

        self.px = px[np.isfinite(vx) & np.isfinite(p)]
        self.pz = pz[np.isfinite(vx) & np.isfinite(p)]
        self.p = p[np.isfinite(vx) & np.isfinite(p)]

        self.v = v[np.isfinite(vx) & np.isfinite(p)]
        self.vz = vz[np.isfinite(vx) & np.isfinite(p)]
        self.vx = vx[np.isfinite(vx) & np.isfinite(p)]

        self.ξx = self.px/self.p**2
        self.ξz = self.pz/self.p**2
        self.ξrt = 1/self.px
        self.ξup = 1/self.pz