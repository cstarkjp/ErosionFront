"""
Methods for computing front speed in level-set solver.
"""
import warnings
from collections.abc import Callable

import numpy as np
from numpy.ma import MaskedArray
import scipy.ndimage

from erosionfront.geomorphic.substrate import (
    OneLayerSubstrate, 
    TwoLayerSubstrate, 
    MultiLayerSubstrate,
)
from erosionfront.geomorphic.surface import (
    StraightLineSurface, 
    ErrorFunctionSurface, 
    SemiCircularArcSurface,
    QuarterCircularArcSurface
)
from erosionfront.levelset.grid import LevelSetGridMethods

warnings.filterwarnings("ignore")

__all__ = ["LevelSetSpeedMethods"]

class LevelSetSpeedMethods(LevelSetGridMethods):
    """
    Speed-related methods for use in erosion front modeling.

    Subclasses `LevelSetGridMethods`.

    Attributes
    ----------
    φ: MaskedArray
        Signed-distance grid at this time step.
    φ_next: MaskedArray | None
        Signed-distance grid at next time step.
    """
    φ: MaskedArray
    φ_next: MaskedArray | None

    def get_substrate_ξ0(
            self, 
            φ: MaskedArray,
        ) -> MaskedArray:
        """
        Generate relative η grid (effective speed ξ0).

        Parameters
        ----------
        φ: MaskedArray
            Any grid of the size being used in the sim.

        Attributes
        ----------
        substrate: OneLayerSubstrate | TwoLayerSubstrate | MultiLayerSubstrate
            Instance of class containing substrate info.
        surface.height: float
            Top height of surface above z=0 base.
        η_upperlayer: float
            Relative erodiblity of strong layer (weak layer assumed = 1).
        do_hardtop: bool
            Whether to fake a highly resistant, very thin, upper "caprock".
        do_hardbase: bool
            Whether to fake a highly resistant, very thin, lower "bedrock".
        η_hardtopbase: float
            Relative erodiblity of base/top layers (weak layer assumed = 1).

        Returns
        -------
        MaskedArray
            Modulated erodbility (ref speed) grid ξ0 across the narrow band.
        """
        substrate: OneLayerSubstrate | TwoLayerSubstrate | MultiLayerSubstrate
        if self.substrate is not None:
            substrate = self.substrate 
        else:
            raise ValueError("Substrate not specified")
        surface: (
            StraightLineSurface | 
            ErrorFunctionSurface | 
            SemiCircularArcSurface | 
            QuarterCircularArcSurface
        )
        if self.surface is not None:
            surface = self.surface 
        else:
            raise ValueError("Surface not specified")
        z_max: float = surface.height
        z_change: Callable = lambda z: int(self.xz_to_ij((0, z,))[1])
        ξ0: MaskedArray = np.ones_like(φ)
        # Prevent erosion of top surface
        if type(substrate)==TwoLayerSubstrate:
            ξ0[z_change(0.5):,:] = substrate.η_upperlayer
        if substrate.do_hardtop:
            # Make the hard top either 5% of height, or 10 pixels high,
            #    - whichever is smaller
            ξ0[z_change(z_max-min(self.Δx*substrate.nz_hardtop, 
                                  0.001*substrate.nz_hardtop)):,:] \
                = substrate.η_hardtopbase
        # Prevent erosion of bottom surface and thus motion to left
        if substrate.do_hardbase:
            #BUG: this doesn't work if the initial surface isn't z \in [0,1]
            ξ0[:(z_change(0)),:] = substrate.η_hardtopbase
        # Smear out a bit to stabilize
        ξ0_ = scipy.ndimage.uniform_filter(
            ξ0, 
            size=substrate.nz_hardtop,
        )
        ξ0 = np.ma.masked_array(ξ0_, mask=ξ0.mask,)
        return ξ0

    def get_φ_next(self) -> MaskedArray:
        """
        Return next signed-distance grid φ_next or just φ if not available.

        Returns
        -------
        MaskedArray:
            Next signed-distance φ grid or just current φ if next unavailable.
        """
        return (
            self.φ if self.φ_next is None
            else self.φ_next
        )