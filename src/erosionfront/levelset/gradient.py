"""
Finite differencing of signed-distance φ grid.
"""
import warnings

from numpy.ma import MaskedArray

warnings.filterwarnings("ignore")

__all__ = ["GradientMixin"]

class GradientMixin:
    """
    Mixin providing methods for finite differencing signed-distance φ grid.
    """
    
    @staticmethod
    def fwd_back_differences(
            φ: MaskedArray,
            Δx: float,
        ) -> tuple[MaskedArray, MaskedArray, MaskedArray, MaskedArray]:
        """
        Compute forward, backward, mean & double finite differences.

        Could perhaps use np.roll to make this process more efficient?

        Parameters
        ----------
        φ: MaskedArray
            Signed distance grid.
        Δx: float
            Grid spacing in both x and z directions (assumed equal).
        
        Returns
        -------
        (Δφ_x, Δφ_z, Δ2φ_x2, Δ2φ_z2,): tuple[MaskedArray, MaskedArray, \
            MaskedArray, MaskedArray]
            1st-order (center-weighted) and 2nd-order φ finite differences.
        """
        Δz: float = Δx
        φ_xp: MaskedArray = φ.copy()
        φ_xm: MaskedArray = φ.copy()
        φ_zp: MaskedArray = φ.copy()
        φ_zm: MaskedArray = φ.copy()

        # Generate shifted grids to make differencing cleaner
        # Remember arrays are [rows, cols] and that rows=z, cols=x
        φ_xp[:,:-1] = φ[:,1:]
        φ_xm[:,1:]  = φ[:,:-1]
        φ_zp[:-1,:] = φ[1:,:]
        φ_zm[1:,:]  = φ[:-1,:]

        # Forward and backward differences in x and z
        Δφ_xp: MaskedArray = φ_xp - φ
        Δφ_xm: MaskedArray = φ - φ_xm
        Δφ_zp: MaskedArray = φ_zp - φ
        Δφ_zm: MaskedArray = φ - φ_zm

        # Mean differences ~dφ/dx∇∇𝛻𝚺∇
        Δφ_x: MaskedArray = ((Δφ_xp + Δφ_xm)/2) / Δx
        Δφ_z: MaskedArray = ((Δφ_zp + Δφ_zm)/2) / Δz

        # Double differences ~d2φ/dx2
        Δ2φ_x2: MaskedArray = ((Δφ_xp - Δφ_xm)/2) / Δx
        Δ2φ_z2: MaskedArray = ((Δφ_zp - Δφ_zm)/2) / Δz

        return (Δφ_x, Δφ_z, Δ2φ_x2, Δ2φ_z2,)