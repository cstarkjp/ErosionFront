"""
Level-set solution of HJE model propagation of a rock-slope erosion front.
"""
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from numpy.ma import MaskedArray

from erosionfront.misc.utils import Timer
from erosionfront.levelset.gradient import GradientMixin
from erosionfront.levelset.base import LevelSetSolverBase
from erosionfront.levelset.shock import ShocksMixin
from erosionfront.levelset.export import ExportMixin

warnings.filterwarnings("ignore")

__all__ = ["LevelSetSolver"]

class LevelSetSolver(
    LevelSetSolverBase, 
    GradientMixin, 
    ShocksMixin, 
    ExportMixin
):    
    """
    Solver implementing level-set solution of geomorphic Hamilton-Jacobi eqn.

    Subclasses `LevelSetSolverBase` and uses `GradientMixin`, `ShocksMixin`, 
    `ExportMixin`.

    Parameters
    ----------
    See `LevelSetSolverBase` and its antecedents.

    Attributes
    ----------
    φ: MaskedArray
        Signed-distance function φ grid, masked to narrow band across surface.
    Δφ_x: MaskedArray
        Finite-difference approx of ∂φ/∂x.
    Δφ_z: MaskedArray
        Finite-difference approx of ∂φ/∂z.
    Δ2φ_x2: MaskedArray
        Finite-difference approx of ∂2φ/∂x2.
    Δ2φ_z2: MaskedArray
        Finite-difference approx of ∂2φ/∂z2.
    β: MaskedArray
        Surface tilt angle grid.
    ξ_fn_β: Callable
        Surface-normal erosion speed model function ξ(β).
    ξ_model: MaskedArray
        Grid of model erosion speed ξ_model for narrow-band across the surface.
    ξ0: MaskedArray
        Grid of reference erosion speed ξ_0 for narrow-band across the surface.
    ξ: MaskedArray
        Grid of actual erosion speed ξ for narrow-band across the surface,
        computed as ξ = ξ_0 * ξ_model
    H_ls: MaskedArray
        Level-set numerical Hamiltonian 
        (not to be confused with the geomorphic Hamiltonian)
        used in the HJE solution of erosion-front motion.
        Again computed for a narrow-band across the surface.
    dHlsdφx: MaskedArray
        Finite-difference approximation of the derivative of the level-set
        Hamiltonian wrt ∂φ/∂x. This term is used to compute a numerical
        viscosity such that we obtain a 'viscosity solution' of the HJE.
        Again computed for a narrow-band across the surface.
    dHlsdφz: MaskedArray
        Finite-difference approximation of the derivative of the level-set
        Hamiltonian wrt ∂φ/∂z. This term is used to compute a numerical
        viscosity such that we obtain a 'viscosity solution' of the HJE.
        Again computed for a narrow-band across the surface.
    κx: float
        Numerical viscosity in the x direction.
        Again computed for a narrow-band across the surface.
    κz: float
        Numerical viscosity in the x direction.
        Again computed for a narrow-band across the surface.
    dφdt: MaskedArray
        Rate of change of signed-distance function φ wrt time.
        Obtained from the (level-set) Hamilton-Jacobi equation.
        Again computed for a narrow-band across the surface.
    φ_next: MaskedArray | None
        Signed-distance function φ grid for the next time step.
        Again computed for a narrow-band across the surface.
    φ_everywhere: NDArray
        Signed-distance function φ computed across the whole grid.
        This field is not extrapolated by rather simply extended from 
        the narrow band by filling the remaining grid cells with max/min values.
    φ_aux: NDArray | None
        An auxiliary grid used to compute which pixels are subject to 
        'slab failure'.
    """
    φ: MaskedArray
    Δφ_x: MaskedArray
    Δφ_z: MaskedArray
    Δ2φ_x2: MaskedArray
    Δ2φ_z2: MaskedArray
    β: MaskedArray
    ξ_fn_β: Callable
    ξ_model: MaskedArray
    ξ0: MaskedArray
    ξ: MaskedArray
    H_ls: MaskedArray
    dHlsdφx: MaskedArray
    dHlsdφz: MaskedArray
    κx: float
    κz: float
    dφdt: MaskedArray
    φ_next: MaskedArray | None
    φ_everywhere: NDArray
    φ_aux: NDArray | None

    def first_update(self) -> None:
        """Set up time-stepping solution with requisite initializations."""
        Timer.start("fast-march travel time on signed distance grid",)
        self.β = np.zeros_like(self.φ)
        self.ξ = np.ones_like(self.φ)
        if self.model is None:
            raise ValueError("Erosion model not specified")
        self.φ_next = None
        self.φ_aux = None
        self.T_slices = {}
        Timer.stop()

    # Replace abstract method
    def update(
            self, 
            Δt: float, 
            Δt_reinitialize: float | None=None,
            do_finalize: bool=False,
            do_timing: bool=False,
        ) -> None:
        """
        Update grids for one time step of duration Δt.

        The basic cycle needs to be:

        set t=0, set up grids

        - **loop while t<=T_limit**
            - _sometimes_ reinitialize signed distance φ(t) by fast marching
            - back & forward gradients (Δφ_xp, Δφ_xm), (Δφ_zp, Δφ_zm)
            - mean grad  Δφ_x, Δφ_z = (Δφ_xp+Δφ_xm)/2, (Δφ_zp+Δφ_zm)/2
            - dbl-diff  Δ2φ_x2, Δ2φ_z2 = (Δφ_xp-Δφ_xm)/2, (Δφ_zp-Δφ_zm)/2
            - slope angle from mean grad  β = arctan2(Δφ_z, Δφ_x)
            - front-normal speed from slope angle ξ(β)
            - level-set Hamiltonian  lsH(Δφ_x,Δφ_z) = ξ(β).|(Δφ_x, Δφ_z)|
            - Hamiltonian derivatives lsH_{∂φ/∂x}, lsH_{∂φ/∂z} wrt ∂φ/∂x, ∂φ/∂z
            - numerical viscosities  κx, κz = max(lsH_{∂φ/∂x}), max(lsH_{∂φ/∂z})
            - update rate  dφ/dt = - lsH(Δφ_x, Δφ_z) - κx.Δ2φ_x2 - κz.Δ2φ_z2
            - signed distance change  Δφ = (∂φ/∂t).Δt
            - signed distance update  φ(t+Δt) = φ(t) + Δφ
            - front T(x,z) from zero level-set  {x,z} | φ(t)=0
            - update time  t -> t+Δt
        - **repeat** 
    
        Parameters
        ----------
        Δt: float
            Time step.
        Δt_reinitialize: float | None=None
            Periodicity of reinitialization of φ signed-distance grid.
        do_finalize: bool=False
            Just finish up, don't do another loop of computation.
        do_timing: bool=False
            Measure and report compute times taken at steps during the loop.

        Attributes
        ----------
        Most of the sim grids are modified.
        """
        if self.model is None:
            raise ValueError("Model not specified")
        
        ################### propagate front by Δt #####################
        Timer(do_timing, "Evolving")

        ################### get ready #####################
        Timer.start("getting ready",)
        n_reinitialize: int | None = (
            int(np.round(Δt_reinitialize/Δt))
            if Δt>0 and Δt_reinitialize is not None
            else None
        )
        if not do_finalize:
            self.φ = self.get_φ_next()
        Timer.stop()

        ################### 1st & 2nd order gradients of φ ###############
        Timer.start("compute gradients  ∂φ/∂x, ∂φ/∂z, ∂2φ/∂x2, ∂2φ/∂z2",)
        (self.Δφ_x, self.Δφ_z, self.Δ2φ_x2, self.Δ2φ_z2,) = \
            self.fwd_back_differences(self.φ, self.Δx,)
        assert type(self.Δφ_x) is MaskedArray
        assert type(self.Δφ_z) is MaskedArray
        assert type(self.Δ2φ_x2) is MaskedArray
        assert type(self.Δ2φ_z2) is MaskedArray
        Timer.stop()
        
        ################ convert gradient φ into surface angle β ############
        Timer.start("gradient ∂φ/∂x, ∂φ/∂z -> angle β",)
        self.β = -np.arctan2(self.Δφ_x, self.Δφ_z,)  #type: ignore
        self.β[~np.isfinite(self.β)] = 0
        assert type(self.β) is MaskedArray
        Timer.stop()

        if do_finalize:
            return
        
        ################ convert angle β into front speed ξ ############
        Timer.start("angle β -> model ξ",)
        ξ_model: float | NDArray | MaskedArray = self.model.ξ_fn_β(self.β,)
        if type(ξ_model) is not np.ma.masked_array:
            raise ValueError("ξ_model is not a masked array")
        else:
           self.ξ_model = ξ_model
        # assert type(self.ξ_model) is MaskedArray
        Timer.stop()

        ################ get ξ0 grid, compute ξ, combine ############
        Timer.start("get ξ0 grid, compute ξ, combine",)
        self.ξ0 = self.get_substrate_ξ0(self.φ)
        self.ξ = self.ξ0 * self.ξ_model
        assert type(self.ξ0) is MaskedArray
        assert type(self.ξ) is MaskedArray
        Timer.stop()

        ################### level-set Hamiltonian #####################
        Timer.start("model ξ -> level-set Hamiltonian")
        # Remove the mask!
        self.H_ls = self.ξ.data.copy()
        assert type(self.H_ls) is np.ndarray
        Timer.stop()
        
        ################### ls-Hamiltonian derivatives ###################
        Timer.start("level-set Hamiltonian derivatives")
        # Need to scale by substrate speeds ξ0
        self.dHlsdφx = self.ξ0*self.model.dHlsdφ_fn_Δφx(self.Δφ_x, self.Δφ_z,)
        self.dHlsdφz = self.ξ0*self.model.dHlsdφ_fn_Δφz(self.Δφ_x, self.Δφ_z,)
        assert type(self.dHlsdφx) is MaskedArray
        assert type(self.dHlsdφz) is MaskedArray
        Timer.stop()
        
        ################### numerical viscosities ###################
        Timer.start("numerical viscosities κx, κz")
        self.κx = np.max(self.dHlsdφx)
        self.κz = np.max(self.dHlsdφz)
        Timer.stop()
        
        ################### compute dφdt, next φ ######################
        Timer.start("compute dφdt, next φ")
        self.dφdt = (
            self.H_ls + 
            (self.κx*self.Δ2φ_x2 + self.κz*self.Δ2φ_z2)*self.κ_boost
            * (
                1 +
                  np.heaviside(self.Δ2φ_x2 + self.Δ2φ_z2, 0)
                * np.heaviside(np.abs(self.β)-(np.pi/2)*0.5, 0)
                * 0
            )
        )
        self.φ_next = self.φ + self.dφdt*self.Δt
        assert type(self.dφdt) is MaskedArray
        assert type(self.φ_next) is MaskedArray
        Timer.stop()

        ################ signed distance φ everywhere #################
        Timer.start("compute signed distance φ everywhere",)
        self.φ_everywhere = self.fm_distance_everywhere(
            self.φ_next,
            self.φ_everywhere,
            self.Δx,
            self.fm_order if self.fm_order is not None else 2,
        )
        assert type(self.φ_everywhere) is np.ndarray
        Timer.stop()

        ################# φ thresholding, slab failure ###################
        if (
            self.model.do_slabfailure
            and self.i_step is not None
            and self.i_step>0 
            and self.n_slabfailure is not None 
            and (self.i_step % self.n_slabfailure==0)
        ):
            self.φ_aux = self.φ_everywhere.copy()
            self.φ_aux[self.Δφ_z>0] = 0
            # Need to pick a minimum distance from which to trigger slab failure
            #   - the 3Δx here is a heuristic minimum
            #   - depends critically on κ_boost, here assumed=1
            #   - the maximum is arbitrary
            φ_thresholds: tuple[float,float] = (self.Δx*3, 1,)
            self.φ_aux[(self.φ<φ_thresholds[0]) | (self.φ>=φ_thresholds[1])] = 0
            self.φ_aux[self.φ_aux!=0] = 1
            i_φ_break = np.argmax(self.φ_aux, axis=0)
            for j_, i_ in enumerate(i_φ_break):
                if i_>0:
                    self.φ_everywhere[i_:,j_] = φ_thresholds[0]
            
        ################# possibly reinitialize φ_next ###################
        if (
            n_reinitialize is not None 
            and self.i_step is not None
            and (self.i_step % n_reinitialize)==0
        ):
            Timer.start("reinitializing φ_next",)
            # print(f"Reinitializing: t={self.i_step}  t={self.t_total}")
            self.φ_next = self.fm_distance(
                self.φ_everywhere, 
                self.band_width if self.band_width is not None else 50,
                self.Δx,
                self.fm_order if self.fm_order is not None else 2,
            )
            Timer.stop()
        elif self.i_step is not None and self.i_step>0:
            self.φ_next = np.ma.masked_array(self.φ_next, mask=self.φ.mask,)
        assert type(self.φ_next) is MaskedArray

        ################# advance ###################
        self.t_total += Δt
        