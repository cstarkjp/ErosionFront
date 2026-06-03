"""
Geometric mechanics analysis of geomorphic Hamiltonian.
"""
import warnings
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable
from functools import partial

import numpy as np
from numpy.typing import NDArray
import scipy.interpolate
import scipy.optimize

from erosionfront.geomorphic.model import (
    Isotropic,
    Step,
    ExponentialActivation,
)

warnings.filterwarnings("ignore")

__all__ = ["GeometricMechanicsAnalysis"]

@dataclass
class GeometricMechanicsAnalysis:
    """
    Analysis of geomorphic Hamiltonian using geometric mechanics.

    Parameters
    ----------
    model: Isotropic | Step| ExponentialActivation
        Surface-normal erosion function model.
    
    Attributes
    ----------
    β_mpx: float | None = None
        Surface slope angle at maximum px.

    α_mpx: float | None = None
        Ray angle at maximum px.

    β_c0: float | None = None
        Surface slope at first critical ray angle.

    β_c1: float | None = None
        Surface slope at second critical ray angle.

    α_c0: float | None = None
        First critical ray angle.

    α_c1: float | None = None
        Second critical ray angle.

    β_rs0: float | None = None
        Lower ramp-slope angle.

    β_rs1: float | None = None
        Upper ramp-slope angle.

    α_rs0: float | None = None
        Angle of ray for lower ramp.

    α_rs1: float | None = None
        Angle of ray for upper ramp.

    """
    β_mpx: float | None = None
    α_mpx: float | None = None
    β_c0: float | None = None
    β_c1: float | None = None
    α_c0: float | None = None
    α_c1: float | None = None
    β_rs0: float | None = None
    β_rs1: float | None = None
    α_rs0: float | None = None
    α_rs1: float | None = None

    def __init__(
            self, 
            model: Isotropic | Step| ExponentialActivation,
        ) -> None:
        """
        Instantiate `GeometricMechanicsAnalysis` class.

        Performs a suite of analyses, generating a lot of new
        attributes containing the output.
        """
        self.model = model
        self.build_H_interpolation()
        self.compute_dHdp()
        self.compute_H_det_Hessian()
        self.build_gstar_interpolation()
        self.construct_detgstar_curve()
        self.build_dHdp_interpolation()
        self.construct_α_curve()
        self.find_detgstar_zeros()

    def px_onshell(self, β: float | NDArray,) -> float | NDArray:
        """
        Use speed function to compute slowness component px from β.

        The suffix 'on-shell' emphasizes that this computation only works
        for px/pz/β values where H(px,pz)=1/2.

        Parameters
        ----------
        β: float | NDArray
            Surface slope angle(s).

        Attributes
        ----------
        Uses ξ_fn_β function attribute.

        Returns
        -------
        float | NDArray:
            Value(s) of slowness component px.
        """
        return np.sin(β)/self.model.ξ_fn_β(β)
    
    def pz_onshell(self, β: float | NDArray,) -> float | NDArray:
        """
        Use speed function to compute slowness component pz from β.

        The suffix 'on-shell' emphasizes that this computation only works
        for px/pz/β values where H(px,pz)=1/2.

        Parameters
        ----------
        β: float | NDArray 
            Surface slope angle(s).

        Attributes
        ----------
        Uses ξ_fn_β function attribute.

        Returns
        -------
        float | NDArray:
            Value(s) of slowness component pz.
        """
        return -np.cos(β)/self.model.ξ_fn_β(β)
    
    def p_dot_v(self) -> NDArray:
        """
        Compute inner product p(v).

        Attributes
        ----------
        dHdpx_onshell: NDArray
            px component of H gradient = vx

        dHdpz_onshell: NDArray
            pz component of H gradient = vz

        px: NDArray
            x components of p covector array

        pz: NDArray
            z components of p covector array

        Returns
        -------
        NDArray:
            Inner product p(v), which should all be ones.
        """
        return self.dHdpx_onshell*self.px + self.dHdpz_onshell*self.pz

    def Hamiltonian(
            self,
            px: float | NDArray, 
            pz: float | NDArray
        ) -> float | NDArray:
        """
        Compute Hamiltonian values H(px,pz) anywhere (not just for H=1/2, obvs).

        Parameters
        ----------
        px: float | NDArray
            x component(s) of p covector

        pz: float | NDArray
            z component(s) of p covector

        Attributes
        ----------
        Uses ξ_fn_β function attribute.

        Returns
        -------
        float | NDArray:
            Value(s) of H 'off-shell' or on.
        """
        return 0.5*(self.model.ξ_fn_β(np.arctan2(px,-pz))**2)*(px**2+pz**2)

    def build_H_interpolation(self) -> None:
        """
        Generate a 2D spline interpolant for off-shell Hamiltonian grid.

        Attributes
        ----------
        Uses px_onshell, pz_onshell, Hamiltonian.
        Modifies many.
        """
        n_H_grid_points: int = 3001
        self.β: NDArray = np.linspace(0, np.pi, n_H_grid_points)
        self.pz: NDArray = np.array(self.pz_onshell(self.β))
        self.px: NDArray = np.array(self.px_onshell(self.β))
        self.px_span: tuple[float,float] = (
            np.min(self.px), 
            np.max(self.px)*1.05,
        )
        self.pz_span: tuple[float,float] = (
            np.min(self.pz)*1.03, 
            np.max(self.pz)*1.05,
        )
        self.n_grid_px_pts: int = n_H_grid_points
        self.n_grid_pz_pts: int = n_H_grid_points
        self.pxz_points: tuple[NDArray, NDArray] = (
            np.linspace(*self.px_span, self.n_grid_px_pts), 
            np.linspace(*self.pz_span, self.n_grid_pz_pts),
        )
        self.H_grid: NDArray = np.array(
            self.Hamiltonian(*np.meshgrid(*self.pxz_points, indexing="ij"))
        )
        self.H_interpfn: Callable \
            = scipy.interpolate.RegularGridInterpolator(
                self.pxz_points, self.H_grid
            )
        self.n_interp_grid_px_pts: int = self.n_grid_px_pts
        self.n_interp_grid_pz_pts: int = self.n_grid_pz_pts
        px_pts: NDArray = np.linspace(*self.px_span, self.n_interp_grid_px_pts)
        pz_pts: NDArray = np.linspace(*self.pz_span, self.n_interp_grid_pz_pts)
        self.Δpx: float = px_pts[1]-px_pts[0]
        self.Δpz: float = pz_pts[1]-pz_pts[0]
        px_mg: NDArray
        pz_mg: NDArray
        (px_mg, pz_mg,) = np.meshgrid(px_pts, pz_pts, indexing="ij",)
        self.px_pz_interp_grid_pts: NDArray \
            = np.array([px_mg.ravel(), pz_mg.ravel()]).T
        self.H_interp_grid: NDArray = self.H_interpfn(
            self.px_pz_interp_grid_pts, 
            method="linear").reshape(
                self.n_interp_grid_px_pts, 
                self.n_interp_grid_pz_pts,
            )
        
    def compute_dHdp(self) -> None:
        """
        Compute gradient of H.

        Attributes
        ----------
        Uses H_grid, Δpx, Δpz.
        Modifies dHdpx, dHdpz.
        """
        Δxz: tuple[float,float] = (self.Δpx, self.Δpz,)
        self.dHdpx: NDArray
        self.dHdpz: NDArray
        (self.dHdpx, self.dHdpz,) = np.gradient(self.H_grid, *Δxz,)

    def compute_H_det_Hessian(self) -> None:
        """
        Compute determinant of Hessian of H.

        Attributes
        ----------
        Uses dHdpx, dHdpx.
        Modifies det_gstar_grid.
        """
        Δxz: tuple[float,float] = (self.Δpx, self.Δpz,)
        d2Hdpx2: NDArray
        d2Hdpzdpx: NDArray
        d2Hdpxdpz: NDArray
        d2Hdpz2: NDArray
        (d2Hdpx2, d2Hdpzdpx) = np.gradient(self.dHdpx, *Δxz,)
        (d2Hdpxdpz, d2Hdpz2) = np.gradient(self.dHdpz, *Δxz,)
        self.det_gstar_grid: NDArray = d2Hdpx2*d2Hdpz2 - d2Hdpxdpz*d2Hdpzdpx

    def build_gstar_interpolation(self) -> None:
        """
        Generate an interpolating function for the det Hessian of H.

        Attributes
        ----------
        Uses pxz_points, px_pz_interp_grid_pts, n_grid_px_pts, n_grid_pz_pts.
        Modifies det_gstar_interpfn, det_gstar_interp_grid.
        """
        self.det_gstar_interpfn: Callable \
            = scipy.interpolate.RegularGridInterpolator(
                self.pxz_points, 
                self.det_gstar_grid
            )
        self.det_gstar_interp_grid: NDArray \
            = self.det_gstar_interpfn(
                self.px_pz_interp_grid_pts, 
                method="linear"
            ).reshape(self.n_grid_px_pts, self.n_grid_pz_pts,)
                
    def construct_detgstar_curve(self) -> None:
        """
        Generate a spline-interpolation fn for |g*(β)| for on-shell px, pz.

        Attributes
        ----------
        Uses px, pz, β, det_gstar_interpfn, modifies det_gstar_onshell, 
        det_gstar_onshell_interpfn.
        """
        px_pz_onshell_pts: NDArray = np.array([self.px, self.pz]).T
        self.det_gstar_onshell: NDArray \
            = self.det_gstar_interpfn(px_pz_onshell_pts)
        self.det_gstar_onshell_interpfn: Callable \
            = scipy.interpolate.CubicSpline(
                self.β, 
                self.det_gstar_onshell,
                bc_type="clamped",
            )

    def build_dHdp_interpolation(self) -> None:
        """
        Build interpolating functions and grids for dHdpx, dHdpz.

        Attributes
        ----------
        Uses pxz_points, pxz_points, dHdpx, dHdpz, px_pz_interp_grid_pts,
        n_grid_px_pts, n_grid_pz_pts.

        Modifies dHdpx_interpfn, dHdpz_interpfn, dHdpx_interp_grid,
        dHdpx_interp_grid.
        """
        self.dHdpx_interpfn: Callable \
            = scipy.interpolate.RegularGridInterpolator(
                self.pxz_points, 
                self.dHdpx,
            )
        self.dHdpz_interpfn: Callable \
            = scipy.interpolate.RegularGridInterpolator(
                self.pxz_points, 
                self.dHdpz,
            )
        self.dHdpx_interp_grid: NDArray \
            = self.dHdpx_interpfn(
                self.px_pz_interp_grid_pts, 
                method="linear",
            ).reshape(self.n_grid_px_pts, self.n_grid_pz_pts,)
        self.dHdpz_interp_grid: NDArray \
            = self.dHdpz_interpfn(
                self.px_pz_interp_grid_pts, 
                method="linear",
            ).reshape(self.n_grid_px_pts, self.n_grid_pz_pts,)
                
    def construct_α_curve(self) -> None:
        """
        Generate several grids and spline interpolation fns.
        
        Make grids for onshell dHdpx, dHdpz.
        Build spline interpolation fns for on-shell dHdpx, dHdpz. ray angle α.

        Attributes
        ----------
        Uses px, pz, β, dHdpx_interpfn, dHdpz_interpfn.
        
        Modifies dHdpx_onshell, dHdpz_onshell, dHdpx_onshell_interpfn,
        dHdpz_onshell_interpfn, α_onshell_interpfn.
        """
        px_pz_onshell_pts: NDArray = np.array([self.px, self.pz]).T
        self.dHdpx_onshell: NDArray = self.dHdpx_interpfn(px_pz_onshell_pts)
        self.dHdpz_onshell: NDArray = self.dHdpz_interpfn(px_pz_onshell_pts)
        spline: Callable = partial(
            scipy.interpolate.CubicSpline, x=self.β, bc_type="clamped",
        )
        self.dHdpx_onshell_interpfn: Callable = spline(y=self.dHdpx_onshell)
        self.dHdpz_onshell_interpfn: Callable = spline(y=self.dHdpz_onshell)
        self.α_onshell_interpfn: Callable = spline(
            y=np.arctan2(self.dHdpz_onshell, self.dHdpx_onshell)
        )

    def find_detgstar_zeros(self) -> None:
        """
        Find critical angles.

        Attributes
        ----------
        Uses several, modifies several.
        """
        # Find approx β where px is a maximum using brute force sampling
        beta_samples: NDArray = np.linspace(1e-6, np.pi, 1000)
        px_samples: NDArray = np.array(self.px_onshell(beta_samples))
        # i_sample_max_px: int = np.where(px_samples>=np.max(px_samples))[0]
        beta_sample_max: float = beta_samples[
            px_samples>=np.max(px_samples)
        ][0]
        approx_zero: float = 1e-4
        dgstar_max: float = np.max(self.det_gstar_onshell)
        neg: NDArray = np.s_[
            self.det_gstar_onshell/dgstar_max < (-approx_zero)
        ]
        critical_β_guesses: tuple[float,float] \
            = (
                (float(self.β[neg][0]), float(self.β[neg][-1]),)
                if self.β[neg].shape[0]>0
                else (np.pi, np.pi,)
            )
        print(f"critical_β_guesses: {critical_β_guesses}")
        self.β_mpx = float(beta_sample_max)
        self.α_mpx = float(self.α_onshell_interpfn(self.β_mpx))
        print(f"β_mpx = {np.rad2deg(self.β_mpx):3.2f}")
        print(f"α_mpx = {np.rad2deg(self.α_mpx):3.2f}")

        # Make a wrapper for root finder
        find_detgstar_root: Callable = partial(
            scipy.optimize.root_scalar, 
            self.det_gstar_onshell_interpfn,
            method="newton", 
            bracket=(np.pi/3000,np.pi/2,),
        )
        detgstar_roots: tuple[Any,Any] = (
            find_detgstar_root(
                x0=(
                    critical_β_guesses[0] if np.abs(critical_β_guesses[0])>1e-3 
                    else beta_sample_max*1.4
                    # else np.pi/5
                ),
                x1=beta_sample_max*2,
            ),
            find_detgstar_root(
                x0=critical_β_guesses[1]
            ),
        )

        # Find β_c0, β_c1
        if not detgstar_roots[0].converged:
            print("Failed to find β_c0")
            self.β_c0 = 0
        if not detgstar_roots[1].converged:
            print("Failed to find β_c1")
            self.β_c1 = 0
        if detgstar_roots[0].converged or detgstar_roots[1].converged:
            det_gstar_zeros: list[float] = [
                soln_.root if soln_.root>0 else np.pi+soln_.root
                for soln_ in detgstar_roots
            ]
            self.β_c0 = float(det_gstar_zeros[0])
            self.β_c1 = float(det_gstar_zeros[1])

        # self.β_crit_guesses = critical_β_guesses 
        self.α_c0 = float(self.α_onshell_interpfn(self.β_c0))
        self.α_c1 = float(self.α_onshell_interpfn(self.β_c1))

        α_offset_fn_ = lambda β: self.α_onshell_interpfn(β) - self.α_c1
        soln_ = scipy.optimize.root_scalar(
            α_offset_fn_, 
            # x0 = self.β_c0*0.1,
            method="bisect", 
            bracket=(0, np.pi/2.01,) #self.β_c0),
        )
        if not soln_.converged:
            raise ValueError(r"Failed to find β_rs1")
        self.β_rs1 = float(soln_.root)
        self.α_rs1 = float(self.α_onshell_interpfn(self.β_rs1))

        α_offset_fn_ = lambda β: self.α_onshell_interpfn(β) - self.α_c0
        soln_ = scipy.optimize.root_scalar(
            α_offset_fn_, 
            x0 = np.pi/4 if self.β_c1 is None else self.β_c1*1.1,
            # method="bisect", 
            # bracket=(self.β_c1, np.pi/2),
        )
        if not soln_.converged:
            raise ValueError(r"Failed to find β_rs0")
        self.β_rs0 = float(soln_.root)
        self.α_rs0 = float(self.α_onshell_interpfn(self.β_rs0))