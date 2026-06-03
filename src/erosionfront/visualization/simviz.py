"""
Visualization of level-set solutions of erosion front propagation.
"""
import warnings
from typing import Any
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from numpy.ma import MaskedArray
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib import ticker
from matplotlib.colorbar import Colorbar
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter

from erosionfront.visualization.base import GraphingBase
from erosionfront.geomorphic.surface import (
    StraightLineSurface, ErrorFunctionSurface,
    SemiCircularArcSurface, QuarterCircularArcSurface,
)
from erosionfront.geomorphic.gma import GeometricMechanicsAnalysis
from erosionfront.levelset.solver import LevelSetSolver

warnings.filterwarnings("ignore")

__all__ = ["SimulationViz"]

class SimulationViz(GraphingBase):
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

    def plot_initial_surface(
            self,
            fig_name: str,
            initial_surface: (
                StraightLineSurface 
                | ErrorFunctionSurface
                | SemiCircularArcSurface 
                | QuarterCircularArcSurface
            ),
            gma: GeometricMechanicsAnalysis | None=None,
            # model: Any | None,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(7, 5,),
        ) -> None:
        """
        Plot initial surface x-z curve.

        Parameters
        ----------
        fig_name: str
        initial_surface: ErrorFunctionSurface | StraightLineSurface
        title_font_size: int=12
        fig_size: tuple[float, float]=(7, 5,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        title: str = "Initial surface"
        plt.title(
            title,
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$x$  [-]")
        plt.ylabel(r"$z$  [-]")
        if initial_surface.x is None or initial_surface.z is None:
            raise ValueError("Surface points not given")
        plt.plot(
            initial_surface.x, 
            initial_surface.z, 
            ".-", 
            ms=1, 
            lw=1.5, 
            alpha=1,
        )
        plt.xlim(initial_surface.x[0],initial_surface.x[-1])
        plt.grid(ls=":")
        axes = plt.gca()
        axes.set_aspect(1)

    def plot_rasterized_line(
            self,
            fig_name: str,
            sim: LevelSetSolver,
            do_raw: bool=True,
            title_font_size: int=11,
            fig_size: tuple[float, float]=(7, 5,),
        ) -> None:
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(sim, r"Rasterized initial surface", sim.gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$x$  [-]")
        plt.ylabel(r"$z$  [-]")
        d: NDArray | MaskedArray
        extent_: NDArray | None
        if do_raw:
            d = sim.ρ
            extent_ = (
                sim.raw_domain.extent if sim.raw_domain is not None
                else None
            )
        else:
            d = sim.ρ_padded
            extent_ = (
                sim.domain.extent if sim.domain is not None
                else None
            )
        plt.imshow(
            d, 
            cmap="bwr", 
            extent=tuple(extent_) if extent_ is not None else None, 
            alpha=0.5, 
            norm=CenteredNorm(),
        )
        plt.plot(*sim.get_points_xz(), color="k", lw=0.5,)
        grid_aspect_ratio: float = d.shape[0]/d.shape[1]
        shrink: float = (
            0.65 if grid_aspect_ratio>0.65
            else grid_aspect_ratio
        )
        cbar = plt.colorbar(
            shrink=shrink, 
            pad=0.05, 
            aspect=10,
        )
        cbar.set_label("signed distance  [-]", size=10,)
        cbar.ax.tick_params(labelsize=9)
        plt.xlim(*sim.get_x_limits())
        plt.ylim(*sim.get_z_limits())
        plt.grid(ls=":", color="g",)
        axes = plt.gca()
        axes.set_aspect(1)

    def plot_φ(
            self,
            fig_name: str,
            sim: LevelSetSolver,
            do_φ: bool=False,
            do_φ_everywhere: bool=False,
            do_φ_next: bool=False,
            do_dφdx: bool=False,
            do_dφdz: bool=False,
            do_dTdx: bool=False,
            do_dTdz: bool=False,
            do_beta: bool=False,
            do_speed: bool=False,
            do_aux: bool=False,
            do_ξ0: bool=False,
            # do_travel_time: bool=False,
            do_arrival_time: bool=False,
            do_fix_ticks: bool=False,
            cbar_ticks: NDArray | None=None,
            clip: tuple[float,float]=(-0.2, 0.2,),
            title_font_size: int=11,
            fig_size: tuple[float,float]=(8, 6,),
        ) -> None:
        """
        Image grid of phi function (could be signed distance, β, ξ, etc).

        Parameters
        ----------

        fig_name: str
        sim: LevelSetSolver
        title_font_size: int=12
        do_φ: bool=False
        do_φ_everywhere: bool=False
        do_φ_next: bool=False
        do_dTdx: bool=False
        do_dTdz: bool=False
        do_beta: bool=False
        do_speed: bool=False
        do_travel_time: bool=False
        do_arrival_time: bool=False
        do_fix_ticks: bool=False
        cbar_ticks: NDArray | None=None
        clip: tuple[float, float]=(-0.2, 0.2,)
        fig_size: tuple[float, float]=(8, 6,)

        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        f: NDArray | MaskedArray | None
        d: NDArray | MaskedArray | None
        dd: NDArray | MaskedArray | None
        title: str
        cbar_label: str
        cmap: str
        do_centered_norm: bool
        if do_φ:
            f = sim.φ
            d = f
            dd = None
            title = r"Signed distance  {\varphi}"
            cbar_label = "signed distance  [-]"
            cmap = "bwr"
            do_centered_norm = True
        elif do_φ_everywhere:
            f = sim.φ_everywhere
            d = f
            dd = None
            title = r"Signed distance everywhere  {\varphi}_{\infty}"
            cbar_label = "signed distance  [-]"
            cmap = "bwr"
            do_centered_norm = True
        elif do_φ_next:
            f = sim.φ_next
            d = f
            dd = None
            title = r"Signed distance (next)  {\varphi}_\mathrm{next}"
            cbar_label = "signed distance  [-]"
            cmap = "bwr"
            do_centered_norm = True
        elif do_dφdx:
            f = sim.Δφ_x
            d = sim.φ
            dd = None
            title = r"Gradient {\partial}\varphi/{\partial}x"
            cbar_label = r"${\partial}\varphi/{\partial}x$  [-]"
            cmap = "bwr"
            do_centered_norm = True
        elif do_dφdz:
            f = sim.Δφ_z
            d = sim.φ
            dd = None
            title = r"Gradient {\partial}\varphi/{\partial}z"
            cbar_label = r"${\partial}\varphi/{\partial}z$  [-]"
            cmap = "bwr"
            do_centered_norm = True
        # elif do_dTdx:
        #     f = sim.dTdx
        #     d = sim.φ
        #     dd = None
        #     title = r"Gradient {d}T/{d}x"
        #     cbar_label = r"$\mathrm{d}T/\mathrm{d}x$  [-]"
        #     cmap = "bwr"
        #     do_centered_norm = True
        # elif do_dTdz:
        #     f = sim.dTdz
        #     d = sim.φ
        #     dd = None
        #     title = r"Gradient {d}T/{d}z"
        #     cbar_label = r"$\mathrm{d}T/\mathrm{d}z$  [-]"
        #     cmap = "bwr"
        #     do_centered_norm = True
        elif do_beta:
            f = np.rad2deg(sim.β)
            d = sim.φ_next
            dd = sim.φ
            title = r"Surface slope angle  \beta"
            cbar_label = r"$\beta$  [$\degree$]"
            cmap = "hsv" 
            do_centered_norm = True
        elif do_speed:
            f = sim.ξ
            d = sim.φ_next
            dd = sim.φ
            title = r"Surface erosion speed  \it{\xi^{\perp}}"
            cbar_label = r"$\xi^{\!\perp}$  [-]"
            cmap = "viridis"
            do_centered_norm = False
        elif do_aux:
            f = sim.φ_aux
            d = sim.φ_next
            dd = sim.φ
            title = r"Signed distance (aux)  {\varphi}_\mathrm{next}"
            cbar_label = "signed distance (aux)  [-]"
            cmap = "bwr"
            do_centered_norm = True
        elif do_ξ0:
            f = sim.ξ0
            d = None #sim.φ
            dd = None
            title = r"Substrate erodiblity   {\xi}_0"
            cbar_label = r"$\xi_0$  [-]"
            cmap = "managua"
            do_centered_norm = False
        # elif do_travel_time:
        #     f = np.clip(sim.T_signed, *clip,)
        #     d = sim.φ_next
        #     dd = sim.φ
        #     title = r"Travel time  \Delta{T}"
        #     cbar_label = r"$\Delta{T}$  [-]"
        #     cmap = "RdGy_r"
        #     do_centered_norm = True
        elif do_arrival_time and sim.T_arrival is not None:
            f = np.clip(sim.T_arrival, *clip,)
            d = sim.φ_next
            dd = sim.φ
            title = r"Surface arrival time  T"
            cbar_label = r"${T}$  [-]"
            cmap = "pink_r"
            do_centered_norm = False
        else:
            return
        if f is None:
            return
        plt.title(
            self.make_title(sim, title, sim.gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$x$  [-]")
        plt.ylabel(r"$z$  [-]")
        if (extent:=tuple(sim.get_extent())) is not None:
            plt.imshow(
                f, 
                cmap=cmap, 
                extent=extent, 
                norm=(CenteredNorm() if do_centered_norm else None),
            )
        plt.plot(*sim.get_points_xz(), color="k", lw=0.5,)
        if dd is not None:
            plt.plot(*sim.get_front_points(dd), lw=0.5, color="k", alpha=1,)
        lc: str = (
            "g" if do_arrival_time or do_beta or do_speed # or do_travel_time 
            else "k"
        )
        lw: float = (
            1 if do_arrival_time #or do_travel_time
            else 0.5
        )
        
        if d is not None:
            plt.plot(*sim.get_front_points(d), lw=lw, color=lc, alpha=1,)
            
        ticks: NDArray = (
            np.arange(-180,180+45,45,) if cbar_ticks is None else cbar_ticks
        )
        tick_label_fn: Callable = lambda x: f"{'+' if x>0 else ''}{x:1.0f}"
        tick_labels: list[str] = list(map(tick_label_fn, ticks))
        grid_aspect_ratio: float = f.shape[0]/f.shape[1]
        shrink: float = (
            0.65 if grid_aspect_ratio>0.65
            else grid_aspect_ratio
        )
        cbar: Colorbar = plt.colorbar(
            shrink=shrink, 
            pad=0.05, 
            aspect=10,
            ticks=ticks if do_fix_ticks else None,
            format=ticker.FixedFormatter(tick_labels) if do_fix_ticks else None,
        )
        cbar.set_label(cbar_label, size=10,)
        cbar.ax.tick_params(labelsize=9)

        plt.xlim(*sim.get_x_limits())
        plt.ylim(*sim.get_z_limits())
        plt.grid(ls=":", color="g",)
        axes = plt.gca()
        axes.set_aspect(1)

    def plot_time_slices(
            self,
            fig_name: str,
            sim: LevelSetSolver,
            x_limits: tuple[float|None,float|None] | None=None,
            z_min: float | None=-0.05,
            z_max: float | None=1.05,
            pc_slicing: float = 10,
            pc_labeling: float = 20,
            title_font_size: int=11,
            legend_font_size: int=9,
            fig_size: tuple[float, float]=(7, 5,),
        ) -> None:
        """
        Plot selection of front-propagation time T(x,z) slices.

        Parameters
        ----------
        fig_name: str
            Name used as key in figure dictionary.
        sim: LevelSetSolver
            Instance of level-set solver providing solution data.
        x_limits: tuple[float|None, float|None] | None = None
            Min/max x coordinates on graph.
        z_min: float | None=-0.05
            Minimum z coordinate on graph.
        z_max: float | None=1.05
            Maximum z coordinate on graph.
        pc_slicing: float = 10
            Period of slice plotting as percentage of total number of steps.
        pc_labeling: float = 20
            Period of slice labeling as percentage of total number of steps.
        title_font_size: int=11
            Size of title text.
        legend_font_size: int=9
            Size of legend text.
        fig_size: tuple[float, float]=(7, 5,)
            Figure size (in inches!).
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(sim, "Surface T slices", sim.gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel("x  [-]")
        plt.ylabel("z  [-]")
        cmap = plt.colormaps["brg"] 
        t_: float
        t_max: float = sim.t_total 
        label_: str | None
        n_slices: int = len(sim.T_slices.keys())
        pc_label: float = n_slices if n_slices<50 else pc_labeling
        do_label: Callable \
            = lambda step: (
                False if int((n_slices)*pc_label/100)==0
                else (step)%int((n_slices)*pc_label/100)==0
            )
        pc_plot: float = pc_label/pc_slicing
        do_plot: Callable \
            = lambda step: (
                False if int((n_slices)*pc_plot/100)==0
                else (step)%int((n_slices)*pc_plot/100)==0
            )
        for i_, t_slice_ in enumerate(sim.T_slices.values()):
            if not do_plot(i_):
                continue
            t_ = t_slice_["T"]
            if np.min(t_slice_["φ"])>=-sim.Δx:
                continue
            color_ = cmap(t_/t_max*0.9)
            label_ = (
                rf"$T$ = {np.round(t_,2):g}" if do_label(i_) 
                else None
            )
            plt.plot(
                t_slice_["x"], t_slice_["z"], 
                lw=(1.5 if n_slices<5 or do_label(i_) else 0.25),
                color=color_, 
                label=label_,
            )
        plt.legend(fontsize=legend_font_size, loc=(1.02,0.1))
        plt.xlim(sim.get_x_limits() if x_limits is None else x_limits)
        plt.ylim(
            sim.get_z_limits()[0] if z_min is None else z_min,
            sim.get_z_limits()[1] if z_max is None else z_max
        )
        plt.grid(ls=":",)
        axes = plt.gca()
        axes.set_aspect(1)

    def plot_retreat_speed(
            self,
            fig_name: str,
            sim: LevelSetSolver,
            z_levels: tuple=(0.8, 0.4, 0.2,),
            z_limits: tuple[float, float]=(0.8, 3,),
            title_font_size: int=11,
            fig_size: tuple[float, float]=(7, 3,),
        ) -> Sequence:
        """
        Plot estimated horizontal retreat speed vs time T.

        Parameters
        ----------
        fig_name: str
        sim: LevelSetSolver
        z_levels: tuple=(0.8, 0.4, 0.2,)
        z_limits: tuple[float, float]=(0.8, 3,)
        title_font_size: int=12
        fig_size: tuple[float, float]=(7, 3,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        title: str = (
            r"Horizontal retreat speed  \xi_{\mathrm{h}}" 
                # +r"  =(\partial{T}/\partial{x})^{-1}"
        )
        plt.title(
            self.make_title(sim, title, sim.gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.ylabel(r"Horizontal speed  $\xi_\mathrm{h}$")
        plt.xlabel(r"Arrival time  $T$")
        time_: NDArray
        speed_: NDArray
        for z_level_ in z_levels:
            t_  = np.array([
                ts_["T"]
                for ts_ in sim.T_slices.values() if np.any(ts_["z"]>z_level_)
            ])
            x_  = np.array([
                ts_["x"][ts_["z"]>=z_level_][0] 
                for ts_ in sim.T_slices.values() if np.any(ts_["z"]>z_level_)
            ])
            # z_  = np.array([
            #     ts_["x"][ts_["z"]>=z_level_][0] 
            #     for ts_ in sim.T_slices.values() if np.any(ts_["z"]>z_level_)
            # ])
            time_ = (t_[1:]+t_[:-1])/2
            speed_ = gaussian_filter((x_[1:]-x_[:-1])/(t_[1:]-t_[:-1]), 10,)
            plt.plot(time_, speed_, "-", label=rf"$z=${z_level_}",)
        plt.xlim(0, sim.t_total)
        plt.ylim(*z_limits)
        plt.grid(ls=":")
        plt.legend(fontsize=10, loc=(1.02, 0.3,),)
        return (z_level_, float(speed_[-1]),)

    def plot_front_segment(
            self,
            fig_name: str,
            sim: LevelSetSolver,
            z_range: tuple[float, float]=(0.1, 0.4,),
            title_font_size: int=11,
            fig_size: tuple[float, float]=(5, 5,),
        ) -> None:
        """
        Plot near-base cut of final time slice surface & regression fit.

        Parameters
        ----------
        fig_name: str
        sim: LevelSetSolver
        z_range: tuple[float, float]=(0.1, 0.4,)
        title_font_size: int=12
        fig_size: tuple[float, float]=(5, 5,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(sim, "Front segment tilt", sim.gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel("x  [-]")
        plt.ylabel("z  [-]")
        xz_: NDArray  = sim.get_front_points(sim.φ)
        x_: NDArray = xz_[0]
        z_: NDArray  = xz_[1]
        z_min_: float = z_range[0]
        z_max_: float = z_range[1]
        z_ = xz_[1][(xz_[1]>=z_min_) & (xz_[1]<z_max_)]
        x_ = xz_[0][(xz_[1]>=z_min_) & (xz_[1]<z_max_)]
        if x_.shape[0]>0:
            plt.plot(x_,z_, )
            plt.grid(ls=":")
            axes = plt.gca()
            axes.set_aspect(1)
            fit = linregress(x_,z_)
            plt.plot(x_, fit.slope*x_ + fit.intercept, "r:",);
            label: str = (
                r"$\widehat{\beta}_{\mathrm{ramp}} = $"
                + f"{float(np.round(np.rad2deg(np.arctan(fit.slope)),1))}"
                + r"$\degree$"
            )
            plt.text(0.1, 0.9, label, transform=axes.transAxes,)

    def plot_shocks(
            self,
            fig_name: str,
            sim: LevelSetSolver,
            title_font_size: int=11,
            fig_size: tuple[float, float]=(6,6,),
        ) -> None:
        """
        Plot near-base cut of final time slice surface & regression fit.

        Parameters
        ----------
        fig_name: str
        sim: LevelSetSolver
        z_range: tuple[float, float]=(0.1, 0.4,)
        title_font_size: int=12
        fig_size: tuple[float, float]=(5, 5,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(sim, "Shock progression", sim.gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.plot(
            *sim.shocks_xz, ".", 
            label="breaks in slope",
            zorder=10,
        )
        plt.xlabel("x  [-]")
        plt.ylabel("z  [-]")
        if (
            sim.gma is not None 
            and sim.gma.α_c0 is not None 
            and sim.gma.α_rs0 is not None
        ):
            angle: float = sim.gma.α_c0
            x = np.linspace(0, np.max(sim.shocks_xz[0]), 101,)
            y = x * np.tan((angle)) + 0.0
            alpha_rs0: float = np.rad2deg(sim.gma.α_rs0)
            plt.plot(
                x, y, 
                "-", 
                label=r"$\alpha_{rs0} =$"+rf"{alpha_rs0:0.2f}$\degree$",
            )
        axes = plt.gca()
        axes.set_aspect(1)
        plt.legend()
        plt.grid(ls=":",)