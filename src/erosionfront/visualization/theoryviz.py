"""
Visualization of geomorphic Hamiltonian properties.
"""
import warnings
from typing import Any, Generator
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sympy import parse_expr  #type: ignore
# from IPython import get_ipython #type: ignore

from erosionfront.visualization.base import GraphingBase
from erosionfront.geomorphic.surface import (
    StraightLineSurface, ErrorFunctionSurface,
    SemiCircularArcSurface, QuarterCircularArcSurface,
)
from erosionfront.theory.numerical import RichterSlopeModel, make_rays
from erosionfront.theory.arrays import Arrays

warnings.filterwarnings("ignore")

__all__ = ["TheoryViz"]

def mark_angles(
        x_lambda: Callable | None, 
        z_lambda: Callable | None, 
        idtx_fgtx_markers: list | None, 
        p_or_v: int=0, 
        do_skip_β0: bool=False,
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
    if x_lambda is None or z_lambda is None or idtx_fgtx_markers is None:
        return
    for angle_markup_ in reversed(idtx_fgtx_markers):
        if do_skip_β0 and "\\rightarrow 0" in angle_markup_["label"]:
            continue
        angle_ = angle_markup_["angle"]
        plt.plot(
            x_lambda(float(angle_)),
            z_lambda(float(angle_)),
            angle_markup_["marker"],
            color=angle_markup_["colors"][p_or_v],
            label=angle_markup_["label"],
            ms=angle_markup_["size"],
        )

def plot_segments(
        rsm: RichterSlopeModel, 
        x: NDArray, 
        y: NDArray, 
        β: NDArray, 
        label_fn: Callable[[str],str]=lambda x: f"{x}", 
        do_v: bool=False,
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
    prev_β_crit_ = 0
    colors_ = (
        (
            "DarkRed",
            "red",
            "darkMagenta",
        ) if do_v
        else (
            "DarkBlue",
            "blue",
            "blueviolet",
        )
    )
    line_styles_ = (
        "-",
        "-.",
        "-",
    )
    is_in_range = lambda array_, range_: (
        array_ >= range_[0]) & (array_ <= range_[1]
    )
    if rsm.β_crits is not None:
        for β_crit_, color_, ls_, concave_or_convex in zip(
                (rsm.β_crits + [β[-1]]), #np.pi/2
                colors_,
                line_styles_,
                ("convex", "concave", "convex"),
            ):
            plt.plot(
                x[(is_in_range(β, (prev_β_crit_, β_crit_)))],
                y[(is_in_range(β, (prev_β_crit_, β_crit_)))],
                color=color_,
                linestyle=ls_,
                label=label_fn(concave_or_convex),
            )
            prev_β_crit_ = β_crit_

def patch_idtx_fgtx_markers(
        β_crits: list, 
        markers: list,
        model: Any,
    ) -> Generator:
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
    n_β_crit = len(β_crits)
    for angle_markup_ in markers:
        if "β_crit" in angle_markup_["angle"]:
            for idx, β_crit_ in enumerate(β_crits):
                patched_angle_markup_ = angle_markup_.copy()
                patched_angle_markup_["angle"] = parse_expr(
                    angle_markup_["angle"].replace("β_crit", f"{β_crit_}")
                ).n()
                patched_angle_markup_["label"] = angle_markup_["label"].replace(
                    "=β_crit", (f"({idx+1})" if n_β_crit>1 else "")
                    +f"={np.rad2deg(β_crit_):0.1f}"
                )
                patched_angle_markup_["marker"] = angle_markup_["marker"][
                    min((idx, 1))
                ]
                yield(patched_angle_markup_)
        else:
            patched_angle_markup_ = angle_markup_.copy()
            patched_angle_markup_["angle"] = parse_expr(
                angle_markup_["angle"]
                .replace("phi", f"{model.phi_h}")
            ).n()
            patched_angle_markup_["label"] = angle_markup_["label"].replace(
                "=phi_h", f"={np.rad2deg(float(model.phi_h)):0.1f}"
            )
            yield(patched_angle_markup_)


class TheoryViz(GraphingBase):
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
    idtx_fgtx_markers: list | None
    
    def __init__(
            self, 
            model: Any,
            plot_parameters: Any, 
            rsm: RichterSlopeModel,
            *args, 
            **kwargs,
        ) -> None:
        """
        2D visualization class.

        Subclasses :class:`gmplib.plot.GraphingBase`.

        Arguments:
            kwargs:
                arguments to pass to :class:`gmplib.plot.GraphingBase` 
                constructor
        """
        self.set_markers(model, plot_parameters, rsm,)
        super().__init__(*args, **kwargs)

    def set_markers(
            self, 
            model: Any,
            plot_parameters: Any, 
            rsm: RichterSlopeModel,
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
        markers_ = (
            plot_parameters.idtx_fgtx_zero_marker 
            + plot_parameters.idtx_fgtx_markers
        )
        if rsm.β_crits is not None:
            self.idtx_fgtx_markers = sorted(
                list(patch_idtx_fgtx_markers(
                    rsm.β_crits, markers_, model,
                )),
                key=lambda angle_markup: angle_markup["angle"],
            )
        else:
            self.idtx_fgtx_markers = None

    def plot_erosionrate_vs_surfaceangle(
            self, 
            fig_name: str,
            rsm: RichterSlopeModel,
            arrays: Arrays,
            model: Any,
            aspect: float | None=None,
            fig_size: tuple[float,float]=(8,5),
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
        self.create_figure(fig_name, fig_size=fig_size,)
        if rsm.β_crits is not None:
            plot_segments(
                rsm, np.rad2deg(arrays.β), 1/arrays.p, arrays.β,
            )
        plt.xlabel(r"Surface angle  $\beta$  [$\degree$]")
        plt.ylabel(
            r"Surface-normal erosion speed  $\xi^{\perp}(\beta) = 1/p(\beta)$"
        )

        axes = plt.gca()
        if aspect is not None:
            axes.set_aspect(aspect)
        phi_str = str(model.phi_h).replace("pi",r"$\pi$")
        plt.text(
            *(0.95,0.77),
            rf"{model.beta_type} model, " 
                + rf"$\phi=${phi_str}, " \
                + rf"$n$={model.n}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=axes.transAxes,
            fontsize=15,
        )
        phi_deg = np.rad2deg(float(model.phi_h.n()))
        plt.plot(
            (phi_deg, phi_deg,), (0,1/arrays.p[-1],), 
            color='grey', ls=":" 
        )
        beta_lambda: Callable = lambda beta_: np.rad2deg(beta_)
        xi_lambda: Callable = lambda beta_: 1/rsm.p_lambda(beta_) #type: ignore
        mark_angles(
            beta_lambda, xi_lambda, self.idtx_fgtx_markers, p_or_v=0, 
            do_skip_β0=(
                model.beta_type=="sin" or model.n>1
            ),
        )
        plt.grid(ls=":")
        plt.legend()

    def plot_normalspeedxy_figuratrix(
            self, 
            fig_name: str,
            rsm: RichterSlopeModel,
            arrays: Arrays,
            model: Any,
            aspect: float | None=1,
            fig_size: tuple[float,float]=(6,6),
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
        self.create_figure(fig_name, fig_size=fig_size,)
        label_fn_ = (
            lambda x: r"$(\xi^x,\xi^z) \,\,|\,\,\mathcal{F}_{*}=1$ "+ f"({x})"
        )
        plot_segments(
            rsm, arrays.ξx, arrays.ξz, arrays.β, 
            label_fn=label_fn_,
        )
        plt.xlabel(r"$\xi^x = p_x/p^2$")
        plt.ylabel(r"$\xi^z =p_z/p^2$")
        mark_angles(
            rsm.ξx_lambda, rsm.ξz_lambda, self.idtx_fgtx_markers, p_or_v=0, 
        )
        axes = plt.gca()
        if aspect is not None:
            axes.set_aspect(aspect)
        phi_str = str(model.phi_h).replace("pi",r"$\pi$")
        plt.text(
            *(0.5,0.86),
            rf"{model.beta_type} model, "
            + rf"$\phi=${phi_str}, "
            + rf"$n$={model.n}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=axes.transAxes,
            fontsize=15,
        )
        plt.grid(ls=":")
        # plt.xlim(-0.1, 4,)
        # plt.ylim(-0.6, 0.5,)
        plt.legend(loc=(0.03,0.75), fontsize=9)

    def plot_normalspeed_figuratrix(
            self, 
            fig_name: str,
            rsm: RichterSlopeModel,
            arrays: Arrays,
            model: Any,
            aspect: float | None=1,
            fig_size: tuple[float,float]=(6,6),
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
        self.create_figure(fig_name, fig_size=fig_size,)
        label_fn_ = (
            lambda x: 
                r"$(\xi^{\rightarrow},\xi^{\downarrow}) \,\,|\,\, "
                + r"\mathcal{F}_{*}=1$ "+ f"({x})"
        )
        plot_segments(
            rsm, arrays.ξrt, arrays.ξup, arrays.β, 
            label_fn=label_fn_,
        )
        plt.xlabel(r"$\xi^{\rightarrow} = 1/p_x$")
        plt.ylabel(r"$-\xi^{\downarrow} = 1/p_z$")
        mark_angles(
            rsm.ξrt_lambda, rsm.ξup_lambda, self.idtx_fgtx_markers, p_or_v=0,
        )
        plt.xlim(None, 1.3,)
        plt.ylim(-11,10.1)
        axes = plt.gca()
        if aspect is not None:
            axes.set_aspect(aspect)
        phi_str = str(model.phi_h).replace("pi",r"$\pi$")
        plt.text(
            *(0.1,0.93),
            rf"{model.beta_type} model, "
            + rf"$\phi=${phi_str}, "
            + rf"$n$={model.n}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=axes.transAxes,
            fontsize=15,
        )
        plt.grid(ls=":")
        plt.legend(loc=(0.05,0.01), fontsize=10)

    def plot_normalslowness_figuratrix(
            self, 
            fig_name: str,
            rsm: RichterSlopeModel,
            arrays: Arrays,
            model: Any,
            plot_parameters: Any,
            aspect: float | None=1,
            fig_size: tuple[float,float]=(6,6),
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
        self.create_figure(fig_name, fig_size=fig_size,)
        label_fn_ = (
            lambda x: 
            r"$\mathbf{p}(\beta) \,\,|\,\,\mathcal{F}_{*}=1$ "+ f"({x})"
        )
        plot_segments(
            rsm, arrays.px, arrays.pz, arrays.β, 
            label_fn=label_fn_
        )
        plt.xlabel(r"$p_x$")
        plt.ylabel(r"$p_z$")
        mark_angles(
            rsm.px_lambda, rsm.pz_lambda, self.idtx_fgtx_markers, p_or_v=0,
        )
        plt.xlim(0,4,)
        plt.ylim(-4,0.2,)
        axes = plt.gca()
        if aspect is not None:
            axes.set_aspect(aspect)
        phi_str = str(model.phi_h).replace("pi",r"$\pi$")
        x: float
        y: float
        (x,y,) = plot_parameters.fgtx_label_loc
        plt.text(
            x, y,
            rf"{model.beta_type} model, "
            + rf"$\phi=${phi_str}, "
            + rf"$n$={model.n}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=axes.transAxes,
            fontsize=15,
        )
        plt.grid(ls=":")
        plt.legend(loc=plot_parameters.fgtx_legend_loc)

    def plot_ray_indicatrix(
            self, 
            fig_name: str,
            rsm: RichterSlopeModel,
            arrays: Arrays,
            model: Any,
            aspect: float | None=1,
            fig_size: tuple[float,float]=(6,6),
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
        self.create_figure(fig_name, fig_size=fig_size,)
        label_fn: Callable[[str],str] = (
            lambda x: r"$\mathbf{v}(\alpha) \,\,|\,\,\mathcal{F}=1$ "+ f"({x})"
        )
        plot_segments(
            rsm, 
            arrays.vx, 
            arrays.vz, 
            arrays.β, 
            label_fn, 
            do_v=True
        )
        plt.xlabel(r"$v^x$")
        plt.ylabel(r"$v^z$")
        mark_angles(
            rsm.vx_lambda, 
            rsm.vz_lambda, 
            self.idtx_fgtx_markers, 
            p_or_v=1,
        )
        axes = plt.gca()
        if aspect is not None:
            axes.set_aspect(aspect)
        phi_str = str(model.phi_h).replace("pi",r"$\pi$")
        plt.text(
            0.5,0.4,
            rf"{model.beta_type} model, " \
                + rf"$\phi=${phi_str}, " \
                + rf"$n$={model.n}",
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes.transAxes,
            fontsize=15,
        )
        plt.grid(ls=":")
        plt.legend(loc=(0.03,0.57,), fontsize=10)

    def plot_initial_profile(
            self,
            fig_name: str,
            rsm: RichterSlopeModel,
            surface: (
                StraightLineSurface 
                | ErrorFunctionSurface
                | SemiCircularArcSurface 
                | QuarterCircularArcSurface
            ),
            model: Any,
            aspect: float | None=None,
            Δt: float=0.15,
            fig_size: tuple[float,float]=(6,6),
            limits: tuple[tuple[float,float],tuple[float,float]] \
                = ((0.7,1.5,),(-0.03,1.26,)),
            do_zoom: bool=False,
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
        self.create_figure(fig_name, fig_size=fig_size,)
        vx: NDArray
        vz: NDArray
        (vx, vz,) = make_rays(rsm, surface.β)

        # Topo profile
        if surface.x is None or surface.z is None:
            raise ValueError("Surface points not given")
        plt.plot(
            surface.x,
            surface.z,
            "-",
            color="k",
            alpha=1,
            lw=2,
            label="initial topo profile",
        )
        plt.plot(
            surface.x+vx*Δt,
            surface.z+vz*Δt,
            "-",
            color="k",
            alpha=0.5,
            lw=1,
            label="final topo profile",
        )
        plt.xlabel(r"$x$")
        plt.ylabel(r"$z$")

        # Rays as arrows
        plt.plot(0, 0, "-", color="MediumBlue", label="rays")

        for i_, (x_, z_, Δx_, Δz_, β_) in enumerate(
            zip(
                surface.x, surface.z, 
                vx*Δt, vz*Δt, 
                surface.β,
            )
        ):
            is_critical: bool = bool(np.any([
                np.abs(β_-β_crit_)<0.02 for β_crit_ in rsm.β_crits
                ]) if rsm.β_crits is not None
                else False
            )
            lw_ = (
                (1 if (is_critical and β_<0.9) else 1)
                *(1 if do_zoom else 0.5)
            )
            plt.plot(
                (x_, x_+Δx_,),
                (z_, z_+Δz_,),
                lw=lw_,
                alpha=(1 if (is_critical and β_<0.9) else 0.5),
                color="Red" if is_critical else "MediumBlue",
            )
            # plt.arrow(
            #     x_,
            #     z_,
            #     Δx_,
            #     Δz_,
            #     lw=0.1,
            #     color="Red" if is_critical else "MediumBlue",
            #     length_includes_head=True,
            #     head_width=0.0005 if do_zoom else 0.025*0.01,
            # )
        
        # Info label
        axes = plt.gca()
        phi_str = str(model.phi_h).replace("pi",r"$\pi$")
        plt.text(
            *(0.03, 0.8 if do_zoom else 0.9),
            rf"{model.beta_type} model, "
            + rf"$\phi=${phi_str}, "
            + rf"$n$={model.n}",
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes.transAxes,
            fontsize=15,
        )
        if aspect is not None:
            axes.set_aspect(aspect)
        plt.grid(ls=":", alpha=0.5)
        # plt.legend(loc=None, fontsize=11)
        plt.xlim(limits[0],)
        plt.ylim(limits[1],)