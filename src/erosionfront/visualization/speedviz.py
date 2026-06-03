"""
Visualization of model erosion front speed.
"""
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from numpy.ma import MaskedArray
import matplotlib.pyplot as plt

from erosionfront.visualization.base import GraphingBase
from erosionfront.geomorphic.model import Isotropic, Step, ExponentialActivation
from erosionfront.geomorphic.gma import GeometricMechanicsAnalysis

warnings.filterwarnings("ignore")

__all__ = ["ErosionSpeedViz"]

class ErosionSpeedViz(GraphingBase):
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

    def plot_model_erosion_speed(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis | None = None,
            do_polar: bool=False,
            title_font_size: int=12,
            fig_size: tuple[float, float] | None=None,
        ) -> None:
        """
        Plot surface-normal erosion speed model ξ(β).

        Parameters
        ----------
        fig_name: str
        model: Any
        title_font_size: int=12
        fig_size: tuple[float, float]=(6, 2.5,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        def plot_markers(
                gma: GeometricMechanicsAnalysis,
                x_fn: Callable, 
                y_fn: Callable,
                do_reverse: bool=False,
            ) -> None:
            β_t: float = model.β_t
            β_c0: float = 0 if gma.β_c0 is None else gma.β_c0
            β_c1: float = 0 if gma.β_c1 is None else gma.β_c1
            β_mpx: float = 0 if gma.β_mpx is None else gma.β_mpx
            β_rs0: float = 0 if gma.β_rs0 is None else gma.β_rs0
            β_rs1: float = 0 if gma.β_rs1 is None else gma.β_rs1
            x_: float
            y_: float
            color_: str
            marker_: str
            label_: str
            nrv: bool = do_reverse
            label_βrs1: str = (
                r"$\beta_\mathrm{rs1} = $"
                + rf"{np.rad2deg(β_rs1):3.1f}$\degree$"
            )
            label_mpx: str = (
                r"$\beta_\mathrm{mpx} = $"
                + rf"{np.rad2deg(β_mpx):3.1f}$\degree$"

            )
            markers_: list = list(zip(
                    (
                        x_fn(β_rs1 if nrv else β_mpx),
                        x_fn(β_mpx if nrv else β_rs1), 
                        x_fn(β_t), 
                        x_fn(β_c0), 
                        x_fn(β_c1),
                        x_fn(β_rs0),
                    ),
                    (
                        y_fn(β_rs1 if nrv else β_mpx),
                        y_fn(β_mpx if nrv else β_rs1),
                        y_fn(β_t),
                        y_fn(β_c0),
                        y_fn(β_c1),
                        y_fn(β_rs0),
                    ),
                    (
                        self.props[("βrs1_color" if nrv else "βmpx_color")],
                        self.props[("βmpx_color" if nrv else "βrs1_color")],
                        self.props["βt_color"],
                        self.props["βc0_color"],
                        self.props["βc1_color"],
                        self.props["βrs0_color"],
                    ),
                    (
                        self.props[("βrs1_marker" if nrv else "βmpx_marker")],
                        self.props[("βmpx_marker" if nrv else "βrs1_marker")],
                        self.props["βt_marker"],
                        self.props["βc0_marker"],
                        self.props["βc1_marker"],
                        self.props["βrs0_marker"],
                    ),
                    (
                        (label_βrs1 if nrv else label_mpx),
                        (label_mpx if nrv else label_βrs1),
                        r"$\beta_\mathrm{t} = $"
                            +rf"{np.rad2deg(β_t):3.1f}$\degree$",
                        r"$\beta_\mathrm{c0} = $"
                            +rf"{np.rad2deg(β_c0):3.1f}$\degree$",
                        r"$\beta_\mathrm{c1} = $"
                            +rf"{np.rad2deg(β_c1):3.1f}$\degree$",
                        r"$\beta_\mathrm{rs0} = $"
                            +rf"{np.rad2deg(β_rs0):3.1f}$\degree$",
                    ),
                ))
            markers_ = markers_[::-1] if do_reverse else markers_
            for (x_, y_, color_, marker_, label_,)  in markers_:
                plt.scatter(
                    x_, 
                    y_,
                    marker=marker_,
                    color=color_,
                    s=60 if marker_=="*" else 30,
                    alpha=0.75 if marker_=="*" else 1,
                    label=label_,
                    zorder=10,
                )

        title: str
        beta: NDArray
        speed: NDArray
        x_fn: Callable
        y_fn: Callable
        if not do_polar:
            fig_size_ = fig_size if fig_size is not None else (6, 2.5,)
            self.create_figure(fig_name, fig_size=fig_size_,)
            plt.xlabel(r"$\beta$  $[\degree]$")
            plt.ylabel(r"$\xi^{\!\perp}(\beta)$  [-]")
            beta = np.linspace(-10,180,191)
            speed = np.array(model.ξ_fn_β(np.deg2rad(beta)))
            # print(speed)
            plt.plot(beta, speed, "k", zorder=-1,)
            plt.ylim(-0.05, np.max(speed)*1.05)
            plt.xlim(beta[0],beta[-1])
            x_fn = np.rad2deg
            y_fn = model.ξ_fn_β
            if gma is not None:
                plot_markers(gma, x_fn, y_fn,)
            plt.legend(fontsize=9, loc="upper left",)
        else:
            fig_size_ = fig_size if fig_size is not None else (6, 6,)
            self.create_figure(fig_name, fig_size=fig_size_,)
            beta = np.linspace(0,180,131)
            speed = np.array(model.ξ_fn_β(np.deg2rad(beta)))
            # DO THIS FIRST because otherwise polar axes may not be created
            # axes = fig.add_subplot(projection="polar")
            plt.polar(np.deg2rad(beta), speed, "k", zorder=-1,)
            axes = plt.gca()
            axes.set_thetalim(0, np.deg2rad(130)) #type: ignore
            ξ_max: float = float(model.ξ_fn_β((np.pi/2))*1.05)
            axes.set_ylim(0, ξ_max,) #
            plt.text(
                np.deg2rad(116), ξ_max*1.15, 
                r"$\beta$  $[\degree]$"
            )
            plt.text(
                -0.4, 0.43,
                r"$\xi^{\!\perp}(\beta)$  [-]"
            )
            x_fn = lambda x: x
            y_fn = model.ξ_fn_β
            if gma is not None:
                plot_markers(gma, x_fn, y_fn, do_reverse=True,)
            plt.legend(fontsize=9, loc=(-0.05,0.1,))
        title = r"Model erosion speed  \it{\xi^{\perp}(\beta)}"
        plt.title(
            self.make_title(model, title, gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.grid(ls=":")
