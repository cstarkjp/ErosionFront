"""
Visualization of geometric mechanics analysis of geomorphic Hamiltonian.
"""
import warnings
from typing import Any
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.colorbar import Colorbar

from erosionfront.visualization.base import GraphingBase
from erosionfront.geomorphic.model import Isotropic, Step, ExponentialActivation
from erosionfront.geomorphic.gma import GeometricMechanicsAnalysis

warnings.filterwarnings("ignore")

__all__ = ["GeometricMechanicsViz"]

class GeometricMechanicsViz(GraphingBase):
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

    def __init__(
            self, 
            *args, 
            **kwargs,
        ) -> None:
        """
        2D visualization class.

        Subclasses `GraphingBase`.

        Arguments:
            kwargs:
                arguments to pass to constructor
        """
        self.props = {
            "βt_color" : "orange", #"magenta","violet"
            "βt_marker" : "o",
            "βc0_color" : "magenta", #"magenta","violet"
            "βc0_marker" : "v",
            "βc1_color" : "darkorchid", #"darkorchid",
            "βc1_marker" : "^",
            "βrs0_color" : "magenta", #"magenta","violet"
            "βrs0_marker" : "h",
            "βrs1_color" : "darkorchid", #"darkorchid",
            "βrs1_marker" : "H",
            "βmpx_color" : "g",
            "βmpx_marker" : "*",
            "cvx0_color" : "black",  #"blue",
            "cvx1_color" : "0.4", #"midnightblue",
            "ccv_color" : "crimson",
        }
        super().__init__(*args, **kwargs)

    def plot_model_pxpz(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            n_pts: int=301,
            do_pz: bool=False,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(6, 2.5,),
        ) -> None:
        """
        Plot erosion model slowness components p_x, p_z vs β.

        Parameters
        ----------
        fig_name: str
        model: Any
        n_pts: int=301
        do_pz: bool=False
        title_font_size: int=12
        fig_size: tuple[float, float]=(6, 2.5,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        p_component: str = (
            r"p_z" if do_pz else r"p_x"
        )
        plt.title(
            self.make_title(
                model, f"Normal slowness component  {p_component}", gma,
            ),
            fontdict={"fontsize": title_font_size,},
        )
        y_label: str = (
            r"$p_z$  [-]" if do_pz else r"$p_x$  [-]"
        )
        plt.ylabel(y_label)
        plt.xlabel(r"$\beta$  [$\degree$]")
        x_fn: Callable = np.rad2deg
        y_fn: Callable[[float | NDArray], float | NDArray] = (
            gma.pz_onshell if do_pz
            else gma.px_onshell
        )

        β_c0: float = 0 if gma.β_c0 is None else gma.β_c0
        β_c1: float = 0 if gma.β_c1 is None else gma.β_c1
        β_mpx: float = 0 if gma.β_mpx is None else gma.β_mpx
        β_rs0: float = 0 if gma.β_rs0 is None else gma.β_rs0
        β_rs1: float = 0 if gma.β_rs1 is None else gma.β_rs1

        # Plot main curve as three different-color segments
        β: NDArray = np.linspace(0, np.pi, n_pts, endpoint=True,)
        plt.plot(
            x_fn(β[β<=β_c0]), 
            y_fn(β[β<=β_c0]),
            self.props["cvx0_color"], 
            lw=1.5,
        )
        plt.plot(
            x_fn(β[(β>=β_c0) & (β<=β_c1)]), 
            y_fn(β[(β>=β_c0) & (β<=β_c1)]), 
            self.props["ccv_color"], 
            lw=1.5,
        )
        plt.plot(
            x_fn(β[β>β_c1]), 
            y_fn(β[β>β_c1]), 
            self.props["cvx1_color"], 
            lw=1.5,
        )

        x_: float | NDArray
        y_: float | NDArray
        color_: str
        marker_: str
        for (x_, y_, color_, marker_, label_,)  in list(zip(
                (
                    x_fn(β_rs1), 
                    x_fn(β_mpx), 
                    x_fn(β_c0), 
                    x_fn(β_c1), 
                    x_fn(β_rs0),
                 ),
                (
                    y_fn(β_rs1),
                    y_fn(β_mpx),
                    y_fn(β_c0),
                    y_fn(β_c1),
                    y_fn(β_rs0),
                 ),
                (
                    self.props["βrs1_color"], 
                    self.props["βmpx_color"], 
                    self.props["βc0_color"], 
                    self.props["βc1_color"], 
                    self.props["βrs0_color"],
                 ),
                (
                    self.props["βrs1_marker"], 
                    self.props["βmpx_marker"], 
                    self.props["βc0_marker"], 
                    self.props["βc1_marker"], 
                    self.props["βrs0_marker"],
                 ),
                (
                    r"transition $\beta_\mathrm{rs1} = $"
                        +rf"{np.rad2deg(β_rs1):3.1f}$\degree$",
                    r"max $p_x$ @ $\beta_\mathrm{mpx} = $"
                        +rf"{np.rad2deg(β_mpx):3.1f}$\degree$",
                    r"$\det{g_*}=0$  @  $\beta_{\mathrm{c0}} = $" 
                        + rf"{np.rad2deg(β_c0):3.1f}$\degree$",
                    r"$\det{g_*}=0$  @  $\beta_{\mathrm{c1}} = $"
                        + rf"{np.rad2deg(β_c1):3.1f}$\degree$",
                    r"transition $\beta_\mathrm{rs0} = $"
                        +rf"{np.rad2deg(β_rs0):3.1f}$\degree$",
                  ),
            ))[::-1]:
            plt.scatter(
                x_, 
                y_,
                marker=marker_,
                color=color_,
                s=50 if marker_=="*" else 25,
                alpha=0.75 if marker_=="*" else 1,
                label=label_,
                zorder=10,
            )

        plt.legend(fontsize=8,)
        plt.grid(ls=":")
        plt.xlim(0,180)

    def plot_model_Hamiltonian_onshell(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            title_font_size: int=12,
            fig_size: tuple[float,float]=(6, 2,),
        ) -> None:
        """
        Plot on-shell Hamiltonian H(β), which should be H=1/2 everywhere.

        Parameters
        ----------
        fig_name: str
        model: Any
        title_font_size: int=12
        fig_size: tuple[float,float]=(6, 2,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(
                model,
                r"Hamiltonian at erosion surface  \mathcal{H}(\beta)",
                gma,
            ),
            fontdict={"fontsize": title_font_size,},
        )
        plt.ylabel(r"$\mathcal{H}$  [-]")
        plt.xlabel(r"$\beta$  [$\degree$]")
        β: NDArray = np.linspace(-0.,np.pi/1,101)
        px: NDArray = np.array(gma.px_onshell(β))
        pz: NDArray = np.array(gma.pz_onshell(β))
        plt.plot(np.rad2deg(β), gma.Hamiltonian(px, pz))
        plt.ylim(0,1)
        plt.xlim(0,180)
        plt.grid(ls=":")

    def plot_model_Hamiltonian_grid(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(5, 6,),
        ) -> None:
        """
        Image Hamiltonian H and plot H=1/2 figuratrix on p_x, p_z grid.

        Parameters
        ----------
        fig_name: str
        model: Any
        title_font_size: int=12
        fig_size: tuple[float, float]=(5, 6,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(
                model, 
                r"Hamiltonian  \mathcal{H}(p_x,p_z)",
                gma,
            ),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$p_x$")
        plt.ylabel(r"$p_z$")
        H_interp_grid_clipped: NDArray = np.clip(gma.H_interp_grid, 0, 1,)
        H_interp_grid_clipped[0,0] = 0  # force cbar range to include zero
        H_interp_grid_clipped[-1,-1] = 1  # force cbar range to include one
        extent_: tuple[float,float,float,float] = (*gma.px_span, *gma.pz_span)
        plt.imshow(
            (np.flipud(H_interp_grid_clipped.T)), 
            extent=extent_, 
            alpha=0.5,
            cmap="bwr_r",
        )
        ticks: tuple = (0, 0.5, 1,)
        grid_aspect_ratio: float = (
            H_interp_grid_clipped.T.shape[0]/H_interp_grid_clipped.T.shape[1]
        )
        shrink: float = (
            0.4 if grid_aspect_ratio>0.4
            else grid_aspect_ratio
        )        
        cbar: Colorbar = plt.colorbar(
            shrink=shrink, 
            pad=0.05, 
            aspect=10,
            ticks=ticks,
        )
        cbar.set_label(r"$\mathcal{H}$  [-]", size=12,)
        # cbar.ax.tick_params(labelsize=9)
        # plt.contour(
        #     model.H_interp_grid.T, 
        #     extent=extent_, 
        #     levels=[0.5],
        #     linewidths=[2],
        #     colors=["k"],
        #     linestyles=[":"],
        # )
        plt.plot(
            gma.px,
            gma.pz, 
            color="k", 
            lw=1,
            label=r"on-shell $\mathcal{H}=\frac{1}{2}$",
        )
        plt.legend(loc=(1.08, 0.8,))
        axes = plt.gca()
        axes.set_aspect(1)
        plt.grid(ls=":")

    def plot_model_detgstar_grid(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(5, 6,),
        ) -> None:
        """
        Image |g_*| on p_x, p_z grid plus H=1/2 contour, critical β angles.

        Parameters
        ----------
        fig_name: str
        model: Any
        title_font_size: int=12
        fig_size: tuple[float, float]=(5, 6,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        fig = self.create_figure(fig_name, fig_size=fig_size,)
        # plt.title(r"Determinant of dual metric tensor  $\det{g_*}$",
        plt.title(
            self.make_title(
                model, 
                r"Convexity \det{g_*} & figuratrix ||\mathbf{p}||_{g*}=1",
                gma,
            ), 
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$p_x$")
        plt.ylabel(r"$p_z$")
        extent_: tuple[float,float,float,float] = (*gma.px_span, *gma.pz_span)
        sf: float = 1/np.max(gma.det_gstar_grid)
        grid: NDArray = np.clip(gma.det_gstar_interp_grid*sf, -1,1)
        image = plt.imshow(
            (np.flipud(grid.T)), 
            extent=extent_, 
            alpha=0.5,
            cmap="bwr_r",
            norm=CenteredNorm(),
        )

        β_c0: float = 0 if gma.β_c0 is None else gma.β_c0
        β_c1: float = 0 if gma.β_c1 is None else gma.β_c1
        β_mpx: float = 0 if gma.β_mpx is None else gma.β_mpx
        β_rs0: float = 0 if gma.β_rs0 is None else gma.β_rs0
        β_rs1: float = 0 if gma.β_rs1 is None else gma.β_rs1

        # plt.contour(det_gstar, extent=extent_, levels=[0])
        # plt.contour(H_interp_grid.T, extent=extent_, levels=[0.5])
        β: NDArray = gma.β
        plt.plot(
            gma.px[β<=β_c0],
            gma.pz[β<=β_c0], 
            color=self.props["cvx0_color"], 
            lw=1.5,
            label=r"$\mathcal{H}=\frac{1}{2}$  (convex)",
        )
        plt.plot(
            gma.px[(β>=β_c0) & (β<=β_c1)],
            gma.pz[(β>=β_c0) & (β<=β_c1)], 
            color=self.props["ccv_color"], 
            lw=1,
            label=r"$\mathcal{H}=\frac{1}{2}$  (concave)",
        )
        plt.plot(
            gma.px[β>=β_c1],
            gma.pz[β>=β_c1], 
            color=self.props["cvx1_color"], 
            lw=1,
            # label=r"$\mathcal{H}=\frac{1}{2}$",
        )

        x_: float
        y_: float
        color_: str
        marker_: str
        x_fn: Callable = gma.px_onshell
        y_fn: Callable = gma.pz_onshell
        for (x_, y_, color_, marker_, label_,)  in list(zip(
                (
                    x_fn(β_rs1), 
                    x_fn(β_mpx), 
                    x_fn(β_c0), 
                    x_fn(β_c1),
                    x_fn(β_rs0), 
                 ),
                (
                    y_fn(β_rs1),
                    y_fn(β_mpx),
                    y_fn(β_c0),
                    y_fn(β_c1),
                    y_fn(β_rs0),
                 ),
                (
                    self.props["βrs1_color"], 
                    self.props["βmpx_color"], 
                    self.props["βc0_color"], 
                    self.props["βc1_color"],
                    self.props["βrs0_color"],
                 ),
                (
                    self.props["βrs1_marker"], 
                    self.props["βmpx_marker"], 
                    self.props["βc0_marker"], 
                    self.props["βc1_marker"],
                    self.props["βrs0_marker"],
                 ),
                (
                    r"transition $\beta_\mathrm{rs1} = $"
                        +rf"{np.rad2deg(β_rs1):3.1f}$\degree$",
                    r"max $p_x$ @ $\beta_\mathrm{mpx} = $"
                        +rf"{np.rad2deg(β_mpx):3.1f}$\degree$",
                    r"$\det{g_*}=0$  @  $\beta_{\mathrm{c0}} = $" 
                        + rf"{np.rad2deg(β_c0):3.1f}$\degree$",
                    r"$\det{g_*}=0$  @  $\beta_{\mathrm{c1}} = $"
                        + rf"{np.rad2deg(β_c1):3.1f}$\degree$",
                    r"transition $\beta_\mathrm{rs0} = $"
                        +rf"{np.rad2deg(β_rs0):3.1f}$\degree$",
                 ),
            ))[::-1]:
            plt.scatter(
                x_, 
                y_,
                marker=marker_,
                color=color_,
                s=50 if marker_=="*" else 25,
                alpha=0.75 if marker_=="*" else 1,
                label=label_,
                zorder=10,
            )

        plt.legend(loc=(1.07,0.67), fontsize=9, borderpad=0.3,)

        axes = plt.gca()
        # axins = inset_axes(
        #     axes,
        #     width="7%",  # width: 5% of parent_bbox width
        #     height="40%",  # height: 50%
        #     loc="lower left",
        #     bbox_to_anchor=(550,-50,350,800),
        #     # bbox_to_anchor=(1,-0.2,1.5,1),
        #     # bbox_transform=axes.transAxes,
        #     borderpad=10,
        # )
        # fig.colorbar(image, cax=axins)
        grid_aspect_ratio: float = (
            grid.T.shape[0]/grid.T.shape[1]
        )
        shrink: float = (
            0.3 if grid_aspect_ratio>0.3
            else grid_aspect_ratio
        )     
        cbar: Colorbar = plt.colorbar(
            image,
            # cax=axins,
            shrink=shrink, 
            pad=0.05, 
            aspect=10,
            location="right",
        )
        cbar.set_label(
            r"modified $\det{g_*}$", 
            size=10,
        )
        cbar.ax.tick_params(labelsize=9)

        axes.set_aspect(1)
        plt.grid(ls=":")

    def plot_model_detgstar_onshell(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(6, 3,),
        ) -> None:
        """
        Plot |g_*| vs β colorized by convex or concave; also critical β values.

        Parameters
        ----------
        fig_name: str
        model: Any
        title_font_size: int=12
        fig_size: tuple[float, float]=(6, 3,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(
                model, 
                r"Dual metric tensor determinant  "
            +r"\det{g_{*}(\beta)} \,\,|\,\, \mathcal{H}(\beta)=\frac{1}{2}",
                gma,
            ),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$\beta$  [$\degree$]")
        plt.ylabel(r"normalized  $\det{g_*}$")

        β: NDArray = gma.β
        pos_l: NDArray = np.s_[(β<gma.β_c0)]
        pos_r: NDArray = np.s_[(β>gma.β_c1)]
        # zero = np.s_[np.abs(det_gstar_onshell)<approx_zero]
        neg : NDArray= np.s_[(β>gma.β_c0) & (β<gma.β_c1)]
        sf: float = 1/np.max(gma.det_gstar_onshell)
        plt.plot(
            np.rad2deg(β[pos_l][3:]), 
            gma.det_gstar_onshell_interpfn(β[pos_l][3:])*sf,
            color=self.props["cvx0_color"],
            label=r"convex $\mathcal{H}$"
        )
        plt.plot(
            np.rad2deg(β[neg]), 
            gma.det_gstar_onshell_interpfn(β[neg])*sf,
            color=self.props["ccv_color"],
            label=r"concave $\mathcal{H}$"
        )
        plt.plot(
            np.rad2deg(β[pos_r][:-10]), 
            gma.det_gstar_onshell_interpfn(β[pos_r][:-10])*sf,
            color=self.props["cvx1_color"],
        )

        β_c0: float = 0 if gma.β_c0 is None else gma.β_c0
        β_c1: float = 0 if gma.β_c1 is None else gma.β_c1
        β_mpx: float = 0 if gma.β_mpx is None else gma.β_mpx

        x_: float
        y_: float
        color_: str
        marker_: str
        label_: str
        x_fn: Callable = np.rad2deg
        y_fn: Callable = lambda y: gma.det_gstar_onshell_interpfn(y)*sf
        for (x_, y_, color_, marker_, label_,)  in zip(
                (x_fn(β_mpx), 
                 x_fn(β_c0), 
                 x_fn(β_c1),),
                (y_fn(β_mpx),
                 y_fn(β_c0),
                 y_fn(β_c1),),
                (self.props["βmpx_color"], 
                 self.props["βc0_color"], 
                 self.props["βc1_color"],),
                (self.props["βmpx_marker"], 
                 self.props["βc0_marker"], 
                 self.props["βc1_marker"],),
                (r"max $p_x$ @ $\beta_\mathrm{mpx}=$"
                    +rf"{np.rad2deg(β_mpx):3.1f}$\degree$",
                 r"$\det{g_*}=0$  @  $\beta_{\mathrm{c0}} = $" 
                 + rf"{np.rad2deg(β_c0):3.1f}$\degree$",
                 r"$\det{g_*}=0$  @  $\beta_{\mathrm{c1}} = $"
                  + rf"{np.rad2deg(β_c1):3.1f}$\degree$",),
            ):
            plt.scatter(
                x_, 
                y_,
                marker=marker_,
                color=color_,
                s=50 if marker_=="*" else 25,
                alpha=0.75 if marker_=="*" else 1,
                label=label_,
                zorder=10,
            )

        plt.legend(fontsize=9)
        plt.grid(ls=":")
        # plt.ylim(-1.5,1.05)
        # plt.xlim(0,180)
        
    def plot_model_α_β(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(5, 5,),
        ) -> None:
        """
        Plot ray angle α(β) annotated by critical angles, concave/convex colors.

        Parameters
        ----------
        fig_name: str
        model: Any
        title_font_size: int=12
        fig_size: tuple[float, float]=(5, 5,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(
                model, 
                r"Ray angle \alpha versus slope angle \beta",
                gma,
            ),
            fontdict={"fontsize": title_font_size,},
        )
        plt.ylabel(r"$\alpha$  [$\degree$]")
        plt.xlabel(r"$\beta$  [$\degree$]")

        β: NDArray = gma.β
        α: NDArray = gma.α_onshell_interpfn(β)
        β_c0: float = 0 if gma.β_c0 is None else gma.β_c0
        β_c1: float = 0 if gma.β_c1 is None else gma.β_c1
        beta: NDArray = np.rad2deg(β)
        alpha: NDArray = np.rad2deg(α)
        plt.plot(beta[β<β_c0], alpha[β<β_c0], self.props["cvx0_color"], lw=1.5,)
        plt.plot(
            beta[(β>=β_c0) & (β<=β_c1)], 
            α[(β>=β_c0) & (β<=β_c1)], 
            self.props["ccv_color"], lw=1.5,
        )
        plt.plot(beta[β>β_c1], α[β>β_c1], self.props["cvx1_color"], lw=1.5,)

        β_mpx: float = 0 if gma.β_mpx is None else gma.β_mpx
        x_: float
        y_: float
        color_: str
        marker_: str
        label_: str
        x_fn: Callable = np.rad2deg
        y_fn: Callable = lambda y: np.rad2deg(gma.α_onshell_interpfn(y))
        for (x_, y_, color_, marker_, label_,)  in zip(
                (x_fn(β_mpx), 
                 x_fn(β_c0), 
                 x_fn(β_c1),),
                (y_fn(β_mpx),
                 y_fn(β_c0),
                 y_fn(β_c1),),
                (self.props["βmpx_color"], 
                 self.props["βc0_color"], 
                 self.props["βc1_color"],),
                (self.props["βmpx_marker"], 
                 self.props["βc0_marker"], 
                 self.props["βc1_marker"],),
                (r"max $p_x$ @ $\beta_\mathrm{mpx} = $"
                    +rf"{np.rad2deg(β_mpx):3.1f}$\degree$",
                 r"$\det{g_*}=0$  @  $\beta_{\mathrm{c0}} = $" 
                 + rf"{np.rad2deg(β_c0):3.1f}$\degree$",
                 r"$\det{g_*}=0$  @  $\beta_{\mathrm{c1}} = $"
                  + rf"{np.rad2deg(β_c1):3.1f}$\degree$",),
            ):
            plt.scatter(
                x_, 
                y_,
                marker=marker_,
                color=color_,
                s=50 if marker_=="*" else 25,
                alpha=0.75 if marker_=="*" else 1,
                label=label_,
                zorder=10,
            )
        
        axes = plt.gca()
        axes.set_aspect(1)
        plt.legend(loc="lower right", fontsize=9,)
        plt.grid(ls=":")

    def plot_model_pdotv(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(6, 2.5,),
        ) -> None:
        """
        Plot p(v) aka p dot v vs β, which for conjugacy should be =1.

        Parameters
        ----------
        fig_name: str
        model: Any
        title_font_size: int=12
        fig_size: tuple[float, float]=(6, 2.5,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(
                model, 
                r"Conjugacy of slowness covector {p} "
                            + r"and ray velocity {v}",
                gma,
            ),
            fontdict={"fontsize": title_font_size,},
        )
        plt.ylabel(r"$\mathbf{p}\mathbf{\cdot}\mathbf{v}$  [-]")
        plt.xlabel(r"$\beta$  [$\degree$]")
        plt.plot(np.rad2deg(gma.β), gma.p_dot_v())
        # plt.legend(fontsize=10)
        plt.ylim(0,2)
        plt.xlim(0,180)
        plt.grid(ls=":")

    def plot_threshold_ray_angle(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(4, 5,),
        ) -> None:
        """
        Plot β(α-α_c1) and the critical slope angle β_rs1(α_rs1).

        Parameters
        ----------
        fig_name: str
        model: Any
        title_font_size: int=12
        fig_size: tuple[float, float]=(4, 5,)
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(model, "Threshold ray angle \\alpha", gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$\alpha - \alpha_{\mathrm{c1}}$  [$\degree$]")
        plt.ylabel(r"$\beta$  [$\degree$]")

        β_c0: float = 0 if gma.β_c0 is None else gma.β_c0
        β_rs1: float = 0 if gma.β_rs1 is None else gma.β_rs1
        α_rs1: float = 0 if gma.α_rs1 is None else gma.α_rs1
        α_c1: float = 0 if gma.α_c1 is None else gma.α_c1
        
        β = gma.β
        plt.scatter(
            np.rad2deg(α_rs1-α_c1), 
            np.rad2deg(β_rs1), 
            color = self.props["βrs1_color"], 
            marker = self.props["βrs1_marker"], 
            label = (
                r"$\beta_{\mathrm{rs1}}$"
                + rf"$= {np.rad2deg(β_rs1):3.1f}\degree$,  "
                + r"$\alpha_{\mathrm{rs1}} = \alpha_{\mathrm{c1}} $"
                + rf"$= {np.rad2deg(α_rs1):3.1f}\degree$"
            ),
            zorder=10,
        )
        plt.plot(
            np.rad2deg(gma.α_onshell_interpfn(β[β<β_c0])-α_c1),
            np.rad2deg(β[β<β_c0]), 
        )
        plt.legend()
        plt.grid(ls=":")

    def plot_ray_velocity(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(5, 5,),
        ) -> None:
        """
        Plot ray velocity indicatrix, plus convex/concave colors, critical βs.

        Parameters
        ----------
        fig_name: str,
        model: Any,
        title_font_size: int=12,
        fig_size: tuple[float, float]=(5, 5,),
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(model, r"Ray velocity  v", gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$v^x$  [-]")
        plt.ylabel(r"$v^z$  [-]")

        β: NDArray = gma.β[gma.dHdpz_onshell<1.3]
        vx = gma.dHdpx_onshell[gma.dHdpz_onshell<1.3]
        vz = gma.dHdpz_onshell[gma.dHdpz_onshell<1.3]
        vx_max = np.max(vx)*1.1

        β_c0: float = 0 if gma.β_c0 is None else gma.β_c0
        β_c1: float = 0 if gma.β_c1 is None else gma.β_c1
        β_mpx: float = 0 if gma.β_mpx is None else gma.β_mpx
        β_rs0: float = 0 if gma.β_rs0 is None else gma.β_rs0
        β_rs1: float = 0 if gma.β_rs1 is None else gma.β_rs1
        α_c0: float = 0 if gma.α_c0 is None else gma.α_c0
        α_c1: float = 0 if gma.α_c1 is None else gma.α_c1
        α_rs0: float = 0 if gma.α_rs0 is None else gma.α_rs0
        α_rs1: float = 0 if gma.α_rs1 is None else gma.α_rs1

        # plt.plot(
        #     vx, vz, self.props["cvx0_color"], lw=0.5,
        # )
        plt.plot(
            vx[β<=β_c0], vz[β<=β_c0], self.props["cvx0_color"], lw=2,
        )
        plt.plot(
            vx[(β>=β_c0) & (β<=β_c1)], vz[(β>=β_c0) & (β<=β_c1)], 
            self.props["ccv_color"], lw=1.5,
        )
        plt.plot(
            vx[β>=β_c1], vz[β>=β_c1], self.props["cvx1_color"], lw=1.5,
        )
        axes = plt.gca()
        axes.set_aspect(1)
        plt.grid(ls=":")
        
        if model.type=="ISO":
            return
        
        # plt.plot(
        #     (0,vx_max),
        #     (0,vx_max*np.tan(α_mpx)), 
        #     ":",
        #     lw=1,
        #     color=self.props["βmpx_color"],
        #     label=(
        #         r"$\beta_{\mathrm{mpx}}$" 
        #         + rf"$={np.rad2deg(β_mpx):3.1f}\degree$, "
        #         + r"$\alpha_{\mathrm{mpx}}$"
        #         + rf"$={np.rad2deg(α_mpx):3.1f}\degree$"
        #     ),
        # )
        plt.plot(
            (0,vx_max),
            (0,vx_max*np.tan(α_rs0)), 
            "-",
            lw=3,
            alpha=0.2,
            color=self.props["βrs0_color"],
            label=(
                r"$\beta_{\mathrm{rs0}}$" 
                + rf"$={np.rad2deg(β_rs0):3.1f}\degree$, "
                + r"$\alpha_{\mathrm{rs0}}$"
                + rf"$={np.rad2deg(α_rs0):3.1f}\degree$"
            ),
        )
        plt.plot(
            (0,vx_max),
            (0,vx_max*np.tan(α_c0)),
            "--",
            lw=1,
            color=self.props["βc0_color"],
            label=(
                r"$\beta_{\mathrm{c0}}$" 
                + rf"$={np.rad2deg(β_c0):3.1f}\degree$, "
                + r"$\alpha_{\mathrm{c0}}$"
                + rf"$={np.rad2deg(α_c0):3.1f}\degree$"
            ),
        )
        plt.plot(
            (0,vx_max),
            (0,vx_max*np.tan(α_c1)), 
            "--",
            lw=1,
            color=self.props["βc1_color"],
            label=(
                r"$\beta_{\mathrm{c1}}$" 
                + rf"$={np.rad2deg(β_c1):3.1f}\degree$, "
                + r"$\alpha_{\mathrm{c1}}$"
                + rf"$={np.rad2deg(α_c1):3.1f}\degree$"
                # + r",  $\beta_{\mathrm{rs}}$"
                # + rf"$= {np.rad2deg(model.β_rs1):3.1f}\degree$"
            ),
        )
        plt.plot(
            (0,vx_max),
            (0,vx_max*np.tan(α_rs1)), 
            "-",
            lw=3,
            alpha=0.4,
            color=self.props["βrs1_color"],
            label=(
                r"$\beta_{\mathrm{rs1}}$" 
                + rf"$={np.rad2deg(β_rs1):3.1f}\degree$, "
                + r"$\alpha_{\mathrm{rs1}}$"
                + rf"$={np.rad2deg(α_rs1):3.1f}\degree$"
            ),
        )

        x_: float
        y_: float
        color_: str
        marker_: str
        label_: str
        x_fn: Callable = gma.dHdpx_onshell_interpfn
        y_fn: Callable = gma.dHdpz_onshell_interpfn
        for (x_, y_, color_, marker_, label_,)  in zip(
                (
                    x_fn(β_rs0),
                    x_fn(β_c0), 
                    x_fn(β_c1),
                    x_fn(β_rs1),
                    x_fn(β_mpx), 
                 ),
                (
                    y_fn(β_rs0),
                    y_fn(β_c0),
                    y_fn(β_c1),
                    y_fn(β_rs1),
                    y_fn(β_mpx),
                 ),
                (
                    self.props["βrs0_color"],
                    self.props["βc0_color"],
                    self.props["βc1_color"],
                    self.props["βrs1_color"],
                    self.props["βmpx_color"],
                 ),
                (
                    self.props["βrs0_marker"],
                    self.props["βc0_marker"],
                    self.props["βc1_marker"],
                    self.props["βrs1_marker"],
                    self.props["βmpx_marker"],
                 ),
                (
                    r"transition $\beta_\mathrm{rs0} = $"
                        +rf"{np.rad2deg(β_rs0):3.1f}$\degree$",
                    r"$\det{g_*}=0$  @  $\beta_{\mathrm{c0}} = $" 
                        + rf"{np.rad2deg(β_c0):3.1f}$\degree$",
                    r"$\det{g_*}=0$  @  $\beta_{\mathrm{c1}} = $"
                        + rf"{np.rad2deg(β_c1):3.1f}$\degree$",
                    r"transition $\beta_\mathrm{rs1} = $"
                        +rf"{np.rad2deg(β_rs1):3.1f}$\degree$",
                    r"max $p_x$ @ $\beta_\mathrm{mpx} = $"
                        +rf"{np.rad2deg(β_mpx):3.1f}$\degree$",
                ),
            ):
            # Switched to axes.plot from plt.scatter in hope we can overlay
            #   two plots and have separate legends for markers & lines
            plt.scatter(
                x_, 
                y_,
                marker=marker_,
                color=color_,
                s=50*2 if marker_=="*" else 25*2,
                alpha=0.75 if marker_=="*" else 1,
                label=label_,
                zorder=10,
            )

        plt.legend(loc=(1.03,0.3,), fontsize=9,)

    def plot_semicircle_rays(
            self,
            fig_name: str,
            model: Isotropic | Step | ExponentialActivation,
            gma: GeometricMechanicsAnalysis,
            x_max: float=2.5,
            title_font_size: int=12,
            fig_size: tuple[float, float]=(7, 5,),
        ) -> None:
        """
        Plot ray velocity indicatrix, plus convex/concave colors, critical βs.

        Parameters
        ----------
        fig_name: str,
        model: Any,
        title_font_size: int=12,
        fig_size: tuple[float, float]=(5, 5,),
        
        Attributes
        ----------
        Uses create_figure (adds to fig dict), props.
        """
        self.create_figure(fig_name, fig_size=fig_size,)
        plt.title(
            self.make_title(model, r"Semicircle propagation", gma,),
            fontdict={"fontsize": title_font_size,},
        )
        plt.xlabel(r"$x$  [-]")
        plt.ylabel(r"$z$  [-]")

        β: NDArray = gma.β
        vx: NDArray = gma.dHdpx_onshell
        vz: NDArray = gma.dHdpz_onshell

        β_c0: float = 0 if gma.β_c0 is None else gma.β_c0
        β_c1: float = 0 if gma.β_c1 is None else gma.β_c1
        β_rs1: float = 0 if gma.β_rs1 is None else gma.β_rs1
        α_c0: float = 0 if gma.α_c0 is None else gma.α_c0

        β_: NDArray
        for sf_ in np.linspace(0, 3, 31, endpoint=True,):
            β_ = β[β<β_c0]
            plt.plot(
                np.sin(β_) + vx[β<β_c0]*sf_, 
                1 -np.cos(β_) + vz[β<β_c0]*sf_, 
                # self.props["cvx0_color"], 
                color="k",
                lw=0.5,
            )
            β_ = β[(β>=β_c0) & (β<=β_c1)]
            plt.plot(
                np.sin(β_) + vx[(β>=β_c0) & (β<=β_c1)]*sf_, 
                1 -np.cos(β_) + vz[(β>=β_c0) & (β<=β_c1)]*sf_, 
                "-",
                color="r",
                # self.props["ccv_color"], 
                lw=1,
            )
            # β_ = β[β>β_c1]
            # plt.plot(
            #     np.sin(β_) + vx[β>β_c1]*sf_, 
            #     -np.cos(β_) + vz[β>β_c1]*sf_, 
            #     self.props["cvx1_color"], 
            #     lw=1,
            # )
        β_ = β[β<β_c0][-1]
        vx_ = vx[β<β_c0][-1]
        vz_ = vz[β<β_c0][-1]
        sf_ = np.linspace(0, 3, 101, endpoint=True,)
        plt.plot(
            np.sin(β_) + vx_*sf_, 
            1 -np.cos(β_) + vz_*sf_, 
            lw=3,
            alpha=0.2,
            color=self.props["βrs0_color"],
        )
        vx_fn: Callable = gma.dHdpx_onshell_interpfn
        vz_fn: Callable = gma.dHdpz_onshell_interpfn
        plt.plot(
            np.sin(β_c0) + vx_fn(β_c0)*sf_, 
            1 -np.cos(β_c0) + vz_fn(β_c0)*sf_, 
            "--",
            lw=1,
            color=self.props["βc0_color"],
            label=(
                r"$\beta_{\mathrm{c0}}$" 
                + rf"$={np.rad2deg(β_c0):3.1f}\degree$, "
                + r"$\alpha_{\mathrm{c0}}$"
                + rf"$={np.rad2deg(α_c0):3.1f}\degree$"
            ),
        )
        plt.plot(
            np.sin(β_rs1) + vx_fn(β_rs1)*sf_, 
            1 -np.cos(β_rs1) + vz_fn(β_rs1)*sf_, 
            lw=3,
            alpha=0.5,
            color=self.props["βrs1_color"],
            label=(
                r"$\beta_{\mathrm{rs1}}$" 
                + rf"$={np.rad2deg(β_rs1):3.1f}\degree$, "
                + r"$\alpha_{\mathrm{rs1}}$"
                + rf"$={np.rad2deg(β_rs1):3.1f}\degree$"
            ),
        )
        axes = plt.gca()
        axes.set_aspect(1)
        plt.grid(ls=":")
        plt.ylim(0, 1)
        plt.xlim(0, x_max,)
        plt.legend(fontsize=9, loc="upper left",)