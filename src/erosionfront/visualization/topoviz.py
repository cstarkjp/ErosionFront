"""
Visualization of 3D topo data.
"""
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
pv.global_theme.anti_aliasing = "msaa"
pv.global_theme.multi_samples = 1

from pyvista import MultiBlock, Plotter
from scipy.signal import medfilt

from erosionfront.visualization.base import GraphingBase
from erosionfront.topo.analysis import (
    Geometry,
    Profiles,
    ProfileData,
)
warnings.filterwarnings("ignore")

__all__ = ["TopoViz"]

class TopoViz(GraphingBase):
    """
    Visualization of 3D topo data.
    """

    def plot_3d(
        self,
        mesh: MultiBlock, 
        geometry: Geometry,
        profiles: Profiles,
    ) -> Plotter:
        """
        Display (possibly) interactive 3D view using PyVista

        Parameters
        ----------
        mesh: MultiBlock
            X
        geometry: Geometry
            X
        profiles: Profiles
            X

        Returns
        -------
        Plotter:
            An instance of a PyVista 3D viewer.
        """
        pl = Plotter(window_size=(1024, 768,),) #, notebook=True
        grid: tuple[dict[float, MultiBlock]] = geometry.grid
        grid_colors: tuple[str,str] = ("k","k",) # ("coral","limegreen",)
        for lines_, color_ in zip(grid, grid_colors,):
            for step_, line_ in lines_.items():
                pl.add_mesh(
                    line_, 
                    show_edges=False, 
                    color=color_, 
                    line_width=2 if step_==0 else 1, 
                    render_lines_as_tubes=False,
                )
        # Profile slices
        profiles: dict[Any: ProfileData] = profiles.data
        colors = list(mcolors.TABLEAU_COLORS.values())
        for color_, profile_ in zip(colors,profiles.values()):
            slice_ = profile_.slice
            pl.add_mesh(
                slice_, show_edges=False, color=color_, 
                line_width=5, render_lines_as_tubes=True,
            )
        # 3D DEM
        pl.add_mesh(mesh, show_edges=False, color="lightgray",)
        pl.show_axes()
        return pl

    def plot_profiles(
        self,
        fig_name: str,
        profiles: Profiles,
        choices: tuple[str,...] | None,
        marker_size: float | None,
        line_width: float | None,
        xlim: tuple[float,float] | None = None,
        ylim: tuple[float,float] | None = None,
        scale: float=1,
        do_equal: bool=True,
        # use_normals: bool=False,
        do_normals: bool=False,
        vector_sf: float=1,
        # do_flip_orientation: bool=False,
        θ_clip: tuple[float,float]=(0,90,),
        title_font_size: int=12,
        fig_size: tuple[float, float]=(7, 5,),
    ) -> None:
        """
        X

        Parameters
        ----------
        fig_name: str
            Name used as key in figure dictionary.
        profiles: dict[Any: Profile]
            X
        choices: tuple[str,...] | None
            X
        marker_size: float | None
            X
        line_width: float | None
            X
        xlim: tuple[float,float] | None = None
            X
        ylim: tuple[float,float] | None = None
            X
        scale: float=1
            X
        do_equal: bool=True
            X
        # use_normals: bool=False
            X
        do_normals: bool=False
            X
        vector_sf: float=1
            X
        # do_flip_orientation: bool=False
            X
        θ_clip: tuple[float,float]=(0,90,)
            X
        title_font_size: int=12
            Size of title text.
        fig_size: tuple[float, float]=(7, 5,)
            Figure size (in inches!).

        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=np.array(fig_size)*scale,)
        # title: str = ""
        # plt.title(
        #     title,
        #     fontdict={"fontsize": title_font_size,},
        # )
        profiles: dict[Any: ProfileData] = profiles.data
        colors = list(mcolors.TABLEAU_COLORS.values())
        for j_, profile_ in enumerate(profiles.values()):
            if choices is not None and profile_.name not in choices:
                continue
            chords_ = profile_.chords
            ψ_deg = profile_.orientation
            mid_points_ = profile_.mid_points
            parallels_ = profile_.parallels
            normals_ = profile_.normals
            tilts_ = profile_.tilts
            tilts_r_ = profile_.tilts_r
            for (i_, 
                 (chord_, mid_point_, parallel_, normal_, tilt_, tilt_r_,),) \
                in enumerate(zip(
                    chords_, mid_points_,parallels_,normals_,tilts_,tilts_r_
                ),):
                # Don't bother plotting chords whose endpts lie beyond the 
                # limits
                if xlim is not None:
                    if chord_[0][0]<xlim[0] or chord_[0][1]>xlim[1]:
                        continue
                plt.plot(
                    *chord_, 
                    ("." if marker_size is not None else "") 
                        + ("-" if line_width is not None else ""),
                    color=colors[j_] if not do_normals else "Gray",
                    ms=marker_size,
                    lw=line_width,
                    label=rf"$\psi=${ψ_deg}$\degree$" if i_==0 else None,
                )
                if do_normals:
                    normal_ #if use_normals else parallel_
                    is_in_bounds = lambda tilt: \
                        ((tilt>=θ_clip[0] and tilt<=θ_clip[1]) 
                            or (-tilt>=θ_clip[0] and -tilt<=θ_clip[1])) 
                    if tilt_ is not None and is_in_bounds(tilt_):
                        plt.plot(
                            (mid_point_[0], 
                             mid_point_[0]+normal_[0]*vector_sf,), 
                            (mid_point_[1], 
                             mid_point_[1]+normal_[1]*vector_sf,), 
                            "-",
                            color="k" if tilt_>0 else "b",
                            lw=0.5,
                        )
                    elif tilt_r_ is not None and is_in_bounds(tilt_r_):
                        plt.plot(
                            (mid_point_[0], 
                             mid_point_[0]+normal_[0]*vector_sf,), 
                            (mid_point_[1], 
                             mid_point_[1]+normal_[1]*vector_sf,), 
                            "-",
                            color="g" if tilt_r_>0 else "r",
                            lw=0.5,
                        )
        plt.xlabel("Distance along profile  [m]")
        plt.ylabel(r"Elevation $z$  [m]")
        plt.grid(ls=":",)
        plt.legend()
        axes = plt.gca()
        if do_equal:
            axes.set_aspect("equal")
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)

    def plot_slopes(
        self,
        fig_name: str,
        profiles: Profiles,
        choices: tuple[str,...] | None,
        w_filter: int=15,
        scale: float=1,
        line_width: float=1,
        do_points: bool=False,
        title_font_size: int=12,
        fig_size: tuple[float,float]=(6,3.5,),
    ) -> None:
        """
        X

        Parameters
        ----------
        fig_name: str
            Name used as key in figure dictionary.
        profiles: Profiles
            X
        choices: tuple[str,...] | None
            X
        w_filter: int=15
            X
        scale: float=1
            X
        line_width: float=1
            X
        do_points: bool=False
            X
        title_font_size: int=12
            Size of title text.
        fig_size: tuple[float, float]=(7, 5,)
            Figure size (in inches!).

        Attributes
        ----------
        Uses create_figure (adds to fig dict).
        """
        self.create_figure(fig_name, fig_size=np.array(fig_size)*scale,)
        # title: str = ""
        # plt.title(
        #     title,
        #     fontdict={"fontsize": title_font_size,},
        # )
        profiles: dict[Any: ProfileData] = profiles.data
        colors = list(mcolors.TABLEAU_COLORS.values())
        for j_, profile_ in enumerate(profiles.values()):
            if choices is not None and profile_.name not in choices:
                continue
            xg_ = np.array([
                ( 
                    (chord_[0][1]+chord_[0][0])/2, 
                    np.rad2deg(np.atan2( 
                        (chord_[1][1]-chord_[1][0]), 
                        (chord_[0][1]-chord_[0][0]) 
                    ))
                )
                for chord_ in profile_.chords   
            ])
            xgs_ = np.array(sorted(xg_, key=lambda x: x[0]))
            xgf_ = medfilt(xgs_[:,1], w_filter,)
            ψ_deg = profile_.orientation
            plt.plot(
                xgs_[:,0], xgf_, 
                "." if do_points else "-", 
                ms=3, 
                lw=line_width,
                color=colors[j_],
                label=rf"$\psi=${ψ_deg}$\degree$",
            )
        plt.xlabel("Distance along profile  [m]")
        plt.ylabel(r"Slope angle  [$\degree$]")
        axes = plt.gca()
        # axes.set_aspect("equal")
        plt.yticks(np.arange(-90,+90+15,15))
        plt.grid(ls=":",)
        plt.legend()