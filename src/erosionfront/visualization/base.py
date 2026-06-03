"""
Provide a data visualization class.
"""
import warnings
import logging
import operator as op
from itertools import cycle
from typing import Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")

__all__ = ["GraphingBase"]

class GraphingBase:
    """
    Provide a visualization base class.

    Parameters
    ----------
    dpi:
        resolution for rasterized images
    font_size:
        general font size

    Attributes
    ----------
    dpi (int):
        resolution for rasterized images
    font_size (int):
        general font size
    fdict  (dict):
        dictionary to which each figure is appended as it is generated
    colors  (list):
        list of colors
    n_colors  (int):
        number of colors
    color_cycle  (:obj:`itertools cycle <itertools.cycle>`):
        color property cycle
    markers  (list):
        list of markers
    n_markers  (:obj:`itertools cycle <itertools.cycle>`):
        number of markers
    marker_cycle  (int):
        cycle of markers
    linestyle_list  (list):
        list of line styles (solid, dashdot, dashed, custom dashed)
    color (:obj:`lambda(i) <lambda>`):
        return i^th color
    marker (:obj:`lambda(i) <lambda>`):
        return i^th marker
    """

    dpi: int
    font_size: int
    fdict: dict
    colors: Callable
    n_colors: int
    color_cycle: Callable
    markers: tuple
    n_markers: int
    marker_cycle: cycle
    linestyle_list: tuple
    color: Callable
    marker: Callable
    font_family: str

    props: dict
    make_title: Callable

    def __init__(self, dpi: int = 100, font_size: int = 11) -> None:
        """Initialize."""
        self.dpi = dpi
        self.font_size = font_size
        self.fdict: dict[Any, Any] = {}
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        self.colors = prop_cycle.by_key()["color"]  # type: ignore
        self.n_colors = len(self.colors)  # type: ignore
        self.color_cycle = cycle(self.colors)  # type: ignore
        self.markers = ("o", "s", "v", "p", "*", "D", "X", "^", "h", "P")
        self.n_markers = len(self.markers)
        self.marker_cycle = cycle(self.markers)
        self.linestyle_list = ("solid", "dashdot", "dashed", (0, (3, 1, 1, 1)))

        color_ = lambda i_: self.colors[i_ % self.n_colors]  # type: ignore
        marker_ = lambda i_: self.markers[i_ % self.n_markers]  # type: ignore
        self.color = color_  # type: ignore
        self.marker = marker_  # type: ignore
        #if "Arial" in self.get_fonts() else "Helvetica"
        self.font_family = "Arial" 
        try:
            mpl.rc("font", size=self.font_size, family=self.font_family)
        except:
            mpl.rc("font", size=self.font_size, family="")

    def get_fonts(self) -> list[str]:
        """Fetch the names of all the font families available on the system."""
        fpaths = matplotlib.font_manager.findSystemFonts()
        fonts: list[str] = []
        for fpath in fpaths:
            try:
                font = matplotlib.font_manager.get_font(fpath).family_name
                fonts.append(font)
            except RuntimeError as re:
                logging.debug(f"{re}: failed to get font name for {fpath}")
                pass
        return fonts

    def create_figure(
        self,
        fig_name: str,
        fig_size: tuple[float, float] | None = None,
        dpi: int | None = None,
    ) -> Figure:
        """
        Initialize a :mod:`Pyplot <matplotlib.pyplot>` figure.

        Set its size and dpi, set the font size,
        choose the Arial font family if possible,
        and append it to the figures dictionary.

        Parameters
        ----------
        fig_name:
            name of figure; used as key in figures dictionary
        fig_size:
            optional width and height of figure in inches
        dpi:
            rasterization resolution

        Returns
        -------
        :obj:`Pyplot figure <matplotlib.figure.Figure>`:
            reference to :mod:`MatPlotLib/Pyplot <matplotlib.pyplot>`
            figure
        """
        fig_size_: tuple[float, float] = (
            (8, 8) if fig_size is None else fig_size
        )
        dpi_: float = self.dpi if dpi is None else dpi
        logging.info(
            "gmplib.plot.GraphingBase:\n   "
            + f"Creating plot: {fig_name} size={fig_size_} @ {dpi_} dpi"
        )
        fig = plt.figure()
        self.fdict.update({fig_name: fig})
        fig.set_size_inches(fig_size_)
        fig.set_dpi(dpi_)
        return fig

    def get_aspect(self, axes: plt.Axes) -> float:
        """
        Get aspect ratio of graph.

        Parameters
        ----------
        axes:
            the `axes` object of the figure

        Returns
        -------
        float:
            aspect ratio
        """
        # Total figure size
        figWH: tuple[float, float] \
            = axes.get_figure().get_size_inches() #type: ignore
        figW, figH = figWH
        # Axis size on figure
        bounds: tuple[float, float, float, float] = axes.get_position().bounds
        _, _, w, h = bounds
        # Ratio of display units
        disp_ratio: float = (figH * h) / (figW * w)
        # Ratio of data units
        # Negative over negative because of the order of subtraction
        # logging.info(axes.get_ylim(),axes.get_xlim())
        data_ratio: float = op.sub(*axes.get_ylim()) / op.sub(*axes.get_xlim())
        aspect_ratio: float = disp_ratio / data_ratio
        return aspect_ratio

    def naturalize(self, fig: Figure) -> None:
        """Adjust graph aspect ratio into 'natural' ratio."""
        axes: plt.Axes = fig.gca()
        # x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
        # axes.set_aspect((y_lim[1]-y_lim[0])/(x_lim[1]-x_lim[0]))
        axes.set_aspect(1 / self.get_aspect(axes))

    def stretch(
        self,
        fig: Figure,
        xs: tuple[float, float] | None = None,
        ys: tuple[float, float] | None = None,
    ) -> None:
        """Stretch graph axes by respective factors."""
        axes: plt.Axes = fig.gca()
        if xs is not None:
            x_lim = axes.get_xlim()
            x_range = x_lim[1] - x_lim[0]
            axes.set_xlim(
                x_lim[0] - x_range * xs[0], x_lim[1] + x_range * xs[1]
            )
        if ys is not None:
            y_lim = axes.get_ylim()
            y_range = y_lim[1] - y_lim[0]
            axes.set_ylim(
                y_lim[0] - y_range * ys[0], y_lim[1] + y_range * ys[1]
            )