"""
Shock detection mixin leveraging `ruptures` package changepoint methods.
"""
import warnings
from enum import StrEnum, unique

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
import ruptures  #type: ignore

warnings.filterwarnings("ignore")

__all__ = [
    "ChangepointMethods", 
    "ShocksMixin", 
    "choose_changepoint_method"
]

@unique
class ChangepointMethods(StrEnum):
    """`ruptures` package changepoint method choices."""
    PELT = "Pelt"
    WINDOW = "Window"
    BINSEG = "Binseg"

def choose_changepoint_method(method: str="Binseg",) -> ChangepointMethods:
    """
    `ruptures` package changepoint method selector.

    Parameters
    ----------
    method: str="Binseg"
        One of "Pelt", "Window", "Binseg".

    Returns
    -------
    ChangepointMethods:
        One of the enumerated set of methods: `PELT`, `WINDOW`, `BINSEG`.
    """
    match method:
        case ChangepointMethods.PELT:
            return ChangepointMethods.PELT
        case ChangepointMethods.WINDOW:
            return ChangepointMethods.WINDOW
        case _: #ChangepointMethods.BINSEG:
            return ChangepointMethods.BINSEG
        # case _:
        #     raise ValueError("Unknown changepoint method")

class ShocksMixin:
    """
    Mixin providing shock (break-in-slope) detection methods.

    Parameters
    ----------
    T_slices: dict
        All the time-sliced surfaces (x,z coordinate arrays).
    """
    T_slices: dict

    def find_shock(
            self, 
            i_step: int, 
            method: ChangepointMethods=ChangepointMethods.BINSEG,
            i_max: int=300,
            n_filterwidth: int=50,
            which_changepoint: int=1,
            do_plot: bool=False, 
        ) -> tuple[float, float]:
        """
        Locate main break-in-slope by smoothing & changepoint detection.

        Parameters
        ----------
        i_step: int
            
        method: ChangepointMethods=ChangepointMethods.BINSEG
            
        i_max: int=300
            
        n_filterwidth: int=50
            
        which_changepoint: int=1
            
        do_plot: bool=False

        Returns
        -------
        tuple[float, float]:
            x, z coordinate of break-in-slope shock changepoint.
        """
        changepoint_fn: ruptures.Pelt | ruptures.Window | ruptures.Binseg
        changepoints: list
        β_max: float = np.pi/2 * 0.95
        β_: NDArray = median_filter(self.T_slices[i_step]["β"], n_filterwidth,)
        i_β_cliff: NDArray = np.where(β_>β_max)[0]
        i_end: int = max(
            i_β_cliff[0] if i_β_cliff.shape[0]>1
            else len(β_), i_max
        )
        β_ = β_[:i_end]
        match method:
            case ChangepointMethods.PELT: 
                changepoint_fn = ruptures.Pelt(model="l2", min_size=30,)
                changepoint_fn.fit(β_)
                changepoints = changepoint_fn.predict(pen=3,)
            case ChangepointMethods.WINDOW:
                changepoint_fn = ruptures.Window(model="l2", width=10,)
                changepoint_fn.fit(β_)
                changepoints = changepoint_fn.predict(n_bkps=2,)
            case ChangepointMethods.BINSEG:
                changepoint_fn = ruptures.Binseg(model="l2", min_size=10,)
                changepoint_fn.fit(β_)
                changepoints = changepoint_fn.predict(n_bkps=2,)
        i_shock: int = changepoints[which_changepoint]
        if do_plot:
            ruptures.display(β_, [], changepoints)
        return (
            self.T_slices[i_step]["x"][i_shock], 
            self.T_slices[i_step]["z"][i_shock],
        )
    
    def find_shocks(
            self,
            method: ChangepointMethods=ChangepointMethods.BINSEG,
            subset: slice=np.s_[100:1000:50],
            i_max: int=1000,
            n_filterwidth: int=50,
            which_changepoint=0,
            i_plot: int=200,
            i_plot_offset: int=50,
        ) -> None:
        """
        Find breaks-in-slope in the set of T-sliced surfaces.

        Parameters
        ----------
        method: ChangepointMethods=ChangepointMethods.BINSEG

        subset: slice=np.s_[100:1000:50]
            
        i_max: int=1000
            
        n_filterwidth: int=50
            
        which_changepoint=0,
            
        i_plot: int=200

        i_plot_offset: int=50

        Attributes
        ----------
        Modifies `shocks_xz` with estimated locations of breaks-in-slope.
        """
        slices: list = list(self.T_slices.keys())[subset]
        self.shocks_xz: NDArray = np.array([
            self.find_shock(
                i_step_, 
                method=method,
                i_max=i_max,
                n_filterwidth=n_filterwidth,
                which_changepoint=which_changepoint,
                do_plot=(
                    (i_step_==slices[-1]) | (i_step_-i_plot_offset)%i_plot==0
                ), 
            )
            for i_step_ in slices
        ]).T