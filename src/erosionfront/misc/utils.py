"""
Utility functions.
"""
import warnings
import time
import json
from typing import Any
from io import TextIOWrapper

import numpy as np
from numpy.typing import NDArray
from sympy import Eq   #type: ignore
from sympy.parsing.sympy_parser import parse_expr #type: ignore

warnings.filterwarnings("ignore")

__all__ = ["load_parameters", "make_name", "Timer"]

def numify(str) -> float:
    """
    Cast to float after replacing decimal point letter 'p'.

    Parameters
    ----------
    str:
        Number as string with p used instead of decimal point.

    Returns
    -------
        Number as float.
    """
    return float(str.replace("p", "."))

def e2d(
    eqn_or_eqns: Eq | tuple[Eq] | list[Eq],
    do_flip: bool = False,
    do_negate: bool = False,
) -> dict[Any, Any]:
    """
    Convert a SymPy equation (or list of equations) into a dictionary item.

    Do this by mapping the LHS into a key and the RHS into a value.

    Parameters
    ----------
    eqn_or_eqns:  Eq | tuple[Eq] | list[Eq]
        Equation or equations to be converted.
    do_flip: bool = False
        Choose to reverse LHS-RHS.
    do_negate: bool = False
        Choose to negate both sides.

    Returns
    -------
    dict[Any, Any]: 
        Dictionary with key=LHS of eqn, value=RHS of eqn.
    """

    def negate_eqn(eqn_):
        return Eq(-eqn_.lhs, -eqn_.rhs) if do_negate else eqn_

    def flip_eqn(eqn_):
        return Eq(eqn_.rhs, eqn_.lhs) if do_flip else eqn_

    def make_dict(eqn_):
        return dict([(flip_eqn(negate_eqn(eqn_))).args])

    eqns = (
        eqn_or_eqns if isinstance(eqn_or_eqns, (list, tuple)) else [eqn_or_eqns]
    )
    eqn_dict: dict[Any, Any] = {}
    for eqn in eqns:
        eqn_dict.update(make_dict(eqn))
    return eqn_dict

def load_parameters(file_name: str,) -> dict:
    """
    Read and adapt parameters specified in a JSON file.

    Parameters
    ----------
    file_name: str
        JSON file name.

    Returns
    -------
    dict:
        Parameters dictionary.
    """
    file: TextIOWrapper
    with open(f"{file_name}.json", "r",) as file:
        parameters = json.load(file)

    # Scale and/or convert to Sympy where needed
    # They are deleted after conversion to make it easy to pass a whole
    # subdict as kwargs
    # if "σ_gradient_filter_raw" in parameters["grid"].keys():
    #     parameters["grid"]["σ_gradient_filter"] = (
    #         parameters["grid"]["σ_gradient_filter_raw"] 
    #         *(0.001/ parameters["grid"]["Δx"])
    #     )
    #     del(parameters["grid"]["σ_gradient_filter_raw"])
    parameters["misc"] = {}

    # if "type" in parameters["surface"].keys():
    #     parameters["misc"]["surface_type"] = (
    #         parameters["surface"]["type"] 
    #     )
    #     del(parameters["surface"]["type"])

    if "n_band_width_raw" in parameters["run"]:
        parameters["grid"]["band_width"] = (
            parameters["run"]["n_band_width_raw"] 
            * parameters["grid"]["Δx"]
        )
        parameters["misc"]["n_band_width_raw"] \
            = parameters["run"]["n_band_width_raw"]
        del(parameters["run"]["n_band_width_raw"])

    if "pad_width_raw" in parameters["grid"]:
        parameters["grid"]["n_pad_pixels"] = int(
            parameters["grid"]["pad_width_raw"]
            /parameters["grid"]["Δx"] + 1
        )
        parameters["misc"]["pad_width_raw"] \
            = parameters["grid"]["pad_width_raw"]
        del(parameters["grid"]["pad_width_raw"])

    if "β_ξt_πrelative" in parameters["model"]:
        β_ξt_πrelative: float = parameters["model"]["β_ξt_πrelative"]
        parameters["model"]["β_t"] \
            = float(parse_expr(f"pi*{β_ξt_πrelative}"))
        parameters["misc"]["β_ξt_πrelative"] \
            = parameters["model"]["β_ξt_πrelative"]
        del parameters["model"]["β_ξt_πrelative"]

    if "Δt_raw" in parameters["run"]:
        parameters["run"]["Δt"] \
            = parameters["run"]["Δt_raw"] * parameters["grid"]["Δx"]
        parameters["misc"]["Δt_raw"] = parameters["run"]["Δt_raw"]
        del(parameters["run"]["Δt_raw"])

    if "substrate" not in parameters:
        parameters["substrate"] = {}

    parameters["analysis"]["chpt_subset"] = (
        np.s_[100:1000:50] if "chpt_subset" not in parameters["analysis"]
        else np.s_[
            parameters["analysis"]["chpt_subset"][0]:
            parameters["analysis"]["chpt_subset"][1]:
            parameters["analysis"]["chpt_subset"][2]
        ]
    )
    
    if "analysis" in parameters:
        parameters["analysis"]["chpt_i_max"] = (
            1000 if "chpt_i_max" not in parameters["analysis"]
            else parameters["analysis"]["chpt_i_max"]
        )
        parameters["analysis"]["chpt_which"] = (
            0 if "chpt_which" not in parameters["analysis"]
            else parameters["analysis"]["chpt_which"]
        )
        parameters["analysis"]["chpt_n_filterwidth"] = (
            50 if "chpt_n_filterwidth" not in parameters["analysis"]
            else parameters["analysis"]["chpt_n_filterwidth"]
        )
        parameters["analysis"]["chpt_i_plot"] = (
            200 if "chpt_i_plot" not in parameters["analysis"]
            else parameters["analysis"]["chpt_i_plot"]
        )
        parameters["analysis"]["chpt_i_plot_offset"] = (
            50 if "chpt_i_plot_offset" not in parameters["analysis"]
            else parameters["analysis"]["chpt_i_plot_offset"]
        )

    if "viz" in parameters:
        parameters["viz"]["pc_slicing"] = (
            10 if "pc_slicing" not in parameters["viz"]
            else parameters["viz"]["pc_slicing"]
        )
        parameters["viz"]["pc_labeling"] = float(
            100/10 if "pc_labeling" not in parameters["viz"]
            else 100/parameters["viz"]["pc_labeling"]
        )

    # Convert to None values where needed
    for group_ in parameters:
        for var_ in parameters[group_]:
            if parameters[group_][var_]=="None":
                parameters[group_][var_] = None

    return parameters

def make_name(job_name: str, parts: list | str,) -> list | str:
    """
    Generate a figure name from job name and other info.

    Parameters
    ----------
    job_name: str
        Simulation (job) name.
    parts: list | str
        Extra info specifying figure type.
    Returns
    -------
    list | str:
        Figure name.
    """
    if type(parts) is list:
        return "_".join([job_name]+list(parts))
    else:
        return "_".join([job_name, str(parts)])
    
class Timer:
    """
    Timer to help gauge how long parts of the simulation take.

    Class methods used only.

    Parameters
    ----------
    do_use: bool = False
        Switch to enable/disable the timer.

    Attributes
    ----------
    t: float
        Time elapsed after being switched on.        
    """
    t: float = 0
    do_use: bool = False

    @classmethod
    def __init__(
        cls, do_use: 
        bool=False, 
        announcement: str | None=None,
    ) -> None:
        cls.do_use = do_use
        if do_use and announcement is not None:
            print(f"\n{announcement}\n")

    @classmethod
    def start(cls, info: str | None=None,) -> None:
        if cls.do_use:
            print(info)
            cls.t = time.perf_counter()
    
    @classmethod
    def stop(cls) -> None:
        if cls.do_use:
            Δt: float = time.perf_counter()-cls.t
            report: str = (
                f"t = {(Δt)*1000:0.3f}ms" if Δt<1
                else f"t = {(Δt):0.3f}s"
            )
            print(report)

def is_serializable(value: Any,) -> bool:
    """
    Check value is serializable and can be written to a JSON file.

    Parameters
    ----------
    value: Any
        To be checked.

    Returns
    -------
    bool:
        True if value is serializable.
    """
    try:
        json.dumps(value)
        return True
    except:
        return False

def make_serializable(value: Any,) -> Any:
    """
    Convert value into serializable version.

    Parameters
    ----------
    value: Any
        To be converted.

    Returns
    -------
    Any:
        Serializable value.
    """
    if type(value)==np.float64 or type(value)==float:
        return float(value)
    elif type(value)==int:
        return int(value)
    elif type(value)==str or type(value)==tuple or type(value)==list:
        return value
    elif type(value)==np.ndarray:
        return value.tolist()
    
