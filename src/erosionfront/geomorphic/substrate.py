"""
Substrate models: single, double or multiple layered.
"""
import warnings
from enum import StrEnum, unique
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

warnings.filterwarnings("ignore")

__all__ = [
    "Substrates", 
    "OneLayerSubstrate",
    "TwoLayerSubstrate",
    "MultiLayerSubstrate",
    "choose_substrate",
]

# JSON parameters substrate.type must be one of these
@unique
class Substrates(StrEnum):
    """
    Substrate type enumeration.

    Attributes
    ----------
    L1: str = "single-layer"
        Uniform substrate.

    L2: str = "double-layer"
        Two-layer (horizontal) sandwich substrate model.

    LM: str = "multi-layer"
        Multilayer (horizontal) sandwich substrate model.
    """
    L1 = "single-layer"
    L2 = "double-layer"
    LM = "multi-layer"

# @unique
# class Substrates(Enum):
#     L1 = ("single-layer", "1L", OneLayerSubstrate,)
#     L2 = ("double-layer", "2L", TwoLayerSubstrate,)
#     LM = ("multi-layer", "ML", MultiLayerSubstrate,)

#     def __init__(self, type: str, label: str, subclass: Any,) -> None:
#         self.type: str = type
#         self.label: str = label
#         self.subclass: Any = subclass

@dataclass
class BaseSubstrate:
    """
    Container for bedrock substrate info.

    Parameters
    ----------
    η_upperlayer: float = 0.1
        Erodibility (ξ function scaling constant) of upper layer, assuming
        a two-layer sandwich substrate. Usually the upper layer is stronger 
        than the lower layer, so this number is <1 by default.
    η_hardtopbase: float = 1e-6
        Erodibility of thin upper cap and lower base layer, 
        usually set to be very low so as to prevent substantial erosion of the 
        mesa top and suppress spurious boundary effects at the base.
    nz_hardtop: int = 10
        Number of (vertical) pixels defining thickness of the cap layer.
    do_hardtop: bool = True
        Choose whether to implement the hard cap or not.
    do_hardbase: bool = True
        Choose whether to implement the hard base or not.

    Attributes
    ----------
    type: str = ""
        Substrate type (long-form).
        One of: `single-layer`, `double-layer`, `multi-layer`.
        Set by `choose_substrate`.
    label: str = ""
        Substrate type (short-form).
        Set by `choose_substrate`.
    η_upperlayer: float = 0.1
        Erodibility (ξ function scaling constant) of upper layer, assuming
        a two-layer sandwich substrate. Usually the upper layer is stronger 
        than the lower layer, so this number is <1 by default.
    η_hardtopbase: float = 1e-6
        Erodibility of thin upper cap and lower base layer, 
        usually set to be very low so as to prevent substantial erosion of the 
        mesa top and suppress spurious boundary effects at the base.
    nz_hardtop: int = 10
        Number of (vertical) pixels defining thickness of the cap layer.
    do_hardtop: bool = True
        Choose whether to implement the hard cap or not.
    do_hardbase: bool = True
        Choose whether to implement the hard base or not.
    """
    type: str = ""
    label: str = ""
    # η_ref: float | None = None
    # η_ratio: float | None = None
    η_upperlayer: float = 0.1
    η_hardtopbase: float = 1e-6
    nz_hardtop: int = 10
    do_hardtop: bool = True
    do_hardbase: bool = True

    def __post_init__(self) -> None:
        """
        Supply default values for unspecified parameters. Generate bounds etc.
        """
        if self.type not in Substrates._value2member_map_.keys():
            raise ValueError("Invalid substrate type")
        substrate_labels: dict = {
            Substrates.L1: "1L",
            Substrates.L2: "2L",
            Substrates.LM: "ML"
        }
        self.label = substrate_labels[self.type]

class OneLayerSubstrate(BaseSubstrate):
    """
    Single-layer, homogeneous substrate.

    Functionality entirely inherited from base class `BaseSubstrate`.
    """
    ...

class TwoLayerSubstrate(BaseSubstrate):
    """
    Two-layer (horizontal sandwich) substrate `BaseSubstrate`.

    Functionality entirely inherited from base class.
    """
    ...

class MultiLayerSubstrate(BaseSubstrate):
    """
    Multi-layer (horizontal sandwich) substrate.

    Functionality largely inherited from base class `BaseSubstrate`.

    Parameters
    ----------
    multilayer_n_levels: int = 3
        (Odd) number of sandwich layers.
    multilayer_dz: float | None = None
        Relative thickness of 'filling' sandwich layer(s) 
        (for normalized height).

    Attributes
    ----------
    multilayer_n_levels: int = 3
        (Odd) number of sandwich layers.
    multilayer_dz: float | None = None
        Relative thickness of 'filling' sandwich layer(s) 
        (for normalized height).
    multilayer_z_levels: NDArray
        Heights (z) of centerlines of layers. Just [0.5] for 3 layers.
    """
    multilayer_n_levels: int = 3
    multilayer_dz: float | None = None
    # multilayer_z_levels: NDArray | None = None
    def __post_init__(self) -> None:
        super().__post_init__()
        self.multilayer_dz = (
            1/self.multilayer_n_levels if self.multilayer_dz is None
            else self.multilayer_dz
        )
        self.multilayer_z_levels: NDArray \
            = np.arange(0, 1, 2/(self.multilayer_n_levels+1))[1:]
        # Selectively make more resistant some horizontal strips
        # if substrate.η_ratio is None:
        #     raise ValueError("Erodibility ratio not specified")
        # if substrate.do_twolayer:
        #     ξ0[z_change(0.5):,:] *= substrate.η_ratio
        # elif substrate.do_midlayer:
        #     ξ0[z_change(0.4):z_change(0.6),:] *= substrate.η_ratio
        # elif substrate.do_multilayer:
        #     if substrate.multilayer_z_levels is None:
        #         raise ValueError("Multilayer z levels not specified")
        #     z_levels = substrate.multilayer_z_levels
        #     if substrate.multilayer_dz is None:
        #         raise ValueError("Multilayer dz not specified")
        #     dz: float = substrate.multilayer_dz/2
        #     z_: NDArray
        #     for z_ in np.vstack((z_levels-dz, z_levels+dz,)).T:
        #         ξ0[z_change(z_[0]):z_change(z_[1]),:] \
        #             *= substrate.η_ratio
        # Scale everywhere by base η
        # ξ0 *= substrate.η_ref

def choose_substrate(parameters: dict,) -> (
          OneLayerSubstrate 
        | TwoLayerSubstrate
        | MultiLayerSubstrate
    ):
    """
    Choose substrate model.

    Parameters
    ----------
    dict: 
        Dictionary of keyword arguments to be passed to instantiate
        a substrate class. Choice of substrate type is specified in
        the `parameters['type']` element.

    Returns
    -------
    OneLayerSubstrate | TwoLayerSubstrate | MultiLayerSubstrate:
        An instance of these substrate classes. 
        Instantiation takes `**parameters` as keyword arguments.

    """
    substrates: dict = {
        Substrates.L1 : OneLayerSubstrate,
        Substrates.L2 : TwoLayerSubstrate,
        Substrates.LM : MultiLayerSubstrate
    }
    return substrates[parameters["type"]](**parameters)
