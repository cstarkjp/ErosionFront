"""
Surface geometry classes for use in erosion front propagation.
"""
import warnings
from dataclasses import dataclass
from enum import StrEnum, unique

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

warnings.filterwarnings("ignore")

__all__ = [
    "Surfaces",
    "StraightLineSurface",
    "ErrorFunctionSurface",
    "SemiCircularArcSurface",
    "QuarterCircularArcSurface",
    "choose_surface"
]

# JSON parameters surface.type must be one of these
@unique
class Surfaces(StrEnum):
    """
    Surface-type enumeration.

    Attributes
    ----------
    SL: str = "straight-line"
        Segmented flat-ramp-flat straight-line model.
    ERF: str = "error-function"
        Smooth-step model computed using an error function.
    SCA: str = "semicircular-arc"
        180º half-circle arc, i.e., with 50% overhang.
    SCA: str = "quartercircular-arc"
        90º quarter-circle arc.
    """
    SL = "straight-line"
    ERF = "error-function"
    SCA = "semicircular-arc"
    QCA = "quartercircular-arc"

@dataclass
class SurfaceMethods:
    """
    Base class providing geometric tools for surface model classes.

    Parameters
    ----------
    n_pts: int = 1001
        Number of sample points along the surface. 
    height: float = 1
        Maximum height of surface function above zero base-level.
    steepness: float = 10
        Steepness factor used in `straight-line` and `error-function` surfaces.
    x_left: float = -1
        Left-most x coordinate for sampling
    x_right: float = +1
        Right-most x coordinate for sampling.
    x_center: float = 0
        Central x coordinate in `error-function` surface (point of symmetry).
    do_anchor_ends: bool = True
        Fix the left and right end z-values at 0 and `height` respectively.

    Attributes
    ----------
    type: str = ""
        Surface type (long-form).
        One of: `straight-line`, `error-function`, `semicircular-arc`, \
            `quartercircular-arc`.
        Set by `choose_surface`.
    label: str = ""
        Surface type (short-form).
        Set by `choose_surface`.
    n_pts: int = 1001
        Number of sample points along the surface. 
    height: float = 1
        Maximum height of surface function above zero base-level.
    steepness: float = 10
        Steepness factor used in `straight-line` and `error-function` surfaces.
    x_left: float = -1
        Left-most x coordinate for sampling
    x_right: float = +1
        Right-most x coordinate for sampling.
    x_center: float = 0
        Central x coordinate in `error-function` surface (point of symmetry).
    do_anchor_ends: bool = True
        Fix the left and right end z-values at 0 and `height` respectively.
    x: NDArray | None = None
        Sample x coordinates along the surface.
    z: NDArray | None = None
        Sample z coordinates along the surface.
    """
    type: str = ""
    label: str = ""
    n_pts: int = 1001
    height: float = 1
    steepness: float = 10
    x_left: float = -1
    x_right: float = +1
    x_center: float = 0
    do_anchor_ends: bool = True
    x: NDArray | None = None
    z: NDArray | None = None

    def get_points_xz(self) -> NDArray:
        """
        Stack x,z arrays of N points into a single (N,2) array.

        Attributes
        ----------
        Uses x and z array attributes.

        Returns
        -------
        NDArray:
            (x,z) point vector array with shape (N,2).
        """
        return np.vstack([self.x, self.z]).T   #type: ignore
    
    def __post_init__(self) -> None:
        """
        Complete instantiation by computing surface angle β along the curve.

        Attributes
        ----------
        β: NDArray
            Angle of surface computed by taking arctan of gradient of x and z
            coordinates wrt index.
        """
        if self.do_anchor_ends:
            if self.z is not None:
                self.z[0] = min(0, self.z[0])
                self.z[-1] = max(self.height, self.z[-1])
            else:
                raise ValueError("z array not assigned yet")
        if self.type not in Surfaces._value2member_map_.keys():
            raise ValueError("Invalid surface type")
        surface_types: dict = {
            Surfaces.SL : "SL",
            Surfaces.ERF : "ERF",
            Surfaces.SCA : "SCA",
            Surfaces.QCA : "QCA",
        }
        self.label = surface_types[self.type]
        self.β: NDArray \
            = np.arctan2(np.gradient(self.z), np.gradient(self.x)) #type: ignore

@dataclass
class StraightLineSurface(SurfaceMethods):
    """
    Synthesize a very simple straight line surface.

    Attributes
    ----------
    Modifies x,z atributes.
    See ``Methods`` parent class for more details.
    """ 
    def __post_init__(self) -> None:
        """
        Generate x and z arrays for the surface.        

        Attributes
        ----------
        Uses model parameter attributes and modifies x,z array attributes.
        """
        self.x = np.linspace(self.x_left, self.x_right, self.n_pts)
        x_offset: float = -0.5/self.steepness
        self.z = (self.x-x_offset)*self.steepness * self.height
        self.z[self.z>self.height] = self.height
        self.z[self.z<0] = 0
        super().__post_init__()

@dataclass
class ErrorFunctionSurface(SurfaceMethods):
    """
    Synthesize a smoothed-step surface profile using an error function.

    Parameters
    ----------
    gradient: float = 0
    do_anchor_ends: bool = True

    Attributes
    ----------
    Set by parameters.
    """
    trend: float = 0

    @staticmethod
    def shape_fn(
            x: float | NDArray, 
            x0: float, 
            s: float, 
            t: float,
        ) -> float | NDArray:
        """
        Compute z(x) value(s) for a error-function surface profile.

        Parameters
        ----------
        x: float | NDArray
        x0: float
        s: float
        t: float

        Returns
        -------
        float | NDArray:
            Array (or single value) for z(x) for erf-type step function.
        """
        return ((1 + erf((x-x0)*s))/2 + x*t)

    def __post_init__(self) -> None:
        """
        Generate x and z arrays for the model surface.        

        Attributes
        ----------
        Uses model parameter attributes and modifies x,z array attributes.
        """
        self.x = np.linspace(self.x_left, self.x_right, self.n_pts,)
        self.z = (
            self.height * np.array(
                self.shape_fn(
                    self.x, self.x_center, self.steepness, self.trend
                )
            )
        )
        super().__post_init__()

@dataclass
class SemiCircularArcSurface(SurfaceMethods):
    """
    Synthesize a circular arc surface.

    Attributes
    ----------
    Modifies x,z atributes.
    See ``Methods`` parent class for more details.
    """    
    def __post_init__(self) -> None:
        """
        Generate x and z arrays for the surface.

        Attributes
        ----------
        Uses model parameter attributes and modifies x,z array attributes.
        """
        radius: float = self.height/2
        n_pts_left: int = int(
            (0-self.x_left)/(self.x_right-self.x_left+np.pi*radius) 
            * self.n_pts
        )
        n_pts_arc: int = int(
            np.pi*radius/(self.x_right-self.x_left+np.pi*radius) 
            * self.n_pts
        )
        n_pts_right: int = int(
            (self.x_right-0)/(self.x_right-self.x_left+np.pi*radius) 
            * self.n_pts
        )
        θ: NDArray = np.linspace(0, np.pi, n_pts_arc,)
        self.x = np.hstack([
            np.linspace(self.x_left, 0, n_pts_left,),
            np.sin(θ)*radius,
            np.linspace(0, self.x_right, n_pts_right)
        ])
        self.z = np.hstack([
            np.zeros(n_pts_left,),
            (1-np.cos(θ))*radius,
            np.ones(n_pts_right)*radius*2
        ])
        super().__post_init__()


@dataclass
class QuarterCircularArcSurface(SurfaceMethods):
    """
    Synthesize a circular arc surface.

    Attributes
    ----------
    Modifies x,z atributes.
    See ``Methods`` parent class for more details.
    """    
    def __post_init__(self) -> None:
        """
        Generate x and z arrays for the surface.

        Attributes
        ----------
        Uses model parameter attributes and modifies x,z array attributes.
        """
        n_pts_left: int = int(
            (0-self.x_left)/(self.x_right-self.x_left) * self.n_pts
        )
        n_pts_arc: int = int(
            self.height/(self.x_right-self.x_left) * self.n_pts
        )
        n_pts_right: int = int(
            (self.x_right-self.height)/(self.x_right-self.x_left) * self.n_pts
        )
        θ: NDArray = np.linspace(0, np.pi/2, n_pts_arc,)
        self.x = np.hstack([
            np.linspace(self.x_left, 0, n_pts_left,),
            np.sin(θ)* self.height,
            np.linspace(self.height, self.x_right, n_pts_right)
        ])
        self.z = np.hstack([
            np.zeros(n_pts_left,),
            (1-np.cos(θ))*self.height,
            np.ones(n_pts_right)*self.height
        ])
        super().__post_init__()

def choose_surface(parameters: dict,) -> (
          StraightLineSurface
        | ErrorFunctionSurface
        | SemiCircularArcSurface
        | QuarterCircularArcSurface
    ):
    """
    Choose initial surface function.

    Parameters
    ----------
    dict: 
        Dictionary of keyword arguments to be passed to instantiate
        a surface class. Choice of surface type is specified in
        the `parameters['type']` element.

    Returns
    -------
    StraightLineSurface | ErrorFunctionSurface | SemiCircularArcSurface \
        | QuarterCircularArcSurface:
        One of these surface function classes.
        Instantiation takes `**parameters` as keyword arguments.
    """
    surfaces: dict = {
        Surfaces.SL : StraightLineSurface,
        Surfaces.ERF : ErrorFunctionSurface,
        Surfaces.SCA : SemiCircularArcSurface,
        Surfaces.QCA : QuarterCircularArcSurface,
    }
    return surfaces[parameters["type"]](**parameters)
