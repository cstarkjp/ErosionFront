"""
Erosion speed ξ(β) models.

Currently only `isotropic` and `exponential-activation` models are fully 
implemented with the necessary HJE level-set functions. The `step` model still
lacks the "curvature" H terms needed for Lax-Friedrichs.
"""
import warnings
from dataclasses import dataclass
from enum import StrEnum, unique
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from numpy.ma import MaskedArray
from scipy.special import erf

warnings.filterwarnings("ignore")

__all__ = [
    "Models",
    "Isotropic",
    "Step",
    "ExponentialActivation",
    "choose_model",
]

# JSON parameters model.type must be one of these
@unique
class Models(StrEnum):
    """
    Model type enumeration.

    Attributes
    ----------

    ISO: str = "isotropic"
        Isotropic eikonal model.

    STEP: str = "step"
        Simple error-function smooth-step model.

    EA: str = "exponential-activation"
        Smooth-step model based on exp(-β_t/β) shape.
    """
    ISO = "isotropic"
    STEP = "step"
    EA = "exponential-activation"

@dataclass
class BaseStepModel(ABC):
    """
    Base class for step-type model erosion speed functions.

    Subclassed models are all explicitly defined, rather than derived
    from Sympy theoretical development.

    Parameters
    ----------
    ξ_min: float = 0.2
        Asymptotic minimum ξ for β → 0.
    ξ_max: float = 1
        Asymptotic minimum ξ for β → ∞.
    ξ_trend: float = 0.05
        Overall "tilt" to ξ model as β rises.
    β_t: float = np.pi/4
        Transition slope angle β where ξ switches from low to high.
    σ_ξ: float = 0.1
        Smoothness (in Gaussian std dev sense w β) of ξ transition.
    n_ξ: float = 2
        Order of transition smoothness (power exponent; alt to σ_ξ).
    Δt_slabfailure: float = 0.05
        Time between periodic removal of overhangs.
    do_slabfailure: bool = True
        Do periodic removal of overhangs.

    
    Attributes
    ----------
    type: str = ""
        Model name (long form).
    label: str = ""
        Model name (short form).
    ξ_min: float = 0.2
        Asymptotic minimum ξ for β → 0.
    ξ_max: float = 1
        Asymptotic minimum ξ for β → ∞.
    ξ_trend: float = 0.05
        Overall "tilt" to ξ model as β rises.
    β_t: float = np.pi/4
        Transition slope angle β where ξ switches from low to high.
    σ_ξ: float = 0.1
        Smoothness (in Gaussian std dev sense w β) of ξ transition.
    n_ξ: float = 2
        Order of transition smoothness (power exponent; alt to σ_ξ).
    Δt_slabfailure: float = 0.05
        Time between periodic removal of overhangs.
    do_slabfailure: bool = True
        Do periodic removal of overhangs.
    """
    type: str = ""
    label: str = ""
    ξ_min: float = 0.2
    ξ_max: float = 1
    ξ_trend: float = 0.05
    β_t: float = np.pi/4
    σ_ξ: float = 0.1
    n_ξ: float = 2
    Δt_slabfailure: float = 0.05
    do_slabfailure: bool = True

    def __post_init__(self) -> None:
        """
        Select the model type and assign its short-form label.

        Attributes
        ----------
        label: str
            Short-form model type string.
        """
        if self.type not in Models._value2member_map_.keys():
            raise ValueError("Invalid model type")
        model_labels: dict = {
            Models.ISO: "ISO",
            Models.STEP: "STP",
            Models.EA: "EA"
        }
        self.label = model_labels[self.type]

    @abstractmethod
    def ξ_fn_β(
            self, 
            β: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Erosion speed ξ(β) method that must be implemented by each model class.

        Parameters
        ----------
        β: float | NDArray | MaskedArray
            Surface tilt angle(s).

        Returns
        -------
        float | NDArray | MaskedArray:
            Surface-normal erosion speed(s) ξ(β).
        """
        pass

    @abstractmethod
    def dHlsdφ_fn_Δφx(
            self, 
            Δφ_x: float | NDArray | MaskedArray,
            Δφ_z: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Derivative of ls H_{∂φ/∂x} wrt ∂φ/∂x that must be set by model class.

        Parameters
        ----------
        Δφ_x: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂x.
        Δφ_z: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂z.

        Returns
        -------
        float | NDArray | MaskedArray:
            Derivative of ls H_{∂φ/∂x} wrt ∂φ/∂x.
        """
        pass

    @abstractmethod
    def dHlsdφ_fn_Δφz(
            self, 
            Δφ_x: float | NDArray | MaskedArray,
            Δφ_z: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Derivative of ls H_{∂φ/∂z} wrt ∂φ/∂z that must be set by model class.

        Parameters
        ----------
        Δφ_x: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂x.
        Δφ_z: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂z.

        Returns
        -------
        float | NDArray | MaskedArray:
            Derivative of ls H_{∂φ/∂z} wrt ∂φ/∂z.
        """
        pass

class Isotropic(BaseStepModel):
    """
    Isotropic erosion speed model, i.e., simple eikonal model.

    Parameters
    ----------
    See base class.

    Attributes
    ----------
    See base class.
    """
    def ξ_fn_β(
            self, 
            β: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        r"""
        Surface-normal erosion speed ξ(β) for isotropic erosion speed model.

        Parameters
        ----------
        β: float | NDArray | MaskedArray
            Surface tilt angle(s).

        Attributes
        ----------
        References ξ_min attribute.

        Returns
        -------
        float | NDArray | MaskedArray:
            Surface-normal erosion speed(s) ξ(β) for isotropic model.
        """
        if type(β) is np.ndarray or type(β) is np.ma.masked_array:
            return np.ones_like(β)*self.ξ_max
        else:
            return float(self.ξ_max)

    def dHlsdφ_fn_Δφx(
            self, 
            Δφ_x: float | NDArray | MaskedArray,
            Δφ_z: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Derivative of ls H_{∂φ/∂x} wrt ∂φ/∂x for isotropic erosion speed model.

        Parameters
        ----------
        Δφ_x: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂x.
        Δφ_z: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂z.

        Returns
        -------
        float | NDArray | MaskedArray:
            Derivative of ls H_{∂φ/∂x} wrt ∂φ/∂x.
        """
        if type(Δφ_x) is np.ndarray or type(Δφ_x) is np.ma.masked_array:
            return np.zeros_like(Δφ_x)
        else:
            return float(0)
        
    def dHlsdφ_fn_Δφz(
            self, 
            Δφ_x: float | NDArray | MaskedArray,
            Δφ_z: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Derivative of ls H_{∂φ/∂z} wrt ∂φ/∂z for isotropic erosion speed model.

        Parameters
        ----------
        Δφ_x: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂x.
        Δφ_z: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂z.

        Returns
        -------
        float | NDArray | MaskedArray:
            Derivative of ls H_{∂φ/∂z} wrt ∂φ/∂z.
        """
        if type(Δφ_z) is np.ndarray or type(Δφ_z) is np.ma.masked_array:
            return np.zeros_like(Δφ_z)
        else:
            return float(0)

class Step(BaseStepModel):
    """
    Smoothed-step erosion speed model with switch at threshold β.

    Parameters
    ----------
    See base class.

    Attributes
    ----------
    See base class.
    """
    def ξ_fn_β(
            self, 
            β: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Compute smoothed-step erosion speed with switch at threshold β.

        Parameters
        ----------
        β: float | NDArray | MaskedArray
            Surface tilt angle(s).

        Attributes
        ----------
        References β_t, σ_ξ, ξ_min, ξ_max.
        Also now: ξ_trend to provide tilt.

        Returns
        -------
        float | NDArray | MaskedArray:
            Surface-normal erosion speed(s) ξ(β) for smoothed-step model.
        """
        switch: float | NDArray | MaskedArray \
            = (1+erf((np.abs(β)-self.β_t)/self.σ_ξ))/2
        return (
            np.abs(β)*self.ξ_trend +
            (self.ξ_min + switch*(self.ξ_max-self.ξ_min))
        )

    def dHlsdφ_fn_Δφx(
            self, 
            Δφ_x: float | NDArray | MaskedArray,
            Δφ_z: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Derivative of ls H_{∂φ/∂x} wrt ∂φ/∂x for smoothed-step erosion model.

        Parameters
        ----------
        Δφ_x: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂x.
        Δφ_z: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂z.

        Returns
        -------
        float | NDArray | MaskedArray:
            Derivative of ls H_{∂φ/∂x} wrt ∂φ/∂x.
        """
        return Δφ_x*0

    def dHlsdφ_fn_Δφz(
            self, 
            Δφ_x: float | NDArray | MaskedArray,
            Δφ_z: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Derivative of ls H_{∂φ/∂z} wrt ∂φ/∂z for smoothed-step erosion model.

        Parameters
        ----------
        Δφ_x: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂x.
        Δφ_z: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂z.

        Returns
        -------
        float | NDArray | MaskedArray:
            Derivative of ls H_{∂φ/∂z} wrt ∂φ/∂z.
        """
        return Δφ_z*0

class ExponentialActivation(BaseStepModel):
    """
    Exponential-activation erosion speed model with switch at threshold β.

    Parameters
    ----------
    See base class.

    Attributes
    ----------
    See base class.
    """
    def ξ_fn_β(
            self, 
            β: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Surface-normal erosion speed ξ(β) for EA model with step at β_t.

        Parameters
        ----------
        β: float | NDArray | MaskedArray
            Surface tilt angle(s).

        Attributes
        ----------
        References β_t, σ_ξ, ξ_min, ξ_max.

        Returns
        -------
        float | NDArray | MaskedArray:
            Surface-normal erosion speed(s) ξ(β) for EA model.
        """
        def xi_fn(
                β_: float | NDArray | MaskedArray
            ) -> float | NDArray | MaskedArray:
            return (
                self.ξ_min 
                + 
                (self.ξ_max-self.ξ_min)
                * np.exp(
                    -np.abs((np.tan(self.β_t)/np.tan(np.abs(β_))))**self.n_ξ
                )
            )
        
        xi_: float | NDArray | MaskedArray = xi_fn(β)
        π: float = np.pi
        if type(β) is np.ndarray or type(β) is MaskedArray:
            xi_[np.abs(β)<=1e-10] = self.ξ_min      #type: ignore
            # If there are overhanging points in the β array,
            #   limit at ξ(π/2)
            if xi_[np.abs(β)>=π/2].shape[0]>0:      #type: ignore
                xi_[np.abs(β)>=π/2] = (             #type: ignore
                    xi_fn(π/2) 
                        # * (1 + ((np.abs(β[np.abs(β)>=π/2])-π/2)/(π/2))*0 )
                )
        else:
            xi_ = (
                self.ξ_min if np.abs(β)<=1e-10
                # 1e-10 if np.abs(β)<=0.05
                else (
                    xi_fn(π/2) 
                        # * (1 + ((np.abs(β)-π/2)/(π/2))*0 )
                    if np.abs(β)>=π/2 else xi_
                )
            )
        return xi_
    
    def dHlsdφ_fn_Δφx(
            self, 
            Δφ_x: float | NDArray | MaskedArray,
            Δφ_z: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Derivative of ls H_{∂φ/∂x} wrt ∂φ/∂x for EA model.

        Parameters
        ----------
        Δφ_x: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂x.
        Δφ_z: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂z.

        Returns
        -------
        float | NDArray | MaskedArray:
            Derivative of ls H_{∂φ/∂x} wrt ∂φ/∂x.
        """
        ξ_tp = (self.ξ_max-self.ξ_min)
        ξ_bs = self.ξ_min
        β_t = self.β_t
        tan = np.tan
        tanβ = Δφ_x/Δφ_z
        secβ = np.sqrt(1+tanβ**2)
        cosβ = 1/secβ
        exp = np.exp
        return (
            ((
                2*ξ_tp*tan(β_t)**2 * exp(-tan(β_t)**2/tanβ**2)
                + (ξ_bs + ξ_tp*exp(-tan(β_t)**2/tanβ**2))*cosβ**2*tanβ**4
            ) / (cosβ*tanβ**3)) * np.heaviside(tanβ, 0)
        )
        
    def dHlsdφ_fn_Δφz(
            self, 
            Δφ_x: float | NDArray | MaskedArray,
            Δφ_z: float | NDArray | MaskedArray,
        ) -> float | NDArray | MaskedArray:
        """
        Derivative of ls H_{∂φ/∂z} wrt ∂φ/∂z for EA model.

        Parameters
        ----------
        Δφ_x: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂x.
        Δφ_z: float | NDArray | MaskedArray
            Finite difference approximation of ∂φ/∂z.

        Returns
        -------
        float | NDArray | MaskedArray:
            Derivative of ls H_{∂φ/∂z} wrt ∂φ/∂z.
        """
        ξ_tp = (self.ξ_max-self.ξ_min)
        ξ_bs = self.ξ_min
        β_t = self.β_t
        tan = np.tan
        tanβ = Δφ_x/Δφ_z
        secβ = np.sqrt(1+tanβ**2)
        cosβ = 1/secβ
        exp = np.exp
        return (
            ((-ξ_bs
             - ξ_tp*exp(-tan(β_t)**2/tanβ**2)
             + 2*ξ_tp*exp(-tan(β_t)**2/tanβ**2)*tan(β_t)**2
                / (1 - cosβ**2)
            ) * cosβ) * np.heaviside(tanβ, 0)
        )
        
def choose_model(parameters: dict,) -> (
          Isotropic 
        | Step
        | ExponentialActivation
    ):
    """
    Choose erosion model.

    Parameters
    ----------
    dict: 
        Dictionary of keyword arguments to be passed to instantiate
        a model class. Choice of model type is specified in
        the `parameters['type']` element.

    Returns
    -------
    Isotropic | Step | ExponentialActivation:
        One of these erosion model classes.
        Instantiation takes `**parameters` as keyword arguments.
    """
    models: dict = {
        Models.ISO : Isotropic,
        Models.STEP : Step,
        Models.EA : ExponentialActivation
    }
    return models[parameters["type"]](**parameters)
