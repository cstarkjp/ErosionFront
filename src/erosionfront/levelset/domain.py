"""
Metadata for level-set domain grid.
"""
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

warnings.filterwarnings("ignore")

__all__ = ["Domain"]

@dataclass
class Domain:
    """
    Handle metadata for model domain geometry.

    Parameters
    ----------
    x_min: float
        Required domain minimum x provided during instantiation.
    x_max: float
        Required domain maximum x provided during instantiation.
    z_min: float
        Required domain minimum z provided during instantiation.
    z_max: float
        Required domain maximum z provided during instantiation.

    Attributes
    ----------
    extent: NDArray
        Sequence of limits as (x_min, x_max, z_max, z_min).
    """
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    extent: NDArray | None = None

    def origin(self) -> NDArray | None:
        """
        Get domain (x,z) origin.

        Returns
        -------
        NDArray | None:
            Extents as array if they exist, otherwise None.
        """
        return np.array([
            self.extent[0], self.extent[-1]
        ]) if self.extent is not None else None

    def bounds(self) -> NDArray | None:
        """
        Get domain (x,z) bounds.

        Returns
        -------
        2x2 matrix containing row vectors (x_min,z_min) and (x_max,z_max).
        """
        return np.array([
            self.extent[:2], self.extent[2:][::-1]
        ]).T if self.extent is not None else None

    def __post_init__(self) -> None:
        """
        Complete instantiation of domain class by computing bounds etc.
        """
        self.extent = np.array([
            self.x_min, self.x_max, self.z_max, self.z_min,
        ])