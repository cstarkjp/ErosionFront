"""
Elementary methods for level-set solver.
"""
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon   #type: ignore

from erosionfront.levelset.domain import Domain
from erosionfront.levelset.data import LevelSetData

warnings.filterwarnings("ignore")

__all__ = ["LevelSetElementaryMethods"]

class LevelSetElementaryMethods(LevelSetData):
    """
    Basic methods for working with gridded domain in level-set solver.

    Subclasses LevelSetData.

    Attributes
    ----------
    points_xz: NDArray
        Surface point coordinates.
    """
    points_xz: NDArray
    # compute_domain: Callable[[tuple | NDArray,tuple | NDArray,float], Domain]

    def ij_to_xz(self, ij: NDArray | tuple,) -> NDArray:
        """
        Convert pixel indices [i,j] into pixel *center* coords (x,z).

        Parameters
        ----------
        ij: NDArray | tuple
            Pixel index pairs [i,j] as numpy array or tuple.

        Returns
        ------- 
        NDArray:
            Pixel (x,z) coordinate pairs as float64 numpy array.
        """
        if self.domain is None or (origin:=self.domain.origin()) is None:
            raise ValueError("Domain/origin not specified")
        return (
            self.Δx * np.array(ij).astype(np.float64) 
            + origin 
            + np.array([1,1])*self.Δx/2
        )
        
    def xz_to_ij(self, xz: NDArray | tuple,) -> NDArray:
        """
        Convert pixel *center* coords (x,z) into pixel indices [i,j].

        Parameters
        ----------
        xz: NDArray | tuple
            Pixel (x,z) coordinate pairs as numpy array or tuple.

        Returns
        ------- 
        NDArray:
            Pixel index pairs [i,j] as int64 numpy array
        """
        if self.domain is None or (origin:=self.domain.origin()) is None:
            raise ValueError("Domain/origin not specified")
        return np.int64(  #type: ignore
            np.array(xz)/self.Δx 
            - origin/self.Δx
        ) 

    def ji_to_xz(self, ji: NDArray | tuple,) -> NDArray:
        """
        Convert reversed pixel indices [j,i] into pixel *center* coords (x,z).

        Parameters
        ----------
        ji: NDArray | tuple
            Pixel (reversed) index pairs [j,i] as numpy array or tuple.

        Returns
        ------- 
        NDArray:
            Pixel (x,z) coordinate pairs as float64 numpy array.

        """
        return self.ij_to_xz(np.array(ji)[::-1])
    
    def get_x_limits(self) -> tuple:
        """
        Get domain x min, max.

        Returns
        -------
        tuple:
            x_min, x_max pair.
        """
        if self.domain is None or self.domain.extent is None:
            raise ValueError("Domain/extent not specified")
        return tuple(self.domain.extent[:2])

    def get_z_limits(self) -> tuple:
        """
        Get domain z min, max.

        Returns
        -------
        tuple:
            z_min, z_max pair.
        """
        if self.domain is None or self.domain.extent is None:
            raise ValueError("Domain/extent not specified")
        return tuple(self.domain.extent[2:][::-1])
    
    def get_extent(self) -> NDArray:
        """
        Get domain (x,z) extent.

        Returns
        -------
        NDArray:
            Domain bounds as [x_min, x_max, z_max, z_min].

        """
        if self.domain is None or self.domain.extent is None:
            raise ValueError("Domain/extent not specified")
        return self.domain.extent
    
    def get_points_xz(self) -> NDArray:
        """
        Get surface (x,z) coordinates.

        Returns
        -------
        NDArray
            Points along surface as (x,z) pairs.
        """
        return self.points_xz.T
    
    @staticmethod
    def compute_pixel_dimensions(
            domain: Domain,
            resolution: float,
        ) -> tuple[int, int]:
        """
        Calculate numbers of pixels spanning the grid.

        Parameters
        ----------
        domain: Domain
            Grid domain geometry.
        resolution: float
            Grid pixel size.

        Returns
        -------
        tuple[int, int]
            Width and height of grid in pixel numbers.
        """
        n_i_pixels: int = int(
            (domain.x_max-domain.x_min)/resolution + 0.5
        )
        n_j_pixels: int = int(
            (domain.z_max-domain.z_min)/resolution + 0.5
        )
        return (n_i_pixels, n_j_pixels,)

    @staticmethod
    def compute_pixel_origin(
            domain: Domain,
            resolution: float,
        ) -> tuple[int, int]:
        """
        Calculate pixel offset of domain x,z position (0,0).

        Parameters
        ----------
        domain: Domain
            Grid domain geometry.
        resolution: float
            Grid pixel size.

        Returns
        -------
        tuple[int, int]
            Pixel indexes of bottom-left origin of grid.        
        """
        i_origin: int = int(domain.x_min/resolution+0.5)
        j_origin: int = int(domain.z_min/resolution+0.5)
        return (i_origin, j_origin,)
    
    @staticmethod
    def compute_domain(
            x: tuple | NDArray,
            z: tuple | NDArray,
            resolution: float,
        ) -> Domain:
        """
        (Re)build domain after resizing.

        Parameters
        ----------
        x: tuple | NDArray
            Vectors of x coordinates for which a domain geometry needs
            to be measured.
        z: tuple | NDArray
            Vectors of z coordinates for which a domain geometry needs
            to be measured.
        resolution: float
            Grid pixel size.

        Returns
        -------
        Domain
            Domain geometry class instance.
        """
        n_digits: int = int(round(-np.log10(resolution)+0.5))+1
        min: Callable = lambda xz: round(float(
            (np.min(np.array(xz)/resolution) - 0.5) 
        ) * resolution, n_digits)
        max: Callable = lambda xz: round(float(
            (np.max(np.array(xz)/resolution) + 0.5) 
        ) * resolution, n_digits)
        return Domain(min(x), max(x), min(z), max(z),)
    
    @staticmethod
    def pad_domain(
            domain: Domain, 
            resolution: float,
            n_pad_pixels: int,
        ) -> Domain:
        """
        Expand the domain vertically by pixel width amounts.

        Parameters
        ----------
        domain: Domain
            (Initial) domain to be padded.
        resolution: float
            Grid pixel size.
        n_pad_pixels: int
            Number of pixel widths to pad by *below* the current domain.

        Returns
        -------
        New domain instance padded vertically.
        """
        dz_bot: float = (n_pad_pixels)*resolution
        dz_top: float = (n_pad_pixels)*resolution
        # dz_top: float = resolution*5 if n_pad_pixels>0 else 0
        n_digits: int = int(round(-np.log10(resolution)+0.5))+1
        return Domain(
            round(domain.x_min, n_digits),
            round(domain.x_max, n_digits),
            round(domain.z_min - dz_bot, n_digits),
            round(domain.z_max + dz_top, n_digits),
        )
    
    @staticmethod
    def pad_grid(
            grid: NDArray, 
            n_pad_pixels: int, 
        ) -> NDArray:
        """
        Pad grid with specified number of pixels below and above.

        Parameters
        ----------
        grid: NDArray
            Grid to be padded.
        n_pad_pixels: int
            Number of pixels to pad below.

        Returns
        -------
        NDArray
            Padded grid.
        """
        pad_ztop: NDArray \
            = np.zeros((n_pad_pixels,grid.shape[1],), dtype=np.int64,)
        pad_zbot: NDArray \
            = np.zeros((n_pad_pixels,grid.shape[1],), dtype=np.int64,)
        expanded_grid: NDArray 
        if n_pad_pixels>0:
            expanded_grid =  np.vstack((pad_zbot, grid, pad_ztop,))
        else:
            expanded_grid = grid
        return expanded_grid

    @staticmethod
    def make_line(points_xz: NDArray,) -> LineString:
        """
        Convert surface (x,z) coordinates into a Shapely Line geometry.

        Parameters
        ----------
        points_xz: NDArray
            Stacked vectors of x and z surface point coordinates.

        Returns
        -------
        LineString:
            Shapely LineString geometry of surface line.
        """
        return LineString(points_xz)
    
    @staticmethod
    def make_polygon(
            points_xz: NDArray,
            resolution: float,
            n_pixels_xz: tuple | NDArray,
        ) -> Polygon:
        """
        Create a 'bounding' polygon from surface x,z points.

        This polygon is used to decide whether a point lies within the bedrock
        or outside it.

        BUG: Crudely done. Should be much tighter geometry close to domain.

        Parameters
        ----------
        points_xz: NDArray
            Surface coordinates
        n_pixels_xz: tuple | NDArray
            Grid dimensions
        resolution: float
            Grid pixel size.

        Returns
        -------
        Polygon
            Shapely geometry of bounding polygon.
        """
        if n_pixels_xz is None:
            raise ValueError("Grid size not specified")
        x_offset: float = n_pixels_xz[0]*resolution*10
        z_offset: float = n_pixels_xz[1]*resolution*10
        left_anchor: NDArray \
            = np.array([points_xz[0][0]-x_offset, 
                        points_xz[0][1]-z_offset])
        right_anchor: NDArray \
            = np.array([points_xz[-1][0]+x_offset, 
                        points_xz[-1][1]-z_offset])
        poly_points: NDArray \
            = np.vstack((
                left_anchor, 
                points_xz[0]-np.array([resolution*10,0]),
                points_xz, 
                points_xz[-1]+np.array([resolution*10,0]),
                right_anchor,
            ))
        return Polygon(poly_points)


    def __post_init__(self):
        """
        Complete instantiation by computing domain, pixel size, offset.

        Attributes
        ----------
        raw_domain: Domain
            Model domain instance.
        n_pixels_xz: tuple[int, int]:
            Grid pixel size.
        n_pixels_xz0: tuple[int, int]:
            Origin location in pixels.
        """
        super().__post_init__()
        # Compute bounds, extent, pixel dimensions of grid.
        # These methods MUST be provided by any subclass.
        if self.surface is None:
            raise ValueError("Surface not specified")
        self.raw_domain \
            = self.compute_domain(
                self.surface.x, self.surface.z, self.Δx,
            )
        self.n_pixels_xz \
            = self.compute_pixel_dimensions(self.raw_domain, self.Δx,)
        self.n_pixels_xz0 \
            = self.compute_pixel_origin(self.raw_domain, self.Δx,)