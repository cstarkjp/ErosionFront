"""
Grid methods for level-set solver.
"""
import warnings

import numpy as np
from numpy.typing import NDArray
from numpy.ma import MaskedArray
import skfmm    #type: ignore
import skimage.measure
import shapely    #type: ignore
from shapely.geometry import LineString, Polygon, Point #type: ignore
import rasterio.features    #type: ignore

from erosionfront.levelset.elementary import LevelSetElementaryMethods

warnings.filterwarnings("ignore")

__all__ = ["LevelSetGridMethods"]

class LevelSetGridMethods(LevelSetElementaryMethods):
    """
    Geometric methods for use in erosion front modeling.

    Inherits LevelSetElementaryMethods class.

    Attributes
    ----------
    in_rock_polygon: Polygon
        Hacked shape to specify which side of the surface line pixels lie.
    line: LineString
        Shapely geometry for surface line.
    """
    in_rock_polygon: Polygon
    line: LineString

    def rasterize_line(
            self, 
            points_xz: NDArray,
            do_extend_line: bool=True,
        ) -> NDArray:
        """
        Rasterize surface line as x,z points into grid pixels using Rasterio.

        The x,z bounds of the raster grid are precisely set by the
        bounding box of the line.

        Parameters
        ----------
        points_xz: NDArray
            Surface line as a series of x,z points.
        do_extend_line: bool=True
            Artificially extend the surface line so it reaches the boundary.

        Attributes
        ----------
        Uses n_pixels_xz and xz_to_ij methods.

        Returns
        -------
        NDArray
            Grid of rasterized line.
        """
        #HACK: artificially extend line horizontally to ensure
        #      we get correct bounds
        points_xz_extended: NDArray
        if do_extend_line:
            points_xz_extended  = np.vstack((
                [2*points_xz[0][0],0],
                points_xz,
                [2*points_xz[-1][0],1],
            ))
        else:
            points_xz_extended = points_xz.copy()
        pts_ij = self.xz_to_ij(points_xz_extended)
        line: LineString = LineString(pts_ij)
        if self.n_pixels_xz is None:
            raise ValueError("Grid size not specified")
        # Specific caching actually slows rasterization down!
        #   with rasterio.Env(GDAL_CACHEMAX=4_096_000_000) as env:
        ρ: NDArray \
            = rasterio.features.rasterize(
                [line],
                out_shape=self.n_pixels_xz[::-1], 
                all_touched=True, 
                fill=0,
            )
        # assert ρ.shape==self.n_pixels_xz[::-1]
        return ρ #, line

    def get_line_points(self) -> NDArray:
        """
        Convert Shapely geometry Line into (x,z) points.

        Attributes
        ----------
        Uses line attribute.

        Returns
        -------
        NDArray
            Line as N coordinates in (2,N) array format, 
            i.e., as two rows of N-length sequences.
        """
        return (shapely.get_coordinates(self.line)).T
    
    @staticmethod
    def measure_normal_distance(
            line: tuple | NDArray | LineString, 
            point: tuple | NDArray | Point,
        ) -> float:
        """
        Compute distance between line and point using Shapely.

        Parameters
        ----------
        line: tuple | NDArray | LineString
            Line represented by various possible types.
        point: tuple | NDArray | Point
            Coordinate (x,z) represented by various possible types.

        Returns
        -------
        float
            Perpendicular distance from line curve to point.
        """
        return float(shapely.distance(
            line if type(line) is LineString else LineString(line), 
            point if type(point) is Point else Point(point)
        ))

    @staticmethod
    def above_or_below_xz(
            xz: tuple[float,float] | NDArray,
            in_rock_polygon: Polygon,
        ) -> int:
        """
        Test whether point x,z lies above or below surface.

        Uses "in_rock_polygon" to test if point is within the bedrock or in 
        the air. In other words, it tests which side of the surface zeroset
        the point lies.

        Parameters
        ----------
        xz: tuple[float,float] | NDArray
            Test point.
        in_rock_polygon: Polygon
            Shapely polygon defining "below ground".

        Returns
        -------
        -1 if "in the air" and +1 if "in the rock".
        """
        return (
            -1 if shapely.contains_xy(in_rock_polygon, *xz,) else +1
        )

    def above_or_below_ji(
            self,
            ji: tuple[int,int] | NDArray,
            in_rock_polygon: Polygon,
        ) -> int:
        """
        Test whether pixel i,j lies above or below surface.

        Uses "in_rock_polygon" to test if pixel is within the bedrock or in 
        the air. In other words, it tests which side of the surface zeroset
        the pixel lies.

        The pixel center is used in this computation.

        Parameters
        ----------
        ji: tuple[float,float] | NDArray
            Test pixel with reversed indexes.
        in_rock_polygon: Polygon
            Shapely polygon defining "below ground".

        Atrributes
        ----------
        Uses ji_to_xz method.

        Returns
        -------
        -1 if "in the air" and +1 if "in the rock".
        """
        # integer pixel indices convert to pixel *center* x,z coords
        return (
            -1 if shapely.contains_xy(in_rock_polygon, *self.ji_to_xz(ji)) 
            else +1
        )

    def measure_φ_xz(
            self,
            line: LineString,
            xz: tuple[float,float] | NDArray,
        ) -> float:
        """
        Compute signed distance between line and xz point using Shapely.

        The sign is -1 for "in the air" and +1 for "in the rock".

        Parameters
        ----------
        line: LineString
            Surface line.
        xz: tuple[float,float] | NDArray
            Point coordinates.

        Attributes
        ----------
        Uses above_or_below_xz method to test which side of the surface the 
        point lies.

        Returns
        -------
        float
            Signed perpendicular distance from the line to the point.
        """
        return (
            self.measure_normal_distance(line, xz)
            *self.above_or_below_xz(xz, self.in_rock_polygon,)
        )

    def measure_φ_ji(
            self,
            line: LineString,
            ji: tuple[int,int] | NDArray,
        ) -> float:
        """
        Compute signed distance between line and i,j pixel using Shapely.

        The sign is -1 for "in the air" and +1 for "in the rock".

        Parameters
        ----------
        line: LineString
            Surface line.
        ji: tuple[int,int] | NDArray
            Pixel indexes in reversed format.

        Attributes
        ----------
        Uses measure_φ_xz method to test which side of the 
        surface the pixel lies.

        Returns
        -------
        float
            Signed perpendicular distance from the line to the pixel.
        """
        return self.measure_φ_xz(line, self.ji_to_xz(ji),)

    def compute_line_pixel_offsets(
            self, 
            line: LineString,
            ρ: NDArray,
            do_φ_everywhere: bool | None=False,
        ) -> NDArray:
        """
        Measure signed offsets between rasterized line pixels and line curve.

        Spatial aliasing during rasterization means each line pixel is likely
        offset a bit from the line curve, so this function computes these
        offsets.

        Parameters
        ----------
        line: LineString
        ρ: NDArray

        do_φ_everywhere: bool
            Compute distances for all grid pixels, not just those not masked.

        Attributes
        ----------
        Uses measure_φ_ji and above_or_below_ji methods.

        Returns
        -------
        NDArray
            Grid with signed-distance normal offsets from the line curve
            at each rasterized line pixel.
        """
        offsets_grid: NDArray \
            = np.zeros_like(ρ, dtype=np.float64,)
        for ji_ in np.argwhere(ρ!=0):
            # integer pixel indices convert to pixel *center* x,z coords
            offsets_grid[*ji_] = self.measure_φ_ji(line,ji_,)
        d_min: float = np.min(offsets_grid)
        d_max: float = np.max(offsets_grid)
        if do_φ_everywhere:
            for ji_ in np.argwhere(ρ==0):
                offsets_grid[*ji_] = (
                    d_min 
                    if self.above_or_below_ji(ji_, self.in_rock_polygon,)==-1 
                    else d_max
                )
        return offsets_grid
    
    @staticmethod
    def fm_distance(
            f: NDArray, 
            band_width: float | None,
            resolution: float,
            order: int | None,
        ) -> MaskedArray:
        """
        Use fast marching to compute signed distance from a ref grid (zeroset).

        The reference grid (typically travel time, but may be rasterized line
        offsets) provides a zero contour from which the signed distance is 
        computed by fast marching.

        Parameters
        ----------
        f: NDArray
            Reference grid to provide a zero contour.
        band_width: float | None,
            Width of the narrow band. 
        resolution: float
            Grid pixel width (and height).
        order: int
            Fast-marching order of computational stencil (1 or 2).

        Returns
        -------
        MaskedArray
            Signed distance grid.
        """
        φ: NDArray | MaskedArray \
              = skfmm.distance(
                    f, 
                    dx=resolution, 
                    order=order, 
                    narrow=band_width,
                )
        return (
            φ if type(φ)==MaskedArray
            else np.ma.masked_array(φ, np.zeros_like(φ, dtype=np.bool,))
        )
    
    def fm_distance_everywhere(
            self, 
            φ: MaskedArray, 
            φ_everywhere: NDArray,
            resolution: float,
            order: int | None,
        ) -> NDArray:
        """
        FM compute a narrowband signed distance and then extend everywhere.

        The previous 'everywhere' grid of signed distance is used to supply
        values beyond the narrowband.

        Parameters
        ----------
        φ: MaskedArray
            Reference grid to provide a zero contour.
        φ_everywhere: NDArray
            Previous grid of signed distance everywhere.
        resolution: float
            Grid pixel width (and height).
        order: int
            Fast-marching order of computational stencil (1 or [2]).

        Attributes
        ----------
        Uses the fm_distance method.

        Returns
        -------
        MaskedArray
            Grid with signed distance values everywhere, not just on the nb.
        """
        previous_φ_everywhere: NDArray \
            = φ_everywhere
        previous_distant_pixels: NDArray = φ.mask
        φ_everywhere_: NDArray \
            = self.fm_distance(φ, 0, resolution, order,).data
        φ_everywhere_[previous_distant_pixels] \
            = previous_φ_everywhere[previous_distant_pixels]
        return φ_everywhere_

    def compute_contour(
            self, 
            f: NDArray | MaskedArray, 
            level: float=0,
        ) -> NDArray:
        """
        Generate a continuous (for now) contour of grid function f.

        BUG: the contouring function actually returns a set of
        disjoint contour segments, often with more than one segment.
        Here it's incorrectly assumed these segments are contiguous 
        and can simply be concatenated. This is not true! But to do
        better downstream use of this method needs to handle the segment set
        properly, which is going to be inconvenient.

        Parameters
        ----------
        f: NDArray | MaskedArray
            Smooth grid (e.g., signed distance) on which to compute contour.
        level: float=0
            Chosen contour level.

        Attributes
        ----------
        Uses ij_to_xz method.

        Returns
        -------
        NDArray
            Single sequence of x,z contour coordinates as (2,N) array.
        """
        contours: list = skimage.measure.find_contours(
            f, 
            level=level, 
            mask=~f.mask if type(f) is MaskedArray else None,
        )
        # print(len(contours))
        #HACK: drop second contour (presumably generated by front sliding
        #      off grid) if it's too short & thus likely erroneous
        # if len(contours)>1:
        #     # print(contours[0].shape, contours[1].shape)
        #     if contours[1].shape[0]<10:
        #         contours = contours[0]
        return self.ij_to_xz(np.fliplr(np.vstack(contours))).T
    
    def get_front_points(
            self, 
            f: NDArray | MaskedArray,
        ) -> NDArray:
        """
        Fetch points along erosion front surface aka zeroset of f.

        Parameters
        ----------
        f: NDArray | MaskedArray
            Presumably a signed distance grid (but not necessarily).

        Attributes
        ----------
        Uses compute_contour method.

        Returns
        -------
        NDArray
            Coordinate sequence (x,z)*N along surface as (2,N) array.
        """
        return self.compute_contour(f, level=0,)

    @staticmethod
    def redo_φ_everywhere(
            φ_everywhere: NDArray, 
            φ: MaskedArray,
        ) -> NDArray:
        """
        Reassign signed distance values to pixels beyond the narrow band.

        Assumes that pixels beyond the narrow band have signed-distance signs
        that are currently correct: given these signs, each beyond-nb pixel is 
        assigned either the min or max value of the nb signed-distance values.

        Parameters
        ----------
        φ_everywhere: NDArray
            Beyond narrow-band signed-distance grid to be recomputed.
        φ: MaskedArray
            Narrow-band signed-distance grid.

        Returns
        -------
        NDArray
            Recomputed signed-distances beyond narrow band.
        """
        mask: NDArray = np.array(φ.mask)
        far_recomputed: NDArray = φ_everywhere.copy()
        far_recomputed[φ_everywhere<0] = np.min(φ)
        far_recomputed[φ_everywhere>0] = np.max(φ)
        far_recomputed[~mask] = φ[~mask]
        return far_recomputed