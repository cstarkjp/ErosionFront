"""
Base class for level-set solver.
"""
import warnings
from dataclasses import dataclass

from erosionfront.geomorphic.surface import (
    StraightLineSurface, 
    ErrorFunctionSurface,
    SemiCircularArcSurface, 
    QuarterCircularArcSurface,
)
from erosionfront.geomorphic.substrate import (
    OneLayerSubstrate, 
    TwoLayerSubstrate, 
    MultiLayerSubstrate,
)
from erosionfront.geomorphic.model import (
    Isotropic,
    Step,
    ExponentialActivation,
)
from erosionfront.geomorphic.gma import GeometricMechanicsAnalysis
from erosionfront.levelset.domain import Domain

warnings.filterwarnings("ignore")

__all__ = ["LevelSetData"]

@dataclass
class LevelSetData:
    """
    Base data class for front propagation.

    Parameters
    ----------
    Δx: float
        Grid spacing in x and z directions (assumed equal).
    Δt: float = 0
        Size of time step. Zero value indicates "not set yet".
        Value obtained from `Δt_raw` in JSON parameters file.
    n_total: int | None = None
        Total number of time steps to take. `None` indicates "not set yet".
    i_step: int | None = None
        Index of time step.
    t_total: float = 0
        Total duration of simulation. Zero value indicates "not set yet".
    Δt_reinitialize: float = 0.1
        Period of reinitialization of φ signed-distance function grid.
    κ_boost: float = 1
        Scale factor for boosting numerical viscosity.
    n_slabfailure: int | None = None
        Width of slab to fail (be removed) in pixels (in x direction).
    n_pad_pixels: int | None = None
        Width of vertical padding (in pixels) above & below main grid.
        Value obtained from `pad_width_raw` in JSON parameters file.
    band_width: float | None = None
        Width of level-set band across which signed-distance fn φ is computed.
        Value obtained from `n_band_width_raw` in JSON parameters file.
    fm_order: int | None = None
        Order of fast marching computational stencil (only 2 seems reliable).
    do_extend_line: bool = True
        Choose to extend surface line artifically to ensure spanning of [0,1] z.
    do_φ_everywhere: bool | None = None
        compute φ across grid when measuring pixel offsets from line
    surface: ( \
        StraightLineSurface \
        | ErrorFunctionSurface \
        | SemiCircularArcSurface \
        | QuarterCircularArcSurface \
        | None \
    ) = None
        Surface class instance.
    substrate:  ( \
        OneLayerSubstrate | TwoLayerSubstrate | MultiLayerSubstrate | None \
    ) = None
        Substrate class instance.
    model: Isotropic | Step| ExponentialActivation | None = None
        Erosion speed model class instance.
    gma: GeometricMechanicsAnalysis | None = None
        Geometric mechanics analysis class instance.

    Attributes
    ----------
    n_pixels_xz0: tuple[int,int] | None = None
        Location of (0,0) in pixels on grid.
    n_pixels_xz: tuple[int,int] | None = None
        Size of grid in pixels.
    domain: Domain | None = None
        Grid domain including padding around rasterized surface line.
    raw_domain: Domain | None = None
        Grid domain, prior to padding, set purely by rasterized surface line.
    """
    Δx: float
    Δt: float = 0
    n_total: int | None = None
    i_step: int | None = None
    t_total: float = 0
    Δt_reinitialize: float = 0.1
    κ_boost: float = 1
    n_slabfailure: int | None = None
    n_pad_pixels: int | None = None
    band_width: float | None = None
    fm_order: int | None = None

    n_pixels_xz0: tuple[int,int] | None = None
    n_pixels_xz: tuple[int,int] | None = None
    
    do_extend_line: bool = True
    do_φ_everywhere: bool | None = None

    domain: Domain | None = None
    raw_domain: Domain | None = None
    surface: (
        StraightLineSurface 
        | ErrorFunctionSurface 
        | SemiCircularArcSurface 
        | QuarterCircularArcSurface
        | None
    ) = None
    substrate:  (
        OneLayerSubstrate | TwoLayerSubstrate | MultiLayerSubstrate | None
    ) = None
    model: Isotropic | Step| ExponentialActivation | None = None
    gma: GeometricMechanicsAnalysis | None = None

    def __post_init__(self) -> None:
        """
        Supply default values for unspecified parameters. Generate bounds etc.
        """
        if self.n_pad_pixels is None:
            self.n_pad_pixels = max(
                int(1/self.Δx), 10
            )
        if self.fm_order is None:
            self.fm_order = 2
        if self.do_φ_everywhere is None:
            self.do_φ_everywhere = True