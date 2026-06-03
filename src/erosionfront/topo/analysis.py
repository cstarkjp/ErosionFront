"""
Analysis of 3D DTM data.
"""
import warnings
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
from functools import partial

import numpy as np
from numpy.typing import NDArray
from pyvista import MultiBlock, PolyData, Plotter

warnings.filterwarnings("ignore")

__all__ = [
    "denest", 
    "patch_up",
    "Geometry",
    "ProfileData",
    "Profiles",
]

def denest(nested_mesh: MultiBlock) -> MultiBlock:
    """
    Recursively search a nested `MultiBlock` for the data `MultiBlock`.

    Parameters
    ----------
    nested_mesh: MultiBlock
        Source `MultiBlock` assumed to be nested to an unknown level.
    
    Returns
    -------
    MultiBlock:
        The lowest-level denested `MultiBlock` whose child is actual data.
    """
    return (
        nested_mesh if type(nested_mesh[0]) is not MultiBlock
        else denest(nested_mesh[0])
    )

def patch_up(raw_mesh: MultiBlock,) -> MultiBlock:
    """
    Rebuild the raw multipart mesh into a single mesh with z vertical. 

    The 3D topo mesh is assumed exported from Blender, with actual mesh 
    doubly nested and with y axis vertical. 

    This PyVista-format mesh is a multiblock polydata object, meaning that 
    the mesh is in multiple pieces. 
    As a result, the mesh rotation and assembly has to be done 
    by rotating and appending one block at a time to a fresh multiblock object.

    Parameters
    ----------
    raw_mesh: MultiBlock
        Mesh in multiple pieces as `pyvista` `MultiBlock`.
    
    Returns
    -------
    MultiBlock:
        Single mesh object with correct orientation (z up).
    """
    mesh_ = MultiBlock()
    for block_ in denest(raw_mesh):
        mesh_.append(block_.rotate_x(90, point=(0,0,0,),))
    return mesh_

@dataclass
class Geometry:
    """
    Slicing geometry.

    Parameters
    ----------
    mesh: MultiBlock
        3D topo as `pyvista` `MultiBlock`
   
    Attributes
    ----------
    x_vector: NDArray | None = None
        X
    y_vector: NDArray | None = None
        X
    rotation_matrix: Callable[[float], NDArray] | None = None
        X
    grid: tuple[dict, dict] | None = None
        X
    """
    mesh: MultiBlock
    x_vector: NDArray | None = None
    y_vector: NDArray | None = None
    rotation_matrix: Callable[[float], NDArray] | None = None
    grid: tuple[dict, dict] | None = None

    def __post_init__(self):
        """
        """
        self.origin = np.array([0,0,0])
        self.x_vector = np.array([1,0,0])
        self.y_vector = np.array([0,1,0])
        x_minmax = np.round(self.mesh.bounds[0:2])
        y_minmax = np.round(self.mesh.bounds[2:4])
        self.rotation_matrix = lambda ψ_: np.array([
            [ np.cos(ψ_), np.sin(ψ_), 0],
            [-np.sin(ψ_), np.cos(ψ_), 0],
            [         0,         0,   1]
        ])
        lines: Callable[
            [MultiBlock, NDArray, NDArray, NDArray, float], 
            dict[float, MultiBlock]
        ] \
            = lambda mesh, origin, vector, minmax, step_size: {
                float(step_): mesh.slice(
                    origin=origin+vector*step_, 
                    normal=vector,
                )
                for step_ in np.arange(
                    *np.round(minmax/step_size+1)*step_size, step_size,
                )
            }
        grid_step: float = 100
        self.grid: tuple[dict[float, MultiBlock]] = (
            lines(self.mesh, self.origin, self.y_vector, y_minmax, grid_step,),
            lines(self.mesh, self.origin, self.x_vector, x_minmax, grid_step,),
        )

@dataclass
class ProfileData:
    """
    XXX

    Parameters
    ----------
    mesh: MultiBlock
        X
    geometry: Geometry
        X
    name: str
        X
    orientation: float
        X
    offset: tuple[float,float,float]
        X

    Attributes
    ----------
    mid_points: NDArray | None = None
        X
    normals: NDArray | None = None
        X
    parallels: NDArray | None = None
        X
    tilts: list[float | None] | None = None
        X
    tilts_r: list[float | None] | None = None
        X
    """
    mesh: MultiBlock
    geometry: Geometry
    name: str
    orientation: float
    offset: tuple[float,float,float]
    mid_points: NDArray | None = None
    normals: NDArray | None = None
    parallels: NDArray | None = None
    tilts: list[float | None] | None = None
    tilts_r: list[float | None] | None = None
    
    def slicing(self,) -> None:
        """
        """
        self.ψ: float = np.deg2rad(self.orientation)
        self.rotation: NDArray = self.geometry.rotation_matrix(self.ψ)
        self.normal_vector: NDArray = self.rotation @ self.geometry.y_vector
        self.slice: MultiBlock \
            = self.mesh.slice(origin=self.offset, normal=self.normal_vector,)
    
    def extract_slice_chords(self,) -> None:
        """
        """
        mesh: MultiBlock = self.mesh
        offset: tuple[float,float,float] = self.offset
        normal: NDArray = self.normal_vector
        slice_segments_raw: tuple[PolyData] = [
            (m_.slice(origin=offset, normal=normal,))
            for m_ in mesh[:]
        ]    
        slice_segments: tuple[PolyData] = [
            slice_segment_raw_ 
            for slice_segment_raw_ in slice_segments_raw
            if slice_segment_raw_.n_cells>0
        ]
        slice_chords: NDArray = np.concatenate([
            np.array([
                np.vstack([ pair_[0,[0,2]],pair_[1,[0,2]], ]).T
                for pair_ in np.stack(
                    [c_.points for c_ in list(slice_segment_.cell)]
                )
            ]) 
            for slice_segment_ in slice_segments
        ])
        self.chords = slice_chords

    def compute_angles(
            self,
            use_normals: bool=False,
            do_flip_orientation: bool=False,
        ) -> None:
        """
        X

        Parameters
        ----------
        use_normals: bool=False
            X
        do_flip_orientation: bool=False
            X
        """
        profile_ = self
        chords_: NDArray = profile_.chords
        n_chords: int = self.chords.shape[0]
        self.mid_points = np.zeros((n_chords,2,))
        self.normals = np.zeros((n_chords,2,))
        self.parallels = np.zeros((n_chords,2,))
        self.tilts = [None]*n_chords
        self.tilts_r = [None]*n_chords
        for i_, chord_ in enumerate(chords_):
            parallel_raw: NDArray = np.array([
                (chord_[0][1]-chord_[0][0]),
                (chord_[1][1]-chord_[1][0]),
            ])
            normal_raw: NDArray = np.array([
                 (chord_[1][1]-chord_[1][0]),
                -(chord_[0][1]-chord_[0][0]),
            ])
            normalize: Callable[[NDArray],NDArray] = lambda vector: (
                (vector/np.linalg.norm(vector))
                * (-1 if do_flip_orientation else 1)
            )
            parallel: NDArray = normalize(parallel_raw)
            normal: NDArray = normalize(normal_raw)
            vector: NDArray = normal if use_normals else parallel
            tilt_raw: float = np.rad2deg(np.atan2(vector[1],vector[0]))
            θ_clip: tuple[float,float]=(0,90,)
            in_bounds: Callable[[float], float | None] = lambda tilt: \
                tilt if ((tilt>=θ_clip[0] and tilt<θ_clip[1]) 
                    or (-tilt>=θ_clip[0] and -tilt<θ_clip[1])) \
                else None
            tilt: float | None = in_bounds(tilt_raw)
            tilt_reversed: float | None \
                = in_bounds(180*np.sign(tilt_raw)-(tilt_raw))

            self.mid_points[i_] \
                = np.array([np.mean(chord_[0]), np.mean(chord_[1])])
            self.tilts[i_] = tilt
            self.tilts_r[i_] = tilt_reversed
            self.parallels[i_] = parallel
            self.normals[i_] = normal

@dataclass
class Profiles:
    """
    Tools for generating 2D slices through 3D topo in `MultiBlock` mesh format.

    Parameters
    ----------
    mesh: MultiBlock
        3D topo mesh.
    geometry: Geometry
        Basic geometry of 3D topo mesh.
    info: dict[int, dict]
        Info on each desired profile, inc. name, orientation, offset.

    Attributes
    ----------
    data: dict[int: ProfileData] | None = None
        Instances of profiles.
    """
    mesh: MultiBlock
    geometry: Geometry
    info: dict[int, dict]
    data: dict[int: ProfileData] | None = None

    def __post_init__(self) -> None:
        """
        Generate profiles per the name/orientation/offset info provided.

        Attributes
        ----------
        data: dict[int: ProfileData]
            Instances of profiles.
        """
        SpecificProfile = partial(ProfileData, self.mesh, self.geometry,)
        self.data = {
            key_: SpecificProfile(**value_)
            for key_,value_ in self.info.items()
        }

    def perform_slicing(self) -> None:
        """
        Perform slicing along chosen profiles.

        Attributes
        ----------
        """
        for profile_ in self.data.values():
            profile_.slicing()

    def perform_chord_extraction(self) -> None:
        """
        Extract sets of chords (point sequences) along profiles.

        Attributes
        ----------
        """
        for profile_ in self.data.values():
            profile_.extract_slice_chords()


    def perform_angles_computation(self) -> None:
        """
        Compute surface angle information at chords along profiles.

        Attributes
        ----------
        """
        for profile_ in self.data.values():
            profile_.compute_angles(  
                use_normals=True,
                do_flip_orientation=True,
            )
