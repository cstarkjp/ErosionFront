"""
Level-set solution of rock-slope erosion-front evolution.
"""
import warnings
from copy import copy
from collections.abc import Callable
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from numpy.ma import MaskedArray
from scipy.interpolate import griddata

from erosionfront.misc.utils import Timer
from erosionfront.levelset.speed import LevelSetSpeedMethods

warnings.filterwarnings("ignore")

__all__ = ["LevelSetSolverBase"]

class LevelSetSolverBase(LevelSetSpeedMethods, ABC):
    """
    Solver for propagating an erosion front using a level-set scheme.

    Subclasses LevelSetSpeedMethods.

    Attributes
    ----------
    ρ: NDArray
        Rasterized surface-line grid.
    ρ_padded: NDArray
        Rasterized surface-line grid with padding.
    ρ_offsets: NDArray
        Gridded surface-normal distances.
    T_slices: dict
        Dictionary recording time T slice data of erosion front evolution.
    T_arrival: MaskedArray | None = None
        Grid of interpolated arrival times T, with mask removing pixels
        in air or in uneroded substrate.
    """
    ρ: NDArray
    ρ_padded: NDArray
    ρ_offsets: NDArray
    T_slices: dict
    T_arrival: MaskedArray | None = None

    @abstractmethod
    def update(
            self, 
            Δt: float, 
            Δt_reinitialize: float | None=None,
            do_finalize: bool=False,
            do_timing: bool=False,
        ) -> None:
        """
        Update method that must be implemented by the actual solver.

        Parameters
        ----------
        Δt: float
            Time step.
        Δt_reinitialize: float | None=None
            Periodicity of reinitialization of φ signed-distance grid.
        do_finalize: bool=False
            Just finish up, don't do another loop of computation.
        do_timing: bool=False
            Measure and report compute times taken at steps during the loop.

        Attributes
        ----------
        Most of the sim grids are modified.
        """
        pass

    def initialize(self, do_timing: bool=False,) -> None:
        """
        Initialize level-set solver.

        Parameters
        ----------
        do_timing: bool=False
            Perform timing estimates for each computational stage.

        Attributes
        ----------
        Many class attributes are modified.
        """
        Timer(do_timing, "Initializing",)

        Timer.start("create initial surface line; generate subsurface polygon",)
        self.domain = copy(self.raw_domain)
        if self.surface is None:
            raise ValueError("Surface not specified")
        self.points_xz = self.surface.get_points_xz()
        self.line = self.make_line(self.points_xz)
        if self.n_pixels_xz is None:
            raise ValueError("Grid size not specified")
        self.in_rock_polygon \
            = self.make_polygon(
                self.points_xz, 
                self.Δx,
                self.n_pixels_xz, 
            )
        Timer.stop()

        Timer.start("rasterize line",)
        self.ρ = self.rasterize_line(self.points_xz, self.do_extend_line,)
        Timer.stop()
        Timer.start("pad the grid vertically",)
        if self.raw_domain is None:
            raise ValueError("Domain not specified")
        if self.n_pad_pixels is None:
            raise ValueError("Number of pad pixels not specified")
        self.domain = self.pad_domain(
            self.raw_domain, 
            self.Δx,
            self.n_pad_pixels,
        )
        self.n_pixels_xz \
            = self.compute_pixel_dimensions(self.domain, self.Δx,)
        self.n_pixels_xz0 \
            = self.compute_pixel_origin(self.domain, self.Δx,)
        if self.n_pad_pixels is None:
            raise ValueError("Number of padding pixels not specified")
        self.ρ_padded \
            = self.pad_grid(
                grid=self.ρ, 
                n_pad_pixels=self.n_pad_pixels,
            )
        Timer.stop()

        Timer.start("measure pixel offsets from line",)
        self.ρ_offsets \
            = self.compute_line_pixel_offsets(
                line=self.line,
                ρ=self.ρ_padded,
                do_φ_everywhere=self.do_φ_everywhere,
            )
        Timer.stop()

        Timer.start("fast-march signed distance from pixel offsets",)
        self.φ \
            = self.fm_distance(
                f=self.ρ_offsets, 
                band_width=self.band_width,
                resolution=self.Δx,
                order=self.fm_order,
            )
        Timer.stop()

        Timer.start("fill whole signed-distance grid with very-far values",)
        self.φ_everywhere \
            = self.redo_φ_everywhere(
                φ_everywhere=self.ρ_offsets, 
                φ=self.φ,
            )
        Timer.stop()


    def evolve(
            self, 
            Δt: float=0, 
            t_span: float | None = None,
            n_steps: int | None = None,
            Δt_reinitialize: float | None = None, 
            κ_boost: float = 1,
            report_pc: int = 5,
            n_record_steps: int = 5,
            do_initial: bool=False,
            do_timing: bool=False,
        ) -> None:
        """
        Iteratively propagate the erosion front surface.

        Parameters
        ----------
        Δt: float=0
            Simulation time step.
        t_span: float | None = None
            Duration of simulation.
        n_steps: int | None = None
            Number of time steps to take in the simulation.
            If neither `t_span` nor `n_steps` is specified, 
            If neither are, a single step is assumed.
        Δt_reinitialize: float | None = None
            Period of reinitialization of φ signed-distance function grid.
        κ_boost: float = 1
            Scale factor for boosting numerical viscosity.
        report_pc: int = 5
            Reporting interval as percentage of runtime.
        n_record_steps: int = 5
            Number of steps between T slicings.
        do_initial: bool=False
            Specify if this is the first update, and act accordingly.
        do_timing: bool=False
            Choose whether to perform timings of sim stages.
        Attributes
        ----------
        Most of the sim grids are modified.
        """
        self.Δt = Δt
        if self.Δt>0 and self.model is not None:
            self.n_slabfailure = int(self.model.Δt_slabfailure/self.Δt)
        self.κ_boost = κ_boost
        if Δt_reinitialize is None:
            self.Δt_reinitialize = Δt
        else:
            self.Δt_reinitialize = Δt_reinitialize
        n_steps_: int
        t_span_: float
        if n_steps is None:
            if t_span is None:
                n_steps_ = 1
                t_span_ = Δt
            else:
                n_steps_ = int(t_span/Δt+0.5)
                t_span_ = t_span
        else:
            n_steps_ = n_steps
            t_span_ = n_steps*Δt
        print(
            rf"Run time T={t_span_} for n_steps={n_steps_} "
            + rf"with Δt={Δt} and Δx={self.Δx}"
        )
        do_report: Callable \
            = lambda step: (
                False if int(n_steps_*report_pc/100)==0
                else (step+1)%int(n_steps_*report_pc/100)==0
            )
        # do_record: Callable \
        #     = lambda step: (
        #         False if int(n_steps_*record_pc/100)==0 
        #         else (step+1)%int(n_steps_*record_pc/100)==0
        #     )
        rnd: int = int(np.round(-np.log10(Δt if Δt>0 else 1))+1)
        def record_report(step) -> None:
            progress_pc_: float
            if do_report(step):
                progress_pc_ = int(100*(step+1)/(n_steps_))
                print(f"   {progress_pc_}%:  step={self.i_step}"
                    + f"  t={np.round(self.t_total, rnd)}")
            if ((step+1) % n_record_steps)==0:
                self.record_T_slice(
                    self.i_step if self.i_step is not None else 1, 
                    self.t_total, 
                )

        #######################################################################
        if self.t_total is None or do_initial:
            # Initial do-nothing step...
            self.record_T_slice(0, 0,)
            self.i_step = 0
            self.update(0, do_timing=do_timing,)
        else:
            # ...or a full run
            for step_ in range(n_steps_):
                self.i_step = self.i_step+1 if self.i_step is not None else 1
                self.update(Δt, Δt_reinitialize,)
                record_report(step_)
            self.update(Δt, do_finalize=True, do_timing=do_timing,)
        self.n_total = self.i_step
        #######################################################################
    

    def record_T_slice(
            self, 
            step: int, 
            t_total: float, 
        ) -> None:
        """
        Record selected sim data for a single time step.

        Recorded data are placed in a 'T_slice' dictionary.

        Parameters
        ----------
        step: int
            Simulation step number.
        t_total: float
            Total time in simulation so far.

        Attributes
        ----------
        Data for this sim time step are appended to the time-slicing dict.
        """
        if self.φ_next is None:
            d = self.φ/np.max(self.φ)
        else:
            d = self.φ_next/np.max(self.φ_next)
        # d: NDArray | MaskedArray = self.φ_everywhere/np.max(self.φ_everywhere)
        xz_profile: NDArray = self.get_front_points(d)
        x: NDArray = xz_profile[0]
        z: NDArray = xz_profile[1]
        β: NDArray = np.arctan2(np.gradient(z), np.gradient(x))
        self.T_slices[step] = {
            "x" : x,
            "z" : z,
            "φ": self.φ,
            "β" : β,
            "T" : float(t_total),
        }

    def rasterize_T_slices(self, n_subset: int=1,) -> None:
        """
        Rasterize time T slices onto grids.

        Parameters
        ----------
        n_subset: int=1
            Rate of subsetting of T slices (n=1 means all of them).
        """
        print("Gridding T slices...", end="",)
        if self.domain is None:
            raise ValueError("Domain not defined")
        bounds: NDArray | None= self.domain.bounds()
        if bounds is None:
            raise ValueError("Domain bounds not defined")
        if self.n_pixels_xz is None:
            raise ValueError("Number of x,z pixels not defined")
        n_x: int = self.n_pixels_xz[0]
        n_z: int = self.n_pixels_xz[1]
        x_bounds: NDArray = (bounds.T)[0]
        z_bounds: NDArray = (bounds.T)[1]
        x_pts: NDArray = np.linspace(x_bounds[0], x_bounds[1], n_x,) + self.Δx/2
        z_pts: NDArray = np.linspace(z_bounds[0], z_bounds[1], n_z,) + self.Δx/2
        assert x_pts.shape[0]==n_x
        assert z_pts.shape[0]==n_z
        x_mg: NDArray
        z_mg: NDArray
        (z_mg, x_mg,) = np.meshgrid(z_pts, x_pts, indexing="ij",)
        x_points: NDArray = np.concatenate(
            [slice_["x"] for slice_ in self.T_slices.values()]
        )[::n_subset]
        z_points: NDArray = np.concatenate(
            [slice_["z"] for slice_ in self.T_slices.values()]
        )[::n_subset]
        T_values: NDArray = np.concatenate(
            [slice_["x"]*0+slice_["T"] for slice_ in self.T_slices.values()]
        )[::n_subset]
        zx_points: NDArray = np.vstack((z_points, x_points,)).T
        assert type(zx_points) is np.ndarray
        assert type(T_values) is np.ndarray
        T_arrival: NDArray \
            = griddata(zx_points, T_values, (z_mg, x_mg), method="linear",)
        not_reached: NDArray = np.zeros_like(self.φ_everywhere).astype(np.bool)
        not_reached[self.φ_everywhere<0] = True
        self.T_arrival = np.ma.masked_array(T_arrival, mask=not_reached,)
        print(" done")