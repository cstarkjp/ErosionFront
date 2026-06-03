"""
Visualization of geomorphic Hamiltonian model.
"""
import warnings
from typing import Any

import numpy as np

from erosionfront.geomorphic.surface import (
    StraightLineSurface, 
    ErrorFunctionSurface,
    SemiCircularArcSurface, 
    QuarterCircularArcSurface, 
    Surfaces,
)
from erosionfront.geomorphic.substrate import (
    OneLayerSubstrate, 
    TwoLayerSubstrate, 
    MultiLayerSubstrate, 
    Substrates,
)
from erosionfront.geomorphic.model import (
    Isotropic, 
    Step, 
    ExponentialActivation, 
    Models,
)
from erosionfront.levelset.solver import LevelSetSolver
from erosionfront.geomorphic.gma import GeometricMechanicsAnalysis
from erosionfront.visualization.speedviz import ErosionSpeedViz
from erosionfront.visualization.gmaviz import GeometricMechanicsViz
from erosionfront.visualization.simviz import SimulationViz
from erosionfront.visualization.topoviz import TopoViz

warnings.filterwarnings("ignore")

__all__ = ["Viz2D"]

class Viz2D(
    ErosionSpeedViz, 
    GeometricMechanicsViz, 
    SimulationViz,
    TopoViz,
):
    """
    XXXX

    Parameters
    ----------
    XXX

    Attributes
    ----------
    TBD

    Returns
    -------
    XXX
    """
    @staticmethod
    def make_title(
        sim_or_model: LevelSetSolver | Isotropic | Step | ExponentialActivation, 
        name: str,
        gma: GeometricMechanicsAnalysis | None = None,
    ) -> str:
        sim: LevelSetSolver | None
        surface: StraightLineSurface | ErrorFunctionSurface | None
        substrate: (
            OneLayerSubstrate 
            | TwoLayerSubstrate
            | MultiLayerSubstrate
            | None
        )
        model: Isotropic | Step | ExponentialActivation
        if hasattr(sim_or_model, "model"):
            # object is a sim instance
            sim = sim_or_model #type: ignore
            surface = sim_or_model.surface #type: ignore
            substrate = sim_or_model.substrate #type: ignore
            model = sim_or_model.model #type: ignore
        else:
            # object is a model instance
            sim = None
            surface = None
            substrate = None
            model = sim_or_model
        if gma is None and hasattr(sim_or_model, "gma"):
            gma = sim_or_model.gma
        bold_name: str \
            = " ".join([
                (r"\bf{" + f"{piece_}" + r"}\,\,") #type: ignore
                for piece_ in name.split(" ")
            ])
        title_: str 
        if (
            sim is not None 
            and surface is not None
            and substrate is not None
            and model is not None
        ):
            title_ = (
                rf"${bold_name}$ "
                    + f"[{surface.label}/{substrate.label}/{model.label}]  "
            )
        else:
            title_ = rf"${bold_name}$ [{model.label}]  "
        if sim is not None:
            title_ = (
                title_ 
                + rf"  $\Delta x=${sim.Δx}"
                + rf"   $\Delta t=${sim.Δt:g}"
                + r"   $n_\mathrm{steps}=$"+rf"${sim.n_total}$"
                + "\n"
            )
        if sim is None:
            title_ = title_ + "\n"
        if gma is not None and gma.β_rs1 is not None:
            title_ = (
                title_
                + r"$\beta_\mathrm{rs1}=$" 
                    + f"{np.rad2deg(gma.β_rs1):3.1f}" + r"$\degree$   "
            )
        title_ = (
            title_
            + r"$\beta_\mathrm{t}=$" 
                + f"{np.rad2deg(model.β_t):2.2g}" + r"$\degree$"
        )
        if sim is not None:
            title_ = (
                title_ 
                + (
                    r"   $n_\xi=$" + f"{model.n_ξ}"
                    if model.label=="EA"
                    else r"   $\sigma_\xi=$" + f"{model.σ_ξ}"
                )
            )
        title_ = (
            title_
            + r"   $\xi_\mathrm{min}=$" + f"{model.ξ_min}"
            + r"   $\xi_\mathrm{max}=$" + f"{model.ξ_max}"
            + (
                r"   $\xi_\mathrm{trend}=$" + f"{model.ξ_trend}"
                if model.ξ_trend>0
                else ""
            )
        )
        if substrate is not None:
            if substrate.type==Substrates.L2:
                title_ = (
                    title_ 
                    + r"   $\eta_{\mathrm{ul}}=$" + f"{substrate.η_upperlayer}"
                )
            elif substrate.type==Substrates.L1:
                title_ = (
                    title_ 
                    + r"   $\eta=1$"
                )
        return title_