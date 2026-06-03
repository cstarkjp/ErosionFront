"""
Numerical methods for simulation of erosion front propagation model 
-------------------------------------------------------------------

"""
# mypy: ignore-errors
import warnings
from dataclasses import dataclass
from collections.abc import Callable, Generator

import numpy as np
from numpy.typing import NDArray
from sympy import (  #type: ignore
    Eq,
    Mul,
    Add,
    Poly, Rational,
    lambdify,
    sqrt,
    collect,
    exp,
    tan,
    Abs,
    simplify,
    factor,
    numer,
    re, im,
    expand_log, log,
    numer, poly,
    Matrix,
)
from sympy import pi

from erosionfront.theory.symbols import (
    p, px, pz, alpha, beta, pzp, m_h, n, t_h, lamda,
    alpha_crit, beta_crit, px_h, pz_h, xi_h, xi_h0, phi, beta_h,
    Fstar, H, rdotx, rdotz, gstar,
)
from erosionfront.theory.equations import RichterSlopeTheory

warnings.filterwarnings("ignore")

__all__ = ["RichterSlopeModel", "make_rays"]


# @dataclass(frozen=True,) 
# class Model:
#     n: float | int = 5
#     phi_h: Rational = pi/7
#     xi_h0: float = 1

# @dataclass(frozen=True,)
# class Parameters:
#     model: Model = field(default_factory=Model)
    
# parameters = Parameters()
# (180*parameters.model.phi_h/pi).n()

@dataclass
class RichterSlopeModel:
    """
    Collection of numerical lambdas implementing Hamiltonian rock slope model.

    Attributes
    ----------
    n: int | float | Rational | None = None
    phi: float | None = None
    xi_h0: float | None = None
    sub: dict | None = None
    px_lambda: Callable | None = None
    pz_lambda: Callable | None = None
    p_lambda: Callable | None = None
    ξrt_lambda: Callable | None = None
    ξup_lambda: Callable | None = None
    ξrt_lambda: Callable | None = None
    ξx_lambda: Callable | None = None
    ξz_lambda: Callable | None = None
    vx_lambda: Callable | None = None
    vz_lambda: Callable | None = None
    v_lambda: Callable | None = None
    H_detHessian: Mul | None = None
    H_critical_t: Mul | None = None
    H_critical_t_numer: Add | None = None
    H_critical_t_poly: Poly | None = None
    tanbeta_crits: list | None = None
    beta_crits: list | None = None
    idtx_fgtx_markers: list | None = None
    gstar_eq: Eq | None = None
    """
    n: int | float | Rational | None = None
    phi: float | None = None
    xi_h0: float | None = None
    sub: dict | None = None
    px_lambda: Callable | None = None
    pz_lambda: Callable | None = None
    p_lambda: Callable | None = None
    ξrt_lambda: Callable | None = None
    ξup_lambda: Callable | None = None
    ξx_lambda: Callable | None = None
    ξz_lambda: Callable | None = None
    vx_lambda: Callable | None = None
    vz_lambda: Callable | None = None
    v_lambda: Callable | None = None
    H_detHessian: Mul | None = None
    H_critical_t: Mul | None = None
    H_critical_t_numer: Add | None = None
    H_critical_t_poly: Poly | None = None
    tanbeta_crits: list | None = None
    beta_crits: list | None = None
    idtx_fgtx_markers: list | None = None
    gstar_eq: Eq | None = None

    def build(self, rst: RichterSlopeTheory, parameters: dict,) -> None:
        """
        Lambdify set of Hamiltonian rock slope erosion model equations.

        Parameters
        ----------
        rst: RichterSlopeTheory
            Instance of Hamiltonian rock slope theory class.
        parameters: dict
            Choices of model parameters needed by theory equations.

        Attributes
        ----------
        n: int | float | Rational | None = None
        phi: float | None = None
        xi_h0: float | None = None
        sub: dict | None = None
        px_lambda: Callable | None = None
        pz_lambda: Callable | None = None
        p_lambda: Callable | None = None
        ξrt_lambda: Callable | None = None
        ξup_lambda: Callable | None = None
        ξrt_lambda: Callable | None = None
        ξx_lambda: Callable | None = None
        ξz_lambda: Callable | None = None
        vx_lambda: Callable | None = None
        vz_lambda: Callable | None = None
        v_lambda: Callable | None = None
        H_detHessian: Mul | None = None
        H_critical_t: Mul | None = None
        H_critical_t_numer: Add | None = None
        H_critical_t_poly: Poly | None = None
        tanbeta_crits: list | None = None
        beta_crits: list | None = None
        idtx_fgtx_markers: list | None = None
        gstar_eq: Eq | None = None
        """
        self.n = parameters["n"]
        self.phi = parameters["phi"]
        self.xi_h0 = parameters["xi_h0"]
        self.sub = {
            xi_h0: self.xi_h0, 
            n: self.n, 
            phi: self.phi
        }

        self.px_lambda = lambdify([beta], rst.p_covec_fn[0].subs(self.sub))
        self.pz_lambda = lambdify([beta], rst.p_covec_fn[1].subs(self.sub))
        self.p_lambda = lambdify([beta], rst.p_fn.subs(self.sub))
        
        self.ξrt_lambda = lambda beta_: 1/self.px_lambda(beta_)
        self.ξup_lambda = lambda beta_: 1/self.pz_lambda(beta_)

        self.ξx_lambda \
            = lambda beta_: self.px_lambda(beta_)/self.p_lambda(beta_)**2
        self.ξz_lambda \
            = lambda beta_: self.pz_lambda(beta_)/self.p_lambda(beta_)**2
        
        self.vx_lambda = lambdify(
            [beta],
            exp(expand_log(
                log(rst.v_vec_fn[0].subs({p:rst.p_fn}).subs(self.sub)),
            force=True)),
        )
        self.vz_lambda = lambdify(
            [beta],
            exp(expand_log(
                log(rst.v_vec_fn[1].subs({p:rst.p_fn}).subs(self.sub)),
            force=True)),
        )
        self.v_lambda = lambdify(
            [beta],
            sqrt(
                exp(2*expand_log(
                    log(rst.v_vec_fn[0].subs({p:rst.p_fn}).subs(self.sub)),
                force=True))
                +exp(2*expand_log(
                    log(rst.v_vec_fn[1].subs({p:rst.p_fn}).subs(self.sub)),
                force=True)),
            ),
        )

        self.H_detHessian = collect(
                (factor( 
                    (rst.d2Hdpx2*rst.d2Hdpz2 
                     - rst.d2Hdpxpz*rst.d2Hdpxpz).subs({pz:pzp}) )
                .subs({pzp:pz})
                .subs({n:parameters["n"]})
                .subs({px: -pz*tan(beta)})
                ), 
            pz
        )

        self.H_critical_t = (
            factor(
                self.H_detHessian
                .subs({
                    phi:parameters["phi"], 
                    xi_h0:1, 
                    px: -pz*tan(beta)
                })
            )
            # .subs({sqrt(1+tan(beta)**2):sec(beta)})
            # .subs({tan(beta):sqrt(sec(beta)**2-1)})
            # .subs({sec(beta):1/cos(beta)})
)
        self.H_critical_t_numer = numer(factor(
            self.H_critical_t
        ).subs({tan(beta):t_h}))#.args[1]

        self.gstar_eq = simplify(
            rst.gstar_eq.subs({px:-pz*tan(beta)}).subs(self.sub))

        try:
            self.H_critical_t_poly = poly(self.H_critical_t_numer,t_h)
        except:
            print("Failed to solve critical beta polynomial")
            return

        self.tanbeta_crits = sorted([
            float(root_)
            for root_ in self.H_critical_t_poly.nroots()
            if Abs(im(root_)) < 1e-10 and re(root_) > 0
        ])
        self.β_crits = [
            np.arctan(tanbeta_crit_) for tanbeta_crit_ in self.tanbeta_crits
        ]
                
    def get_gstar_signature(self,) -> Generator[tuple[list, list]]:
        """
        Generate metric signature from dual metric tensor as fn(beta).

        Attributes
        ----------
        Uses px_lambda, pz_lambda, and gstar_eq methods.

        Yields
        -------
        tuple[list,list]
            Lists of critical betas and corresponding metric signatures.
        """
        betas_: list
        for betas_ in list(zip([0]+self.β_crits, self.β_crits+[np.pi/2])):
            bias: float = 0.9
            beta_: float  = betas_[0]*(1-bias) + betas_[1]*bias
            px_: float = self.px_lambda(beta_)
            pz_: float = self.pz_lambda(beta_)
            gstar_: Matrix \
                = self.gstar_eq.rhs.subs({px:px_, pz:pz_}).subs({beta:beta_})
            gstar_eigenvals_: list = gstar_.eigenvals(multiple=True)
            gstar_signature_: list = [np.sign(ev_) for ev_ in gstar_eigenvals_]
            yield(betas_, gstar_signature_,)


def make_rays(
        rsm: RichterSlopeModel, 
        beta: NDArray,
    ) -> tuple[NDArray,NDArray]:
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
    # Just for the record, these are not simply 1/px, 1/pz
    vx: NDArray = rsm.vx_lambda(beta) 
    vz: NDArray = rsm.vz_lambda(beta)
    return (vx,vz,)