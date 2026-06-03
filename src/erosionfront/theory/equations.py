"""
Sympy implementation of theory of erosion front propagation model
-----------------------------------------------------------------

"""

import warnings
from dataclasses import dataclass
from typing import Sequence, Any, Generator

from sympy import (                     #type: ignore
    Eq, Mul, Matrix, MutableDenseMatrix, ImmutableDenseMatrix, Rational,
    sin, cos, exp, tan, sqrt, diff, 
    solve, trigsimp, simplify, powsimp, factor,
)

from erosionfront.misc.utils import e2d
from erosionfront.theory.symbols import (
    p, px, pz, alpha, beta, pzp, m_h, n, t_h, lamda,
    alpha_crit, beta_crit, px_h, pz_h, xi_h, xi_h0, phi, beta_h,
    Fstar, H, rdotx, rdotz, gstar,
)

warnings.filterwarnings("ignore")

__all__ = ["RichterSlopeTheory"]

@dataclass
class RichterSlopeTheory:
    """
    Development of Hamiltonian rock slope erosion model using Sympy.

    Attributes
    ----------
    hillslope_tan_erosion_eq: Eq | None = None
    hillslope_erosion_eq: Eq | None = None
    xih_pxpz_eq: Eq | None = None
    hillslope_pxpz_eq: Eq | None = None
    Fstar_hillslope_pxpz_eq: Eq | None = None
    H_hillslope_pxpz_eq: Eq | None = None
    dHdpx: Mul | None = None
    dHdpz: Mul | None = None
    v_pxpz_eq: Eq | None = None
    v_p_beta_eq: Eq | None = None
    tanalpha_eq: Eq | None = None
    d2Hdpx2: Mul | None = None
    d2Hdpxpz: Mul | None = None
    d2Hdpz2: Mul | None = None
    gstar_eq: Eq | None = None
    fgtx_raw_eq: Eq | None = None
    fgtx_p_beta_eq: Eq | None = None
    p_covec_fn: MutableDenseMatrix | None = None
    p_fn: exp | Any | None = None
    p_conjugate_beta_eq: Eq | None = None
    v_vec_fn: ImmutableDenseMatrix | None = None
    """
    hillslope_tan_erosion_eq: Eq | None = None
    hillslope_erosion_eq: Eq | None = None
    xih_pxpz_eq: Eq | None = None
    hillslope_pxpz_eq: Eq | None = None
    Fstar_hillslope_pxpz_eq: Eq | None = None
    H_hillslope_pxpz_eq: Eq | None = None
    dHdpx: Mul | None = None
    dHdpz: Mul | None = None
    v_pxpz_eq: Eq | None = None
    v_p_beta_eq: Eq | None = None
    tanalpha_eq: Eq | None = None
    d2Hdpx2: Mul | None = None
    d2Hdpxpz: Mul | None = None
    d2Hdpz2: Mul | None = None
    gstar_eq: Eq | None = None
    fgtx_raw_eq: Eq | None = None
    fgtx_p_beta_eq: Eq | None = None
    p_covec_fn: MutableDenseMatrix | None = None
    p_fn: exp | Any | None = None
    p_conjugate_beta_eq: Eq | None = None
    v_vec_fn: ImmutableDenseMatrix | None = None

    def __post_init__(self) -> None:
        """
        Solve Hamiltonian rock slope erosion model equations using Sympy.

        Attributes
        ----------
        hillslope_tan_erosion_eq: Eq | None = None
        hillslope_erosion_eq: Eq | None = None
        xih_pxpz_eq: Eq | None = None
        hillslope_pxpz_eq: Eq | None = None
        Fstar_hillslope_pxpz_eq: Eq | None = None
        H_hillslope_pxpz_eq: Eq | None = None
        dHdpx: Mul | None = None
        dHdpz: Mul | None = None
        v_pxpz_eq: Eq | None = None
        v_p_beta_eq: Eq | None = None
        tanalpha_eq: Eq | None = None
        d2Hdpx2: Mul | None = None
        d2Hdpxpz: Mul | None = None
        d2Hdpz2: Mul | None = None
        gstar_eq: Eq | None = None
        fgtx_raw_eq: Eq | None = None
        fgtx_p_beta_eq: Eq | None = None
        p_covec_fn: MutableDenseMatrix | None = None
        p_fn: exp | Any | None = None
        p_conjugate_beta_eq: Eq | None = None
        v_vec_fn: ImmutableDenseMatrix | None = None
        """
        self.hillslope_tan_erosion_eq = Eq(
            xi_h(beta), 
            xi_h0 * (
                0 +
                (
                    (exp(-(tan(phi*1) / tan(beta))**n) ) 
                    +
                    (exp(-(tan(phi*2*Rational(4,5)) / tan(beta))**n) ) * 0
                )
            )/1
        )
        self.hillslope_erosion_eq = self.hillslope_tan_erosion_eq

        self.xih_pxpz_eq = Eq(xi_h(beta), 1/sqrt(px**2+pz**2))

        self.hillslope_pxpz_eq = (
            self.hillslope_erosion_eq
                .subs(e2d(self.xih_pxpz_eq))
                .subs({tan(beta):(-px/pz)})
                .subs({tan(beta/2):sin(beta)/(1+cos(beta))})
                .subs({sin(beta):(px/sqrt(px**2+pz**2))})
                .subs({cos(beta):(-pz/sqrt(px**2+pz**2))})
        )

        self.Fstar_hillslope_pxpz_eq = Eq(
            Fstar,
            solve(simplify(
                    self.hillslope_pxpz_eq.subs({px: px/Fstar, pz: pz/Fstar})
                ), Fstar)[
                0
            ],
        )

        self.H_hillslope_pxpz_eq \
            = Eq(H, self.Fstar_hillslope_pxpz_eq.rhs**2 / 2)
        
        self.dHdpx = simplify(diff(self.H_hillslope_pxpz_eq.rhs,px))
        self.dHdpz = simplify(diff(self.H_hillslope_pxpz_eq.rhs,pz))

        self.v_pxpz_eq = Eq(
            Matrix([rdotx, rdotz]),
            Matrix(
                [
                    factor(self.dHdpx),
                    factor(self.dHdpz),
                ]
            )
        )

        self.v_p_beta_eq = (
            trigsimp( self.v_pxpz_eq.subs({px:p*sin(beta), pz:-p*cos(beta)}) )
        )

        self.tanalpha_eq = simplify(Eq(
            tan(alpha),
            trigsimp( self.v_p_beta_eq.rhs[1]/self.v_p_beta_eq.rhs[0] )
        ))

        self.d2Hdpx2 = simplify(diff(self.dHdpx,px))
        self.d2Hdpxpz = simplify(diff(self.dHdpx,pz))
        self.d2Hdpz2 = simplify(diff(self.dHdpz,pz))

        self.gstar_eq = Eq(
            gstar, 
            Matrix([
                [self.d2Hdpx2, self.d2Hdpxpz], 
                [self.d2Hdpxpz,self.d2Hdpz2]
            ])
        )

        self.fgtx_raw_eq = simplify(
            self.Fstar_hillslope_pxpz_eq
                .subs({Fstar:1})
                .subs({px:p*sin(beta), pz:-p*cos(beta)})
        )

        self.fgtx_p_beta_eq = Eq(
            p,
            solve( self.fgtx_raw_eq, p )[0]
        )

        self.p_covec_fn = Matrix([
            (self.fgtx_p_beta_eq.rhs*sin(beta)),
            (-self.fgtx_p_beta_eq.rhs*cos(beta))
        ])
        self.p_fn = self.fgtx_p_beta_eq.rhs

        self.p_conjugate_beta_eq = Eq(
            p,
            powsimp(solve(
                simplify(
                    Eq(
                        (self.v_p_beta_eq.rhs[0] * px 
                         + self.v_p_beta_eq.rhs[1] * pz)
                         .subs({px: p*sin(beta), pz: -p*cos(beta)}
                        ),
                        1,
                    )
                ),
                p,
            )[1]),
        )

        self.v_vec_fn =  (self.v_p_beta_eq.rhs)
