"""
Sympy symbols used in front propagation theory
----------------------------------------------

Requires Python packages/modules:
  -  :mod:`SymPy <sympy>`

---------------------------------------------------------------------
"""
# from sympy.physics.units.systems import SI
# from sympy.physics.units import length, time
from sympy import Symbol, MatrixSymbol, Function #type: ignore

Fstar: Symbol = Symbol(r"\mathcal{F}_*", real=True, positive=True)
H: Symbol = Symbol(r"\mathcal{H}", real=True, negative=False)
gstar: MatrixSymbol = MatrixSymbol(r"g_*", 2, 2)
rdotx: Symbol = Symbol(r"v^x", real=True, positive=True)
rdotz: Symbol = Symbol(r"v^z", real=True)
phi: Symbol = Symbol(r"\phi", real=True)
# phi_h: Symbol = Symbol(r"\phi", real=True)

p: Symbol = Symbol(r"p", real=True, positive=True)
px: Symbol = Symbol(r"p_x", real=True, positive=True)
pz: Symbol = Symbol(r"p_z", real=True, negative=True)
alpha: Symbol = Symbol(r"\alpha", real=True)
beta:  Symbol = Symbol(r"\beta", real=True, positive=True)
pzp: Symbol = Symbol(r"p_z^+", real=True, positive=True)
m_h: Symbol = Symbol(r"m", int=True, positive=True)
n: Symbol = Symbol(r"n", int=True, positive=True)

t_h: Symbol = Symbol(r"t", real=True, positive=True)
lamda = Symbol("lambda", positive=True)
alpha_crit: Symbol = Symbol(r"\alpha_{\mathrm{crit}}", real=True,)
beta_crit: Symbol = Symbol(r"\beta_c", real=True)
px_h: Symbol = Symbol(r"p_x", real=True, positive=True)
pz_h: Symbol = Symbol(r"p_z", real=True, negative=True)

xi_h0: Symbol = Symbol(r"\xi^{\perp_0}", real=True, positive=True)
xi_h: Function = Function(r"\xi^{\perp}", real=True, positive=True)

beta_h: Symbol = Symbol(r"\beta", real=True, positive=True)