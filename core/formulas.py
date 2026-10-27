"""SymPy formulas for the Help dialogs.

The Black-Scholes part ports HW#6 (BlackScholes_Greeks_Sympy.ipynb) verbatim: prices are written with the
symbols d_1, d_2 and the custom normal CDF N / PDF n, and Greeks are derived with the chain rule.
Every entry of FORMULAS is a SymPy expression (or a zero-argument callable returning one) and
latex(name) renders it with sympy.latex for st.latex.
"""
from __future__ import annotations

from functools import lru_cache

import sympy as sp

# --------------------------------------------------------------- HW#6 engine (verbatim)
S, K, sigma, r, q, tau, t = sp.symbols('S K sigma r q tau t', positive=True)
d1, d2 = sp.symbols('d_1 d_2')


class N(sp.Function):      # normal CDF:  N'(x) = n(x)
    def fdiff(self, i=1):
        return n(self.args[0])


class n(sp.Function):      # normal PDF:  n'(x) = -x n(x),  n(-x) = n(x)
    def fdiff(self, i=1):
        return -self.args[0] * n(self.args[0])

    @classmethod
    def eval(cls, x):
        if x.could_extract_minus_sign():
            return n(-x)


D1 = (sp.log(S / K) + (r - q + sigma**2 / 2) * tau) / (sigma * sp.sqrt(tau))
D2 = D1 - sigma * sp.sqrt(tau)
LN = d1 * sigma * sp.sqrt(tau) - (r - q + sigma**2 / 2) * tau      # ln(S/K) written with d1

price = {'call': S * sp.exp(-q * tau) * N(d1) - K * sp.exp(-r * tau) * N(d2),
         'put': K * sp.exp(-r * tau) * N(-d2) - S * sp.exp(-q * tau) * N(-d1)}

var = {'S': S, 'K': K, 'sigma': sigma, 'r': r, 'q': q, 'tau': tau, 't': t}


def D(f, x):               # tau = T - t  =>  d/dt = -d/dtau
    return -sp.diff(f, tau) if x == t else sp.diff(f, x)


@lru_cache(maxsize=None)
def greek_expr(option='call', variable='S', order=1):
    """Symbolic Greek: d^order V / d variable^order, simplified as in HW#6."""
    x, f = var[variable], price[option]
    for _ in range(order):
        df = (D(f, x) + sp.diff(f, d1) * D(D1, x).subs(sp.log(S / K), LN)
              + sp.diff(f, d2) * D(D2, x).subs(sp.log(S / K), LN))
        g = sp.expand(df.subs(n(d2), S * sp.exp((r - q) * tau) * n(d1) / K))
        f = sp.collect(g, sorted(g.atoms(N, n), key=str), sp.factor)
    return f


# --------------------------------------------------------------- extra symbols
T, F, alpha, beta, rho, nu = sp.symbols('T F alpha beta rho nu', positive=True)
C, P, V, Vmkt, Csyn, Psyn, gap = sp.symbols('C P V V_{mkt} C_{syn} P_{syn} gap')
S_T, prem, Q, m, sign = sp.symbols('S_T premium Q m s')
u, d, p, Delta, B, Cu, Cd = sp.symbols('u d p Delta B C_u C_d')
dF, dsig, dW1, dW2, dt_ = sp.symbols(r'dF d\sigma dW_1 dW_2 dt')
z, x, sigB, sigATM = sp.symbols(r'z x sigma_B \sigma_{ATM}')
Nu, h, nn, Bd, Pi, Vd, Hd, Sd, DeltaW, GammaW, DeltaH, GammaH = sp.symbols(
    'N_u h_d n_d B_d Pi_d V_d H_d S_d Delta_w Gamma_w Delta_h Gamma_h')
h_prev, n_prev, B_prev = sp.symbols('h_{d-1} n_{d-1} B_{d-1}')
mu, Z, Sd1 = sp.symbols('mu Z S_{d+1}')
Fi, Fi1, sigi, sigi1, Z1, Z2, Npaths, j = sp.symbols(r'F_i F_{i+1} sigma_i \sigma_{i+1} Z_1 Z_2 N j')
FT = sp.Symbol('F_T^{(j)}')
PV, SE, s_hat = sp.symbols('PV SE s')
sig_mkt = sp.Function(r'\sigma_{mkt}')
sigma_B = sp.Function('sigma_B')
Ki = sp.Symbol('K_i')
VEGA = sp.Symbol(r'\mathcal{V}')
ARGMIN = sp.Symbol(r'\operatorname{argmin}')
STAR = sp.Symbol(r'(\alpha,\rho,\nu)^{*}')

omb = 1 - beta
lk = sp.log(F / K)
FK = (F * K) ** (omb / 2)
Z_expr = nu / alpha * FK * lk
X_expr = sp.log((sp.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
HAGAN = (alpha / (FK * (1 + omb**2 / 24 * lk**2 + omb**4 / 1920 * lk**4)) * (z / x)
         * (1 + (omb**2 / 24 * alpha**2 / FK**2 + rho * beta * nu * alpha / (4 * FK) + (2 - 3 * rho**2) / 24 * nu**2) * T))
HAGAN_B1 = alpha * (z / x) * (1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) / 24 * nu**2) * T)
HAGAN_ATM = (alpha / F**omb) * (1 + (omb**2 / 24 * alpha**2 / F**(2 * omb) + rho * beta * nu * alpha / (4 * F**omb)
                                     + (2 - 3 * rho**2) / 24 * nu**2) * T)
d1F = (sp.log(F / K) + sigma**2 * T / 2) / (sigma * sp.sqrt(T))

FORMULAS = {
    # Black-Scholes-Merton (HW#5 / HW#6)
    "bs.d1": sp.Eq(d1, D1),
    "bs.d2": sp.Eq(d2, D2),
    "bs.call": sp.Eq(C, price['call']),
    "bs.put": sp.Eq(P, price['put']),
    "bs.delta_call": lambda: sp.Eq(sp.Symbol('Delta_C'), greek_expr('call', 'S', 1)),
    "bs.delta_put": lambda: sp.Eq(sp.Symbol('Delta_P'), greek_expr('put', 'S', 1)),
    "bs.gamma": lambda: sp.Eq(sp.Symbol('Gamma'), greek_expr('call', 'S', 2)),
    "bs.vega": lambda: sp.Eq(VEGA, greek_expr('call', 'sigma', 1)),
    "bs.theta_call": lambda: sp.Eq(sp.Symbol('Theta_C'), greek_expr('call', 't', 1)),
    # Portfolio
    "pf.intrinsic_call": sp.Eq(sp.Symbol('IV_{call}'), sp.Max(S - K, 0)),
    "pf.intrinsic_put": sp.Eq(sp.Symbol('IV_{put}'), sp.Max(K - S, 0)),
    "pf.market_value": sp.Eq(sp.Symbol('MV'), sign * Q * m * Vmkt),
    "pf.time_value": sp.Eq(sp.Symbol('TV'), Vmkt - sp.Symbol('IV')),
    "pf.pnl": sp.Eq(sp.Symbol('PnL'), sign * Q * m * (Vmkt - prem)),
    "pf.payoff": sp.Eq(sp.Symbol('Payoff_T'), sign * Q * m * sp.Max(S_T - K, 0)),
    # Put-call parity and replication
    "parity.identity": sp.Eq(C - P, S * sp.exp(-q * T) - K * sp.exp(-r * T)),
    "parity.synthetic_call": sp.Eq(Csyn, P + S * sp.exp(-q * T) - K * sp.exp(-r * T)),
    "parity.synthetic_put": sp.Eq(Psyn, C - S * sp.exp(-q * T) + K * sp.exp(-r * T)),
    "parity.gap": sp.Eq(gap, sp.Symbol('C_{mkt}') - Csyn),
    "rep.bs": sp.Eq(V, Delta * S + B),
    "rep.bs_bond": sp.Eq(B, V - Delta * S),
    "rep.bin_delta": sp.Eq(Delta, (Cu - Cd) / (S * (u - d))),
    "rep.bin_bond": sp.Eq(B, (u * Cd - d * Cu) / ((u - d) * sp.exp(r * T))),
    "rep.bin_p": sp.Eq(p, (sp.exp(r * T) - d) / (u - d)),
    "rep.bin_value": sp.Eq(V, sp.exp(-r * T) * (p * Cu + (1 - p) * Cd)),
    "rep.crr": sp.Eq(u, sp.exp(sigma * sp.sqrt(T))),
    # Hedging
    "hedge.delta_shares": sp.Eq(h, Nu * DeltaW),
    "hedge.dg_n": sp.Eq(nn, Nu * GammaW / GammaH),
    "hedge.dg_shares": sp.Eq(h, Nu * DeltaW - nn * DeltaH),
    "hedge.cash0": sp.Eq(sp.Symbol('B_0'), Nu * prem - sp.Symbol('h_0') * sp.Symbol('S_0') - sp.Symbol('n_0') * sp.Symbol('H_0')),
    "hedge.cash": sp.Eq(Bd, B_prev * sp.exp(r * dt_) - (h - h_prev) * Sd - (nn - n_prev) * Hd),
    "hedge.portfolio_value": sp.Eq(Pi, -Nu * Vd + h * Sd + nn * Hd + Bd),
    "hedge.gbm_step": sp.Eq(Sd1, Sd * sp.exp((mu - sigma**2 / 2) * dt_ + sigma * sp.sqrt(dt_) * Z)),
    "hedge.tau": sp.Eq(sp.Symbol('tau_d'), sp.Max(T - sp.Symbol('d') * dt_, 0)),
    # SABR
    "sabr.sde_F": sp.Eq(dF, sigma * F**beta * dW1),
    "sabr.sde_sigma": sp.Eq(dsig, nu * sigma * dW2),
    "sabr.corr": sp.Eq(dW1 * dW2, rho * dt_),
    "sabr.z": sp.Eq(z, Z_expr),
    "sabr.x": sp.Eq(x, X_expr),
    "sabr.hagan_general": sp.Eq(sigB, HAGAN),
    "sabr.hagan_beta1": sp.Eq(sigB, HAGAN_B1),
    "sabr.hagan_atm": sp.Eq(sigATM, HAGAN_ATM),
    "sabr.forward": sp.Eq(F, S * sp.exp(r * T)),
    "sabr.black76": sp.Eq(C, sp.exp(-r * T) * (F * N(d1) - K * N(d2))),
    "sabr.black76_d1": sp.Eq(d1, d1F),
    "sabr.calibration_objective": sp.Eq(STAR, ARGMIN * sp.Sum((sig_mkt(Ki) - sigma_B(Ki, alpha, rho, nu))**2,
                                                              (sp.Symbol('i'), 1, sp.Symbol('n')))),
    # Monte Carlo
    "mc.cholesky": sp.Eq(dW2, rho * Z1 + sp.sqrt(1 - rho**2) * Z2),
    "mc.euler_F": sp.Eq(Fi1, Fi + sigi * sp.Max(Fi, 0)**beta * sp.sqrt(dt_) * Z1),
    "mc.euler_sigma": sp.Eq(sigi1, sigi + nu * sigi * sp.sqrt(dt_) * dW2),
    "mc.logeuler_F": sp.Eq(sp.log(Fi1), sp.log(Fi) - sigi**2 * dt_ / 2 + sigi * sp.sqrt(dt_) * Z1),
    "mc.logeuler_sigma": sp.Eq(sigi1, sigi * sp.exp(-nu**2 * dt_ / 2 + nu * sp.sqrt(dt_) * dW2)),
    "mc.pv": sp.Eq(PV, sp.exp(-r * T) / Npaths * sp.Sum(sp.Max(FT - K, 0), (j, 1, Npaths))),
    "mc.stderr": sp.Eq(SE, sp.exp(-r * T) * s_hat / sp.sqrt(Npaths)),
}


@lru_cache(maxsize=None)
def latex(name: str) -> str:
    """LaTeX string of a registered formula (rendered via sympy.latex)."""
    expr = FORMULAS[name]
    if callable(expr):
        expr = expr()
    return sp.latex(expr)


def names() -> list:
    return list(FORMULAS)
