"""Arbitrage detection with synthetic options built from the stock and a bond.

Three constructions:
  * put-call parity:  C = P + S e^{-qT} - K e^{-rT}          (model-free, riskless)
  * Black-Scholes replication:  V = Delta * S + B           (model-dependent, needs rebalancing)
  * one-period binomial replication (HW#4 u, d):  V = Delta * S + B (riskless inside the model)
Each returns an ArbResult with a verdict, a numbered step list and a cash-flow table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.bs import bs_delta, bs_price, intrinsic, normalize_type


@dataclass
class ArbResult:
    mode: str
    verdict: str
    is_arbitrage: bool
    mispricing: float            # per unit, market - fair
    profit_t0: float             # total locked-in profit today (units applied)
    profit_T: float              # the same amount carried to T at the risk-free rate
    steps: list = field(default_factory=list)
    cashflows: pd.DataFrame = field(default_factory=pd.DataFrame)
    details: dict = field(default_factory=dict)


def _money(x: float) -> str:
    return f"{x:,.4f}"


def _table(legs: list, columns: list) -> pd.DataFrame:
    """legs: (name, cf_t0, cf_scen1, cf_scen2, ...) -> DataFrame with a Total row."""
    df = pd.DataFrame(legs, columns=["Leg"] + columns)
    total = ["Total"] + [float(df[c].sum()) for c in columns]
    return pd.concat([df, pd.DataFrame([total], columns=df.columns)], ignore_index=True)


def ud_from_sigma(sigma: float, T: float) -> tuple:
    """Cox-Ross-Rubinstein moves u = e^{sigma sqrt(T)}, d = 1/u (HW#4 macro)."""
    u = float(np.exp(sigma * np.sqrt(T)))
    return u, 1.0 / u


# ------------------------------------------------------------------ put-call parity
def put_call_parity(S, K, T, r, q, call_mkt, put_mkt, threshold: float = 0.0, units: float = 100) -> ArbResult:
    disc_K = K * np.exp(-r * T)
    stock_leg = S * np.exp(-q * T)
    shares = float(np.exp(-q * T))
    C_syn = put_mkt + stock_leg - disc_K
    P_syn = call_mkt - stock_leg + disc_K
    gap = float(call_mkt - C_syn)                     # = -(put_mkt - P_syn)
    details = {"synthetic_call": float(C_syn), "synthetic_put": float(P_syn), "pv_strike": float(disc_K),
               "stock_leg": float(stock_leg), "shares": shares, "gap": gap, "threshold": threshold}
    S_lo, S_hi = 0.8 * K, 1.2 * K
    cols = ["t = 0", f"T: S_T < K (S_T = {S_lo:,.2f})", f"T: S_T >= K (S_T = {S_hi:,.2f})"]

    if abs(gap) <= threshold:
        verdict = (f"No arbitrage: the market call ({_money(call_mkt)}) is within {_money(threshold)} of the synthetic "
                   f"call ({_money(C_syn)}) built from the put, the stock and a bond.")
        legs = [("Market call", call_mkt, 0.0, 0.0), ("Synthetic call (P + S e^-qT - K e^-rT)", -C_syn, 0.0, 0.0)]
        return ArbResult("parity", verdict, False, gap, 0.0, 0.0,
                         ["Put-call parity holds within the threshold. No trade is required."],
                         _table(legs, cols), details)

    if gap > 0:   # call rich / put cheap: sell the call, buy the synthetic call
        verdict = (f"Call OVERPRICED by {_money(gap)} per share versus the synthetic call "
                   f"({_money(C_syn)}): sell the call and buy the synthetic call (long put + long stock + borrowing).")
        steps = [
            f"Sell 1 call at the market price {_money(call_mkt)} (cash in).",
            f"Buy 1 put at the market price {_money(put_mkt)} (cash out).",
            f"Buy {shares:.4f} share(s) at {_money(S)} = {_money(stock_leg)} (dividends reinvested give exactly 1 share at T).",
            f"Borrow the present value of the strike, K e^(-rT) = {_money(disc_K)}, at r = {r:.4%} until T; repay K = {_money(K)} at T.",
            "At T, if S_T >= K: the call is assigned, deliver the share and receive K; use K to repay the loan. Net = 0.",
            "At T, if S_T < K: exercise the put, deliver the share and receive K; use K to repay the loan. Net = 0.",
            f"Riskless profit locked in today: {_money(gap)} per share, i.e. {_money(gap * units)} for {units:g} shares "
            f"(worth {_money(gap * units * np.exp(r * T))} at T if invested at r).",
        ]
        legs = [
            ("Short call", call_mkt, 0.0, -(S_hi - K)),
            ("Long put", -put_mkt, K - S_lo, 0.0),
            (f"Long {shares:.4f} share(s)", -stock_leg, S_lo, S_hi),
            ("Borrow K e^-rT", disc_K, -K, -K),
        ]
    else:         # call cheap / put rich: buy the call, sell the synthetic call
        g = -gap
        verdict = (f"Call UNDERPRICED by {_money(g)} per share versus the synthetic call "
                   f"({_money(C_syn)}) (equivalently the put is overpriced): buy the call and sell the synthetic call "
                   f"(short put + short stock + lending).")
        steps = [
            f"Buy 1 call at the market price {_money(call_mkt)} (cash out).",
            f"Sell 1 put at the market price {_money(put_mkt)} (cash in).",
            f"Short-sell {shares:.4f} share(s) at {_money(S)} = {_money(stock_leg)} (you owe exactly 1 share at T).",
            f"Lend the present value of the strike, K e^(-rT) = {_money(disc_K)}, at r = {r:.4%} until T; receive K = {_money(K)} at T.",
            "At T, if S_T >= K: exercise the call, pay K (from the loan repayment) and receive the share to close the short. Net = 0.",
            "At T, if S_T < K: the put is assigned, pay K (from the loan repayment) and receive the share to close the short. Net = 0.",
            f"Riskless profit locked in today: {_money(g)} per share, i.e. {_money(g * units)} for {units:g} shares "
            f"(worth {_money(g * units * np.exp(r * T))} at T if invested at r).",
        ]
        legs = [
            ("Long call", -call_mkt, 0.0, S_hi - K),
            ("Short put", put_mkt, -(K - S_lo), 0.0),
            (f"Short {shares:.4f} share(s)", stock_leg, -S_lo, -S_hi),
            ("Lend K e^-rT", -disc_K, K, K),
        ]
    profit = abs(gap) * units
    return ArbResult("parity", verdict, True, gap, profit, profit * float(np.exp(r * T)), steps, _table(legs, cols), details)


# ------------------------------------------------------------------ replication helpers
def _replication_result(mode, S, K, T, r, option_type, market_price, V_theo, delta, B, threshold, units,
                        scen_labels, scen_S, scen_option, details) -> ArbResult:
    """Shared verdict / steps / cash-flow builder for the Delta-shares + bond replications."""
    kind = normalize_type(option_type)
    gap = float(market_price - V_theo)
    growth = float(np.exp(r * T))
    details.update({"fair_value": float(V_theo), "delta": float(delta), "bond": float(B), "gap": gap,
                    "threshold": threshold})
    cols = ["t = 0"] + list(scen_labels)
    caveat = ("Model-dependent: the replication is exact only with continuous rebalancing at the assumed volatility."
              if mode == "bs_replication" else "Exactly riskless inside the one-period binomial model.")
    details["caveat"] = caveat

    if abs(gap) <= threshold:
        verdict = (f"No arbitrage: the market {kind} ({_money(market_price)}) is within {_money(threshold)} of the "
                   f"replicating portfolio value ({_money(V_theo)}).")
        zeros = [0.0] * len(scen_S)
        legs = [tuple([f"Market {kind}", market_price] + zeros),
                tuple(["Replicating portfolio (Delta S + B)", -V_theo] + zeros)]
        return ArbResult(mode, verdict, False, gap, 0.0, 0.0,
                         [f"The {kind} is fairly priced within the threshold. No trade is required."],
                         _table(legs, cols), details)

    share_word = "Buy" if delta > 0 else "Short-sell"
    bond_word = "Borrow" if B < 0 else "Lend"
    rep_T = [delta * s + B * growth for s in scen_S]      # static replica value at T
    if gap > 0:  # option overpriced: sell option, buy replica
        verdict = (f"{kind.capitalize()} OVERPRICED by {_money(gap)} per share versus its replicating portfolio "
                   f"({_money(V_theo)} = {delta:.4f} shares + bond {_money(B)}): sell the {kind}, buy the synthetic {kind}.")
        steps = [
            f"Sell 1 {kind} at the market price {_money(market_price)} (cash in).",
            f"{share_word} {abs(delta):.4f} share(s) at {_money(S)} (= {_money(abs(delta) * S)}) to replicate the delta of the {kind}.",
            f"{bond_word} {_money(abs(B))} at r = {r:.4%} until T (the bond leg of the synthetic {kind}).",
            f"Net cash today = {_money(market_price)} - {_money(delta * S)} - ({_money(B)}) = {_money(gap)} per share.",
            ("Keep the replica delta-neutral: rebalance the share holding as S and time change (see the Hedging calculator)."
             if mode == "bs_replication" else
             "At T the replica pays exactly the option payoff in both nodes, so the short option is covered."),
            f"Profit: {_money(gap)} per share, {_money(gap * units)} for {units:g} shares. {caveat}",
        ]
        legs = [(f"Short {kind}", market_price, *[-v for v in scen_option]),
                (f"{'Long' if delta > 0 else 'Short'} {abs(delta):.4f} share(s)", -delta * S, *[delta * s for s in scen_S]),
                (f"{'Borrow' if B < 0 else 'Lend'} {_money(abs(B))}", -B, *[B * growth] * len(scen_S))]
    else:
        g = -gap
        verdict = (f"{kind.capitalize()} UNDERPRICED by {_money(g)} per share versus its replicating portfolio "
                   f"({_money(V_theo)} = {delta:.4f} shares + bond {_money(B)}): buy the {kind}, sell the synthetic {kind}.")
        anti_share = "Short-sell" if delta > 0 else "Buy"
        anti_bond = "Lend" if B < 0 else "Borrow"
        steps = [
            f"Buy 1 {kind} at the market price {_money(market_price)} (cash out).",
            f"{anti_share} {abs(delta):.4f} share(s) at {_money(S)} (= {_money(abs(delta) * S)}): the opposite of the share leg of the replica.",
            f"{anti_bond} {_money(abs(B))} at r = {r:.4%} until T: the opposite of the bond leg of the replica.",
            f"Net cash today = {_money(delta * S)} + ({_money(B)}) - {_money(market_price)} = {_money(g)} per share.",
            ("Keep the position delta-neutral by rebalancing the share leg as S and time change (see the Hedging calculator)."
             if mode == "bs_replication" else
             "At T the long option pays exactly what the short replica owes in both nodes."),
            f"Profit: {_money(g)} per share, {_money(g * units)} for {units:g} shares. {caveat}",
        ]
        legs = [(f"Long {kind}", -market_price, *scen_option),
                (f"{'Short' if delta > 0 else 'Long'} {abs(delta):.4f} share(s)", delta * S, *[-delta * s for s in scen_S]),
                (f"{'Lend' if B < 0 else 'Borrow'} {_money(abs(B))}", B, *[-B * growth] * len(scen_S))]
    details["replica_terminal"] = rep_T
    profit = abs(gap) * units
    return ArbResult(mode, verdict, True, gap, profit, profit * growth, steps, _table(legs, cols), details)


def replicate_bs(S, K, T, r, q, sigma, option_type, market_price, threshold: float = 0.0, units: float = 100) -> ArbResult:
    """Black-Scholes replication: V = Delta S + B with Delta = dV/dS, B = V - Delta S."""
    kind = normalize_type(option_type)
    V = float(bs_price(S, K, T, r, q, sigma, kind))
    delta = float(bs_delta(S, K, T, r, q, sigma, kind))
    B = V - delta * S
    scen_S = [0.9 * S, S, 1.1 * S]
    labels = [f"T (static replica): S_T = {s:,.2f}" for s in scen_S]
    scen_opt = [float(intrinsic(s, K, kind)) for s in scen_S]
    details = {"sigma": sigma, "q": q, "model": "Black-Scholes-Merton"}
    return _replication_result("bs_replication", S, K, T, r, kind, market_price, V, delta, B, threshold, units,
                               labels, scen_S, scen_opt, details)


def replicate_binomial(S, K, T, r, u, d, option_type, market_price, threshold: float = 0.0, units: float = 100) -> ArbResult:
    """One-period binomial replication (HW#4): Delta = (Cu-Cd)/(S(u-d)), B = (u Cd - d Cu)/((u-d)e^{rT})."""
    kind = normalize_type(option_type)
    growth = float(np.exp(r * T))
    if not (d < growth < u):
        raise ValueError(f"No-arbitrage condition d < e^(rT) < u violated: d={d:.4f}, e^(rT)={growth:.4f}, u={u:.4f}. "
                         "Adjust u, d or r.")
    Su, Sd = S * u, S * d
    Cu, Cd = float(intrinsic(Su, K, kind)), float(intrinsic(Sd, K, kind))
    delta = (Cu - Cd) / (S * (u - d))
    B = (u * Cd - d * Cu) / ((u - d) * growth)
    p = (growth - d) / (u - d)
    V = delta * S + B
    details = {"u": u, "d": d, "p": p, "Su": Su, "Sd": Sd, "Cu": Cu, "Cd": Cd, "model": "one-period binomial",
               "risk_neutral_value": float(np.exp(-r * T) * (p * Cu + (1 - p) * Cd))}
    labels = [f"T up node (S_T = {Su:,.2f})", f"T down node (S_T = {Sd:,.2f})"]
    return _replication_result("binomial", S, K, T, r, kind, market_price, V, delta, B, threshold, units,
                               labels, [Su, Sd], [Cu, Cd], details)
