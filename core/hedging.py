"""Delta and delta-gamma hedging: spot simulator, static HW#7 solver and the daily rebalancing loop.

Conventions: the trader has WRITTEN (is short) Q contracts x multiplier units of the option.
Daily clock dt = 1/252 (trading days); tau_d = max(T - d*dt, 0).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.bs import bs_greeks, bs_price, normalize_type

DT = 1.0 / 252.0


@dataclass
class HedgeResult:
    mode: str
    table: pd.DataFrame
    summary: dict
    trades: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# ------------------------------------------------------------------ spot simulator
def simulate_spot(S0: float, n_days: int, mode: str = "gbm", mu: float = 0.0, sigma: float = 0.2,
                  seed=None, manual=None, hist: pd.DataFrame | None = None, start: dt.date | None = None,
                  dt_: float = DT) -> pd.DataFrame:
    """Daily spot path: columns day (0..n_days), date, spot.

    gbm:        S_{d+1} = S_d exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)
    manual:     user supplied spots for days 1..n_days (day 0 = S0)
    historical: the last n_days+1 actual closes (DataFrame with columns date, close)
    """
    n_days = int(n_days)
    if n_days < 1:
        raise ValueError("n_days must be at least 1")
    if mode == "historical":
        if hist is None or len(hist) < n_days + 1:
            raise ValueError("historical mode needs n_days + 1 closes")
        h = hist.tail(n_days + 1).reset_index(drop=True)
        return pd.DataFrame({"day": np.arange(n_days + 1), "date": pd.to_datetime(h["date"]).dt.date,
                             "spot": h["close"].astype(float)})
    start = start or dt.date.today()
    dates = pd.bdate_range(pd.Timestamp(start), periods=n_days + 1).date
    if mode == "manual":
        vals = [float(v) for v in (manual or [])][:n_days]
        if len(vals) < n_days or any(v <= 0 for v in vals):
            raise ValueError(f"manual mode needs {n_days} positive spot values")
        spots = np.array([float(S0)] + vals)
    elif mode == "gbm":
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n_days)
        log_ret = (mu - 0.5 * sigma**2) * dt_ + sigma * np.sqrt(dt_) * z
        spots = float(S0) * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
    else:
        raise ValueError(f"unknown simulator mode {mode}")
    return pd.DataFrame({"day": np.arange(n_days + 1), "date": dates, "spot": spots})


# ------------------------------------------------------------------ static hedge (HW#7)
def solve_static_hedge(S, K, T, r, sigma, kind_w, Q, hedge_opts, mode="delta_gamma", q=0.0) -> dict:
    """HW#7 one-shot hedge of Q written options with n hedge options (least squares) and stock for the rest.

    hedge_opts: list of (kind, K, T). Returns q (option quantities, +buy / -sell), stock (shares, +buy),
    the greeks and the residual portfolio delta / gamma.
    """
    gw = bs_greeks(S, K, T, r, q, sigma, kind_w)
    dw, gmw = gw["delta"], gw["gamma"]
    greeks = [bs_greeks(S, k, t, r, q, sigma, c) for c, k, t in hedge_opts]
    D = np.array([g["delta"] for g in greeks])
    G = np.array([g["gamma"] for g in greeks])
    if len(hedge_opts) == 0:
        qv = np.zeros(0)
    else:
        A = np.vstack([D]) if mode == "delta" else np.vstack([D, G])
        b = np.array([Q * dw]) if mode == "delta" else np.array([Q * dw, Q * gmw])
        qv = np.linalg.lstsq(A, b, rcond=None)[0]
    port_delta = -Q * dw + (D @ qv if len(qv) else 0.0)
    port_gamma = -Q * gmw + (G @ qv if len(qv) else 0.0)
    stock = -port_delta
    return {"q": qv, "stock": float(stock), "written": {"delta": dw, "gamma": gmw},
            "hedge": [{"kind": normalize_type(c), "K": k, "T": t, "delta": g["delta"], "gamma": g["gamma"]}
                      for (c, k, t), g in zip(hedge_opts, greeks)],
            "port_delta": float(port_delta + stock), "port_gamma": float(port_gamma)}


# ------------------------------------------------------------------ daily rebalancing loop
def simulate_hedge(spot_path: pd.DataFrame, K, T, r, sigma, option_type, Q, multiplier: float = 100,
                   mode: str = "delta", hedge: dict | None = None, premium: float | None = None,
                   q: float = 0.0, dt_: float = DT) -> HedgeResult:
    """Self-financing daily hedge of a SHORT position in Q x multiplier units of a European option.

    mode: 'none' | 'delta' | 'delta_gamma' (needs hedge = dict(K, T, option_type)).
    premium: per-unit price received at day 0 (default: the Black-Scholes price).
    Bookkeeping (per day d, N_u = Q*multiplier, g = e^{r dt}):
      targets    delta:  h_d = N_u Delta_w ;   delta-gamma: n_d = N_u Gamma_w / Gamma_h,  h_d = N_u Delta_w - n_d Delta_h
      cash       B_0 = N_u * premium - h_0 S_0 - n_0 H_0 ;  B_d = B_{d-1} g - (h_d - h_{d-1}) S_d - (n_d - n_{d-1}) H_d
      value      Pi_d = -N_u V_d + h_d S_d + n_d H_d + B_d       (Pi_0 = 0 when premium = V_0)
    """
    kind = normalize_type(option_type)
    if mode not in ("none", "delta", "delta_gamma"):
        raise ValueError(f"unknown hedge mode {mode}")
    if mode == "delta_gamma" and not hedge:
        raise ValueError("delta_gamma mode needs a hedge option (K, T, option_type)")
    N_u = float(Q) * float(multiplier)
    g = float(np.exp(r * dt_))
    rows, trades, warnings = [], [], []
    h_prev = n_prev = 0.0
    B_prev = Pi_prev = 0.0
    hk = normalize_type(hedge["option_type"]) if hedge else None
    V = 0.0
    for i, rec in enumerate(spot_path.itertuples(index=False)):
        day, date, S = int(rec.day), rec.date, float(rec.spot)
        tau = max(T - day * dt_, 0.0)
        V = float(bs_price(S, K, tau, r, q, sigma, kind))
        gr = bs_greeks(S, K, tau, r, q, sigma, kind)
        dw, gmw = gr["delta"], gr["gamma"]
        if hedge:
            tau_h = max(float(hedge["T"]) - day * dt_, 0.0)
            H = float(bs_price(S, hedge["K"], tau_h, r, q, sigma, hk))
            gh = bs_greeks(S, hedge["K"], tau_h, r, q, sigma, hk)
            dh, gmh = gh["delta"], gh["gamma"]
        else:
            H, dh, gmh = 0.0, 0.0, 0.0
        if mode == "none":
            n, h = 0.0, 0.0
        elif mode == "delta":
            n, h = 0.0, N_u * dw
        else:
            if gmh < 1e-10:
                n = n_prev
                warnings.append(f"Day {day}: hedge option gamma is ~0, hedge-option quantity kept at {n:.4f}.")
            else:
                n = N_u * gmw / gmh
            h = N_u * dw - n * dh
        dh_tr, dn_tr = h - h_prev, n - n_prev
        if i == 0:
            prem = V if premium is None else float(premium)
            B = N_u * prem - h * S - n * H
        else:
            B = B_prev * g - dh_tr * S - dn_tr * H
        Pi = -N_u * V + h * S + n * H + B
        daily = Pi - Pi_prev if i else Pi
        rows.append({
            "day": day, "date": date, "spot": S, "tau": tau, "option_price": V, "delta": dw, "gamma": gmw,
            "hedge_opt_price": H, "hedge_delta": dh, "hedge_gamma": gmh,
            "shares_held": h, "shares_traded": dh_tr, "hedge_opt_held": n, "hedge_opt_traded": dn_tr,
            "cash": B, "option_liability": -N_u * V, "portfolio_value": Pi, "daily_pnl": daily, "cum_pnl": Pi,
            "port_delta": -N_u * dw + n * dh + h, "port_gamma": -N_u * gmw + n * gmh,
        })
        parts = []
        if abs(dh_tr) > 1e-9:
            parts.append(f"{'BUY' if dh_tr > 0 else 'SELL'} {abs(dh_tr):,.2f} shares @ {S:,.2f}")
        if abs(dn_tr) > 1e-9:
            parts.append(f"{'BUY' if dn_tr > 0 else 'SELL'} {abs(dn_tr):,.2f} hedge {hk}s @ {H:,.2f}")
        if not parts:
            parts.append("no trade")
        label = "Initial hedge" if i == 0 else "Rebalance"
        trades.append(f"Day {day} ({date}): {label}: " + "; ".join(parts) + f"; cash {B:,.2f}; cum P&L {Pi:,.2f}")
        h_prev, n_prev, B_prev, Pi_prev = h, n, B, Pi
    table = pd.DataFrame(rows)
    first_V = float(table["option_price"].iloc[0]) if len(table) else 0.0
    summary = {
        "mode": mode, "units": N_u, "final_cum_pnl": float(table["cum_pnl"].iloc[-1]),
        "max_abs_daily_pnl": float(table["daily_pnl"].abs().max()),
        "total_shares_traded": float(table["shares_traded"].abs().sum()),
        "total_hedge_options_traded": float(table["hedge_opt_traded"].abs().sum()),
        "final_port_delta": float(table["port_delta"].iloc[-1]), "final_port_gamma": float(table["port_gamma"].iloc[-1]),
        "premium_received": float(N_u * (first_V if premium is None else premium)),
    }
    return HedgeResult(mode, table, summary, trades, warnings)


def compare_strategies(spot_path: pd.DataFrame, K, T, r, sigma, option_type, Q, multiplier=100,
                       hedge: dict | None = None, premium=None, q=0.0, dt_=DT) -> pd.DataFrame:
    """Cumulative P&L per day for unhedged, delta and (if a hedge option is given) delta-gamma hedging."""
    out = spot_path[["day", "date", "spot"]].copy()
    for mode, col in (("none", "pnl_unhedged"), ("delta", "pnl_delta"), ("delta_gamma", "pnl_delta_gamma")):
        if mode == "delta_gamma" and not hedge:
            continue
        res = simulate_hedge(spot_path, K, T, r, sigma, option_type, Q, multiplier, mode, hedge, premium, q, dt_)
        out[col] = res.table["cum_pnl"].to_numpy()
    return out
