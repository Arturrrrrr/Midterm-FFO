"""Black-Scholes-Merton pricing, Greeks, implied volatility and put-call parity.

Ported from HW#5 (Greeks_Dashboard.ipynb): vectorised numpy with a continuous
dividend yield q.  All functions accept scalars or numpy arrays and return a
float when every input is a scalar.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "year_fraction", "normalize_type", "d1_d2", "intrinsic", "bs_price", "bs_greeks",
    "bs_delta", "bs_gamma", "implied_vol", "parity_gap", "put_call_parity", "black76_price",
]


def normalize_type(option_type) -> str:
    """Map Call/C/call -> 'call' and Put/P/put -> 'put'."""
    s = str(option_type).strip().lower()
    if s in ("c", "call", "calls"):
        return "call"
    if s in ("p", "put", "puts"):
        return "put"
    raise ValueError(f"Unknown option type: {option_type}")


def year_fraction(expiry, asof=None) -> float:
    """ACT/365 year fraction between asof (default today) and expiry, floored at 0."""
    asof = asof or dt.date.today()
    if hasattr(expiry, "to_pydatetime"):          # pandas Timestamp
        expiry = expiry.to_pydatetime()
    if isinstance(expiry, dt.datetime):
        expiry = expiry.date()
    if isinstance(asof, dt.datetime):
        asof = asof.date()
    return max((expiry - asof).days, 0) / 365.0


def _ret(x):
    """Return a python float for 0-d results, the array otherwise."""
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def d1_d2(S, K, tau, r, q, sigma):
    """d1 and d2 of Black-Scholes-Merton (HW#5)."""
    S, K, tau, sigma = (np.asarray(v, dtype=float) for v in (S, K, tau, sigma))
    tau = np.maximum(tau, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2


def intrinsic(S, K, option_type="call"):
    """Intrinsic value max(S-K,0) for a call, max(K-S,0) for a put."""
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    if normalize_type(option_type) == "call":
        return _ret(np.maximum(S - K, 0.0))
    return _ret(np.maximum(K - S, 0.0))


def bs_price(S, K, tau, r, q, sigma, option_type="call"):
    """European option price under Black-Scholes-Merton with dividend yield q.

    C = S e^{-q tau} N(d1) - K e^{-r tau} N(d2)
    P = K e^{-r tau} N(-d2) - S e^{-q tau} N(-d1)
    At tau <= 0 the intrinsic value is returned.
    """
    kind = normalize_type(option_type)
    S, K, tau, sigma = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, K, tau, sigma)))
    d1, d2 = d1_d2(S, K, tau, r, q, sigma)
    if kind == "call":
        price = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
    price = np.where(tau <= 0, intrinsic(S, K, kind), price)
    return _ret(price)


def bs_greeks(S, K, tau, r, q, sigma, option_type="call") -> dict:
    """Delta, gamma, vega, theta (per year, -dV/dtau) and rho (HW#5 exact_greek).

    Vega and rho are per unit change of sigma and r (not per 1%).
    """
    kind = normalize_type(option_type)
    S, K, tau, sigma = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, K, tau, sigma)))
    d1, d2 = d1_d2(S, K, tau, r, q, sigma)
    dq, dr = np.exp(-q * tau), np.exp(-r * tau)
    pdf = norm.pdf(d1)
    sq = np.sqrt(np.maximum(tau, 1e-12))
    time_decay = -S * dq * pdf * sigma / (2 * sq)
    if kind == "call":
        delta = dq * norm.cdf(d1)
        theta = time_decay + q * S * dq * norm.cdf(d1) - r * K * dr * norm.cdf(d2)
        rho = K * tau * dr * norm.cdf(d2)
        delta_expired = np.where(S > K, 1.0, 0.0)
    else:
        delta = -dq * norm.cdf(-d1)
        theta = time_decay - q * S * dq * norm.cdf(-d1) + r * K * dr * norm.cdf(-d2)
        rho = -K * tau * dr * norm.cdf(-d2)
        delta_expired = np.where(S < K, -1.0, 0.0)
    gamma = dq * pdf / (S * np.maximum(sigma, 1e-12) * sq)
    vega = S * dq * pdf * sq
    expired = tau <= 0
    return {
        "delta": _ret(np.where(expired, delta_expired, delta)),
        "gamma": _ret(np.where(expired, 0.0, gamma)),
        "vega": _ret(np.where(expired, 0.0, vega)),
        "theta": _ret(np.where(expired, 0.0, theta)),
        "rho": _ret(np.where(expired, 0.0, rho)),
    }


def bs_delta(S, K, tau, r, q, sigma, option_type="call"):
    return bs_greeks(S, K, tau, r, q, sigma, option_type)["delta"]


def bs_gamma(S, K, tau, r, q, sigma, option_type="call"):
    return bs_greeks(S, K, tau, r, q, sigma, option_type)["gamma"]


def implied_vol(price, S, K, tau, r, q, option_type="call", lo=1e-4, hi=5.0) -> float:
    """Black-Scholes implied volatility by Brent root finding; nan if no root in [lo, hi]."""
    kind = normalize_type(option_type)
    if tau <= 0 or price <= 0:
        return float("nan")

    def f(sig):
        return bs_price(S, K, tau, r, q, sig, kind) - price

    try:
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(brentq(f, lo, hi, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def parity_gap(S, K, tau, r, q, call, put):
    """C - P - (S e^{-q tau} - K e^{-r tau}); zero when put-call parity holds."""
    fwd = np.asarray(S, float) * np.exp(-q * tau) - np.asarray(K, float) * np.exp(-r * tau)
    return _ret(np.asarray(call, float) - np.asarray(put, float) - fwd)


def put_call_parity(S, K, tau, r, q, call=None, put=None) -> float:
    """Return the missing leg of put-call parity: C = P + S e^{-q tau} - K e^{-r tau}."""
    forward_leg = S * np.exp(-q * tau) - K * np.exp(-r * tau)
    if call is None and put is not None:
        return float(put + forward_leg)
    if put is None and call is not None:
        return float(call - forward_leg)
    raise ValueError("Give exactly one of call or put")


def black76_price(F, K, tau, r, sigma, option_type="call"):
    """Black (1976) price on a forward: e^{-r tau}[F N(d1) - K N(d2)]."""
    F = np.asarray(F, dtype=float)
    return bs_price(F * np.exp(-r * np.asarray(tau, float)), K, tau, r, 0.0, sigma, option_type)
