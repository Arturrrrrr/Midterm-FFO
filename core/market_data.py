"""yfinance helpers (HW#1 options_scraper.py pattern).

Every fetch function raises on failure so that a transient outage is never cached.
Pages call them through ``safe()`` and fall back to manual input.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from core.bs import normalize_type

try:  # cache only when running inside the Streamlit runtime (pytest stays silent)
    import streamlit as st
    from streamlit.runtime import exists as _st_running

    _cache = st.cache_data(ttl=900, show_spinner=False) if _st_running() else (lambda f: f)
except Exception:  # pragma: no cover
    def _cache(f):
        return f

SCHEMA = ["ticker", "option_type", "strike", "expiry", "last", "bid", "ask",
          "volume", "open_interest", "iv", "spot", "quote_date"]


def _yf():
    import yfinance as yf
    return yf


@_cache
def get_spot(ticker: str) -> float:
    """Last close of the most recent trading day (HW#1), 5-day window so holidays do not return empty."""
    hist = _yf().Ticker(ticker).history(period="5d")
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker}")
    return float(hist["Close"].iloc[-1])


@_cache
def get_history(ticker: str, n_days: int) -> pd.DataFrame:
    """The last n_days+1 daily closes: columns date, close (oldest first)."""
    hist = _yf().Ticker(ticker).history(period="3mo")
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker}")
    out = hist[["Close"]].tail(int(n_days) + 1).reset_index()
    out.columns = ["date", "close"]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date
    out["close"] = out["close"].astype(float)
    return out


@_cache
def get_expiries(ticker: str) -> list:
    exps = list(_yf().Ticker(ticker).options)
    if not exps:
        raise ValueError(f"No option expiries returned for {ticker}")
    return exps


@_cache
def get_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Calls and puts of one expiry, normalised to SCHEMA."""
    chain = _yf().Ticker(ticker).option_chain(expiry)
    frames = []
    for kind, df in (("call", chain.calls), ("put", chain.puts)):
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "option_type": kind,
            "strike": df["strike"].astype(float),
            "expiry": pd.Timestamp(expiry),
            "last": df["lastPrice"].astype(float),
            "bid": df["bid"].astype(float),
            "ask": df["ask"].astype(float),
            "volume": df["volume"].fillna(0).astype(float),
            "open_interest": df["openInterest"].fillna(0).astype(float),
            "iv": df["impliedVolatility"].astype(float),
            "spot": np.nan,
            "quote_date": pd.Timestamp(dt.date.today()),
        }))
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise ValueError(f"Empty option chain for {ticker} {expiry}")
    return out


@_cache
def get_risk_free() -> float:
    """13-week T-bill yield (^IRX, percent) converted to a continuous rate ln(1 + y/100)."""
    hist = _yf().Ticker("^IRX").history(period="5d")
    if hist.empty:
        raise ValueError("No data for ^IRX")
    y = float(hist["Close"].iloc[-1])
    return float(np.log1p(y / 100.0))


def mid_price(bid, ask, last) -> float:
    """(bid+ask)/2 when both quotes are positive, else the last trade."""
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return float((bid + ask) / 2)
    return float(last) if last is not None and not np.isnan(last) else float("nan")


def nearest_quote(chain: pd.DataFrame, strike: float, option_type: str):
    """Quote of the listed strike closest to ``strike`` (dict) or None."""
    if chain is None or chain.empty:
        return None
    sub = chain[chain["option_type"] == normalize_type(option_type)]
    if sub.empty:
        return None
    row = sub.iloc[int((sub["strike"] - float(strike)).abs().argmin())]
    return {
        "strike": float(row["strike"]), "bid": float(row["bid"]), "ask": float(row["ask"]),
        "last": float(row["last"]), "mid": mid_price(row["bid"], row["ask"], row["last"]),
        "iv": float(row["iv"]), "expiry": pd.Timestamp(row["expiry"]).date(),
    }


def safe(fn, *args, default=None, **kwargs):
    """Call fn and return (value, None) or (default, error message)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # network, parsing, yfinance internals
        return default, f"{type(exc).__name__}: {exc}"
