"""Portfolio of European stock options: position model, valuation and payoff diagram.

Position representation follows HW#2 (kind / side / strike / quantity / premium);
market valuation adds live option quotes or Black-Scholes as a fallback.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.bs import bs_price, intrinsic, normalize_type, year_fraction

SIGN = {"long": 1.0, "short": -1.0}
KINDS = ("call", "put", "stock")
POSITION_COLUMNS = ["ticker", "kind", "side", "quantity", "multiplier", "strike",
                    "expiry", "premium", "vol", "market_price"]


@dataclass
class Position:
    ticker: str
    kind: str                       # 'call' | 'put' | 'stock'
    side: str                       # 'long' | 'short'
    quantity: float                 # contracts (shares when multiplier == 1)
    multiplier: float = 100.0       # shares per contract (stock rows: 1)
    strike: float | None = None
    expiry: dt.date | None = None
    premium: float = 0.0            # paid (long) / received (short) per unit
    vol: float | None = None        # volatility for the Black-Scholes fallback
    market_price: float | None = None  # manual per-unit market price override

    @property
    def units(self) -> float:
        return float(self.quantity) * float(self.multiplier)

    @property
    def sign(self) -> float:
        return SIGN[self.side]

    @property
    def label(self) -> str:
        if self.kind == "stock":
            return f"{self.side} stock {self.ticker}"
        return f"{self.side} {self.kind} {self.ticker} K={self.strike:g}"


def _num(v, default=None):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
            return default
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def positions_from_df(df: pd.DataFrame) -> list[Position]:
    """Build validated Position objects from an editable table (st.data_editor)."""
    out: list[Position] = []
    for i, row in df.iterrows():
        ticker = str(row.get("ticker", "") or "").strip().upper()
        if not ticker:
            continue
        kind = str(row.get("kind", "call")).strip().lower()
        if kind in ("c", "p"):
            kind = normalize_type(kind)
        side = str(row.get("side", "long")).strip().lower()
        if kind not in KINDS:
            raise ValueError(f"Row {i + 1}: kind must be call, put or stock (got {kind})")
        if side not in SIGN:
            raise ValueError(f"Row {i + 1}: side must be long or short (got {side})")
        qty = _num(row.get("quantity"), 0.0)
        if qty <= 0:
            raise ValueError(f"Row {i + 1}: quantity must be positive")
        mult = _num(row.get("multiplier"), 1.0 if kind == "stock" else 100.0)
        strike = _num(row.get("strike"))
        expiry = row.get("expiry")
        expiry = None if expiry is None or (isinstance(expiry, float) and np.isnan(expiry)) or pd.isna(expiry) \
            else pd.to_datetime(expiry).date()
        if kind != "stock":
            if strike is None or strike <= 0:
                raise ValueError(f"Row {i + 1}: an option needs a positive strike")
            if expiry is None:
                raise ValueError(f"Row {i + 1}: an option needs an expiry date")
        out.append(Position(
            ticker=ticker, kind=kind, side=side, quantity=qty, multiplier=mult, strike=strike,
            expiry=expiry, premium=_num(row.get("premium"), 0.0), vol=_num(row.get("vol")),
            market_price=_num(row.get("market_price")),
        ))
    return out


def value_positions(positions: list[Position], spots: dict, r: float, q: float = 0.0,
                    asof: dt.date | None = None, chain_lookup=None, default_vol: float = 0.25) -> pd.DataFrame:
    """Intrinsic and market value of every position.

    chain_lookup(ticker, expiry, strike, kind) -> per-unit market price or None (live quote).
    Market price priority: manual override > live quote > Black-Scholes with vol (or default_vol).
    """
    rows = []
    for p in positions:
        S = float(spots[p.ticker])
        if p.kind == "stock":
            T, intrinsic_unit = np.nan, S
            if p.market_price is not None:
                market_unit, source = p.market_price, "manual"
            else:
                market_unit, source = S, "spot"
            vol_used = np.nan
        else:
            T = year_fraction(p.expiry, asof)
            intrinsic_unit = float(intrinsic(S, p.strike, p.kind))
            vol_used = p.vol if p.vol else default_vol
            quote = None
            if p.market_price is not None:
                market_unit, source = float(p.market_price), "manual"
            else:
                if chain_lookup is not None:
                    try:
                        quote = chain_lookup(p.ticker, p.expiry, p.strike, p.kind)
                    except Exception:
                        quote = None
                if quote is not None and not np.isnan(quote) and quote > 0:
                    market_unit, source = float(quote), "market mid"
                else:
                    market_unit = float(bs_price(S, p.strike, T, r, q, vol_used, p.kind))
                    source = "Black-Scholes"
        scale = p.sign * p.units
        rows.append({
            "position": p.label, "ticker": p.ticker, "kind": p.kind, "side": p.side,
            "quantity": p.quantity, "multiplier": p.multiplier, "units": p.units,
            "strike": p.strike, "expiry": p.expiry, "T": T, "spot": S, "vol": vol_used,
            "premium": p.premium, "intrinsic_unit": intrinsic_unit, "market_unit": market_unit,
            "price_source": source,
            "intrinsic_value": scale * intrinsic_unit,
            "market_value": scale * market_unit,
            "time_value": scale * (market_unit - intrinsic_unit),
            "cost_basis": scale * p.premium,
            "pnl": scale * (market_unit - p.premium),
        })
    return pd.DataFrame(rows)


def portfolio_totals(valued: pd.DataFrame) -> dict:
    keys = ["intrinsic_value", "market_value", "time_value", "cost_basis", "pnl"]
    if valued is None or valued.empty:
        return {k: 0.0 for k in keys}
    return {k: float(valued[k].sum()) for k in keys}


def payoff_at_expiry(positions: list[Position], ticker: str, S_grid) -> pd.DataFrame:
    """HW#2 scenarios(): value of each position at expiry for a grid of S_T, plus PORTFOLIO and PROFIT."""
    S_grid = np.asarray(S_grid, dtype=float)
    table = pd.DataFrame(index=pd.Index(S_grid, name="S_T"))
    premium = 0.0
    for p in positions:
        if p.ticker != ticker:
            continue
        if p.kind == "stock":
            val = S_grid
        else:
            val = intrinsic(S_grid, p.strike, p.kind)
        table[p.label] = p.sign * p.units * val + 0.0
        premium += p.sign * p.units * p.premium
    table["PORTFOLIO"] = table.sum(axis=1) if len(table.columns) else 0.0
    table["PROFIT"] = table["PORTFOLIO"] - premium
    return table
