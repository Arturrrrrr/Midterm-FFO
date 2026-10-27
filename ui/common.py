"""Shared Streamlit widgets: spot / rate inputs with online fetch + manual override, print CSS, tables."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from core.market_data import get_risk_free, get_spot, safe

def _version_tuple(v: str) -> tuple:
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


# Streamlit >= 1.46 uses **WIDE; older versions use use_container_width=True.
WIDE = {"width": "stretch"} if _version_tuple(st.__version__) >= (1, 46) else {"use_container_width": True}

PRINT_CSS = """
<style>
@media print {
  [data-testid="stSidebar"], header, footer, [data-testid="stToolbar"], [data-testid="stStatusWidget"],
  .stButton, .stDownloadButton, [data-testid="stFileUploader"], .no-print { display: none !important; }
  .block-container { max-width: 100% !important; padding: 0 !important; }
}
</style>
"""


def inject_print_css() -> None:
    st.markdown(PRINT_CSS, unsafe_allow_html=True)


def spot_input(ticker: str, key: str, label: str = "Spot price S", default: float = 100.0):
    """Number input with a 'fetch online' button (yfinance last close). Returns (S, source)."""
    num_key, src_key, fetched_key = f"{key}_num", f"{key}_src", f"{key}_fetched"
    if num_key not in st.session_state:
        st.session_state[num_key] = float(default)
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    with c2:
        if st.button(f"Fetch {ticker.upper()} online", key=f"{key}_btn", **WIDE):
            val, err = safe(get_spot, ticker.upper())
            if err:
                st.session_state[src_key] = "manual"
                st.warning(f"Could not fetch {ticker.upper()} from Yahoo Finance ({err}). Enter the spot manually.")
            else:
                st.session_state[num_key] = float(val)
                st.session_state[fetched_key] = float(val)
                st.session_state[src_key] = f"Yahoo Finance last close ({ticker.upper()})"
    with c1:
        S = st.number_input(label, min_value=0.0001, key=num_key, format="%.4f")
    fetched = st.session_state.get(fetched_key)
    source = st.session_state.get(src_key, "manual") if fetched is not None and abs(S - fetched) < 1e-12 else "manual"
    st.caption(f"Spot source: {source}")
    return float(S), source


def rate_input(key: str = "rf", label: str = "Risk-free rate r (continuous)", default: float = 0.05) -> float:
    """Risk-free rate with an online fetch of the 13-week T-bill (^IRX); stored in session_state['rf_rate']."""
    num_key = f"{key}_num"
    if num_key not in st.session_state:
        st.session_state[num_key] = float(st.session_state.get("rf_rate", default))
    if st.button("Fetch ^IRX (13-week T-bill)", key=f"{key}_btn", **WIDE):
        val, err = safe(get_risk_free)
        if err:
            st.warning(f"Could not fetch ^IRX ({err}); keeping the manual rate.")
        else:
            st.session_state[num_key] = float(val)
            st.session_state[f"{key}_src"] = "Yahoo Finance ^IRX"
    r = st.number_input(label, min_value=-0.05, max_value=0.5, step=0.0025, key=num_key, format="%.4f")
    st.session_state["rf_rate"] = float(r)
    return float(r)


def show_df(df: pd.DataFrame, decimals: int = 4, height=None, hide_index: bool = True) -> None:
    """st.dataframe with a uniform number format."""
    cfg = {}
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            cfg[c] = st.column_config.NumberColumn(format=f"%.{decimals}f")
    kwargs = {"height": height} if height is not None else {}
    st.dataframe(df, hide_index=hide_index, **WIDE, column_config=cfg, **kwargs)


def parse_number_list(text: str, fallback=None) -> list:
    """'95, 100 105' -> [95.0, 100.0, 105.0]."""
    vals = []
    for tok in str(text).replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok:
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return vals if vals else (fallback or [])


def today() -> dt.date:
    asof = st.session_state.get("asof")
    return asof if isinstance(asof, dt.date) else dt.date.today()


def fmt_money(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.2f}"
