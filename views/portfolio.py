"""1 - Portfolio Evaluation."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.market_data import get_chain, nearest_quote, safe
from core.portfolio import payoff_at_expiry, portfolio_totals, positions_from_df, value_positions
from ui.common import WIDE, fmt_money, show_df, spot_input, today
from ui.help import help_button
from ui.reports import build_html_report, download_csv_button, download_report_button

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_portfolio.csv"

c1, c2 = st.columns([5, 2], vertical_alignment="center")
c1.title("Portfolio Evaluation")
with c2:
    help_button("portfolio")
st.markdown("Market value and intrinsic value of a portfolio of European stock options (and stock). "
            "Edit the table, fetch the spot prices online or type them, then read the valuation below.")

# ------------------------------------------------------------------ positions table
if "pos_version" not in st.session_state:
    st.session_state["pos_version"] = 0
if "positions_df" not in st.session_state:
    st.session_state["positions_df"] = pd.DataFrame([{
        "ticker": st.session_state.get("ticker", "AAPL"), "kind": "call", "side": "long", "quantity": 1.0,
        "multiplier": 100.0, "strike": 100.0, "expiry": pd.Timestamp(today() + dt.timedelta(days=90)),
        "premium": 5.0, "vol": 0.25, "market_price": np.nan,
    }])

b1, b2, _ = st.columns([1, 1, 4])
if b1.button("Load sample portfolio", **WIDE):
    st.session_state["positions_df"] = pd.read_csv(SAMPLE, parse_dates=["expiry"])
    st.session_state["pos_version"] += 1
    st.rerun()
if b2.button("Clear table", **WIDE):
    st.session_state["positions_df"] = st.session_state["positions_df"].iloc[0:0]
    st.session_state["pos_version"] += 1
    st.rerun()

st.caption("One row per position. quantity = contracts (shares for a stock row), multiplier = shares per contract, "
           "premium = price paid (long) or received (short) per share, vol = volatility used when no market quote "
           "is available, market_price = optional manual price per share.")
edited = st.data_editor(
    st.session_state["positions_df"], num_rows="dynamic", **WIDE,
    key=f"positions_editor_{st.session_state['pos_version']}",
    column_config={
        "ticker": st.column_config.TextColumn("ticker", required=True),
        "kind": st.column_config.SelectboxColumn("kind", options=["call", "put", "stock"], required=True),
        "side": st.column_config.SelectboxColumn("side", options=["long", "short"], required=True),
        "quantity": st.column_config.NumberColumn("quantity", min_value=0.0, format="%.2f"),
        "multiplier": st.column_config.NumberColumn("multiplier", min_value=1.0, format="%.0f"),
        "strike": st.column_config.NumberColumn("strike", min_value=0.0, format="%.2f"),
        "expiry": st.column_config.DateColumn("expiry"),
        "premium": st.column_config.NumberColumn("premium", format="%.4f"),
        "vol": st.column_config.NumberColumn("vol", min_value=0.0, max_value=5.0, format="%.4f"),
        "market_price": st.column_config.NumberColumn("market_price", min_value=0.0, format="%.4f"),
    },
)

try:
    positions = positions_from_df(edited)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if not positions:
    st.info("Add at least one position (or load the sample portfolio).")
    st.stop()

# ------------------------------------------------------------------ market inputs
st.subheader("Underlying spot prices")
tickers = sorted({p.ticker for p in positions})
spots = {}
for row_start in range(0, len(tickers), 3):
    cols = st.columns(3)
    for col, tk in zip(cols, tickers[row_start:row_start + 3]):
        with col:
            spots[tk], _ = spot_input(tk, key=f"pf_spot_{tk}", label=f"{tk} spot", default=100.0)

st.subheader("Valuation settings")
p1, p2, p3, p4 = st.columns(4)
r = p1.number_input("Risk-free rate r", value=float(st.session_state.get("rf_rate", 0.05)), format="%.4f", key="pf_r")
q = p2.number_input("Dividend yield q", value=0.0, min_value=0.0, max_value=0.5, format="%.4f", key="pf_q")
default_vol = p3.number_input("Default volatility (rows without vol)", value=0.25, min_value=0.01, max_value=5.0,
                              format="%.4f", key="pf_vol")
use_live = p4.checkbox("Use live option-chain mid prices (Yahoo Finance)", value=False, key="pf_live",
                       help="Looks up the bid/ask of the exact strike and expiry; falls back to Black-Scholes otherwise")
asof = today()
st.caption(f"Valuation date: {asof} (change it in the sidebar). Time to expiry uses calendar days / 365.")

chain_errors = []


def chain_lookup(ticker, expiry, strike, kind):
    chain, err = safe(get_chain, ticker, expiry.strftime("%Y-%m-%d"))
    if err:
        chain_errors.append(f"{ticker} {expiry}: {err}")
        return None
    qte = nearest_quote(chain, strike, kind)
    if qte is None or abs(qte["strike"] - strike) > 1e-6 or not (qte["mid"] > 0):
        chain_errors.append(f"{ticker} {expiry} K={strike:g}: strike not listed, using Black-Scholes")
        return None
    return qte["mid"]


valued = value_positions(positions, spots, r, q, asof, chain_lookup if use_live else None, default_vol)
for msg in chain_errors:
    st.warning(msg)
totals = portfolio_totals(valued)

# ------------------------------------------------------------------ outputs
st.subheader("Portfolio value")
m = st.columns(5)
m[0].metric("Intrinsic value", fmt_money(totals["intrinsic_value"]))
m[1].metric("Market value", fmt_money(totals["market_value"]))
m[2].metric("Time value", fmt_money(totals["time_value"]))
m[3].metric("Cost basis (premiums)", fmt_money(totals["cost_basis"]))
m[4].metric("P&L vs premiums", fmt_money(totals["pnl"]))

show_cols = ["position", "quantity", "multiplier", "units", "strike", "expiry", "T", "spot", "vol", "premium",
             "intrinsic_unit", "market_unit", "price_source", "intrinsic_value", "market_value", "time_value",
             "cost_basis", "pnl"]
table = valued[show_cols].copy()
show_df(table)

st.subheader("Payoff at expiry")
figs = []
tabs = st.tabs([f"{tk}" for tk in tickers])
for tab, tk in zip(tabs, tickers):
    S = spots[tk]
    grid = np.linspace(0.5 * S, 1.5 * S, 201)
    pay = payoff_at_expiry(positions, tk, grid)
    fig = go.Figure()
    for col in pay.columns:
        if col in ("PORTFOLIO", "PROFIT"):
            continue
        fig.add_trace(go.Scatter(x=pay.index, y=pay[col], name=col, line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=pay.index, y=pay["PORTFOLIO"], name="Portfolio payoff", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=pay.index, y=pay["PROFIT"], name="Profit (payoff - premiums)", line=dict(width=3, dash="dash")))
    fig.add_vline(x=S, line_dash="dash", line_color="grey", annotation_text=f"S = {S:,.2f}")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title=f"{tk}: value of the positions at expiry", xaxis_title="Stock price at expiry S_T",
                      yaxis_title="Value", height=420, legend=dict(orientation="h", y=-0.2))
    with tab:
        st.plotly_chart(fig, **WIDE)
    figs.append((tk, fig))

# ------------------------------------------------------------------ downloads
st.subheader("Print / export")
d1, d2 = st.columns(2)
with d1:
    download_csv_button(table, "portfolio_valuation.csv")
with d2:
    tot_df = pd.DataFrame([totals]).rename(columns=str.title)
    sections = [
        {"heading": "Inputs", "text": f"Valuation date {asof}. r = {r:.4%}, q = {q:.4%}, default vol = {default_vol:.2%}. "
                                      f"Spot prices: " + ", ".join(f"{k} = {v:,.2f}" for k, v in spots.items())},
        {"heading": "Totals", "table": tot_df},
        {"heading": "Positions", "table": table},
    ] + [{"heading": f"Payoff at expiry: {tk}", "fig": fig} for tk, fig in figs]
    download_report_button(build_html_report("Portfolio Evaluation", sections), "portfolio_report.html")
