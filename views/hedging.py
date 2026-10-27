"""3 - Delta and Delta-Gamma Hedging with a Spot Price Simulator."""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.bs import bs_greeks, bs_price, year_fraction
from core.hedging import compare_strategies, simulate_hedge, simulate_spot, solve_static_hedge
from core.market_data import get_history, safe
from ui.common import WIDE, fmt_money, show_df, spot_input, today
from ui.help import help_button
from ui.reports import build_html_report, download_csv_button, download_report_button

c1, c2 = st.columns([5, 2], vertical_alignment="center")
c1.title("Delta / Delta-Gamma Hedging Calculator")
with c2:
    help_button("hedging")
st.markdown("Daily hedge adjustments for a **written** option position, a Spot Price Simulator (random walk, manual "
            "or historical closes) and the projected P&L over up to 5 days.")

# ------------------------------------------------------------------ written option
st.subheader("Written option")
ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "AAPL"), key="hg_ticker").upper().strip()
if "hg_spot_pending" in st.session_state:          # historical mode syncs the spot field before the widget exists
    st.session_state["hg_spot_num"] = st.session_state.pop("hg_spot_pending")
S0_manual, _ = spot_input(ticker, key="hg_spot", default=100.0)
a1, a2, a3, a4 = st.columns(4)
kind = a1.selectbox("Option type", ["call", "put"], key="hg_kind")
K = a2.number_input("Strike K", min_value=0.01, value=float(round(S0_manual, 0)), format="%.2f", key="hg_K")
mat_mode = a3.radio("Maturity input", ["Years", "Expiry date"], horizontal=True, key="hg_mat_mode")
if mat_mode == "Expiry date":
    expiry = a4.date_input("Expiry", value=today() + dt.timedelta(days=90), key="hg_expiry")
    T = year_fraction(expiry, today())
else:
    T = a4.number_input("T (years)", min_value=0.01, value=0.5, format="%.4f", key="hg_T")
b1, b2, b3, b4, b5 = st.columns(5)
r = b1.number_input("Risk-free rate r", value=float(st.session_state.get("rf_rate", 0.05)), format="%.4f", key="hg_r")
sigma = b2.number_input("Volatility sigma", min_value=0.001, value=0.25, format="%.4f", key="hg_sigma")
q = b3.number_input("Dividend yield q", value=0.0, min_value=0.0, max_value=0.5, format="%.4f", key="hg_q")
Q = b4.number_input("Contracts written Q", min_value=0.01, value=10.0, format="%.2f", key="hg_Q")
mult = b5.number_input("Multiplier (shares/contract)", min_value=1.0, value=100.0, format="%.0f", key="hg_mult")
if T <= 0:
    st.error("The expiry must be after the valuation date.")
    st.stop()

# ------------------------------------------------------------------ hedge choice
st.subheader("Hedge")
h1, h2, h3, h4 = st.columns(4)
hedge_mode = h1.radio("Strategy", ["Delta hedge", "Delta-gamma hedge"], key="hg_mode")
mode_key = "delta" if hedge_mode == "Delta hedge" else "delta_gamma"
hedge = None
if mode_key == "delta_gamma":
    hk = h2.selectbox("Hedge option type", ["call", "put"], key="hg_hkind")
    hK = h3.number_input("Hedge option strike", min_value=0.01, value=float(round(K * 1.05, 0)), format="%.2f", key="hg_hK")
    hT = h4.number_input("Hedge option T (years)", min_value=0.01, value=float(max(T, 0.02)), format="%.4f", key="hg_hT")
    hedge = {"K": hK, "T": hT, "option_type": hk}
else:
    h2.caption("Delta hedge: hold N_u x Delta shares against the N_u written options and rebalance daily.")

# ------------------------------------------------------------------ spot simulator
st.subheader("Spot Price Simulator")
SIM = {"Forecast: random walk (geometric Brownian motion)": "gbm",
       "Forecast: manual daily spots": "manual",
       "Historical: actual last closes (Yahoo Finance)": "historical"}
sim_mode = SIM[st.radio("Daily spot values", list(SIM), key="hg_sim_mode")]
s1, s2, s3, s4 = st.columns(4)
n_days = s1.slider("Number of days", 1, 5, 5, key="hg_days")
path = None
S0 = S0_manual
if sim_mode == "gbm":
    mu = s2.number_input("Drift mu (annual)", value=0.0, format="%.4f", key="hg_mu")
    sig_sim = s3.number_input("Simulated volatility (annual)", min_value=0.001, value=float(sigma), format="%.4f", key="hg_sigsim")
    seed = int(s4.number_input("Random seed", min_value=0, value=42, step=1, key="hg_seed"))
    path = simulate_spot(S0, n_days, "gbm", mu=mu, sigma=sig_sim, seed=seed, start=today())
elif sim_mode == "manual":
    s2.caption("Type the spot of each of the next days (day 0 is the spot above).")
    default_manual = pd.DataFrame({"day": np.arange(1, n_days + 1), "spot": [round(S0 * (1 + 0.004 * ((-1) ** i) * (i + 1)), 2)
                                                                           for i in range(n_days)]})
    man = st.data_editor(default_manual, key=f"hg_manual_{n_days}_{round(S0, 4)}", hide_index=True, num_rows="fixed",
                         column_config={"day": st.column_config.NumberColumn("day", disabled=True),
                                        "spot": st.column_config.NumberColumn("spot", min_value=0.01, format="%.4f")})
    try:
        path = simulate_spot(S0, n_days, "manual", manual=man["spot"].tolist(), start=today())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
else:
    s2.caption(f"Uses the last {n_days + 1} closes of {ticker}; day 0 is the oldest close, so the hedge is back-tested.")
    hist, err = safe(get_history, ticker, n_days)
    if err or hist is None:
        st.error(f"Could not download the price history of {ticker} ({err}). Use a forecast mode or fix the ticker.")
        st.stop()
    path = simulate_spot(S0, n_days, "historical", hist=hist)
    S0 = float(path["spot"].iloc[0])
    if abs(st.session_state.get("hg_spot_num", S0) - S0) > 1e-9:   # sync the spot field with the history
        st.session_state["hg_spot_pending"] = S0
        st.rerun()
    st.info(f"Historical back-test: day 0 = {path['date'].iloc[0]} close {S0:,.4f} (replaces the spot entered above).")
if abs(K / S0 - 1.0) > 0.5:
    st.warning(f"The strike K = {K:,.2f} is far from the day-0 spot {S0:,.4f}: the option is deep in or out of the money "
               "and its Greeks are ~0 or ~1, so there is little to hedge. Adjust K if this is not intended.")

# ------------------------------------------------------------------ computations
N_u = Q * mult
V0 = float(bs_price(S0, K, T, r, q, sigma, kind))
use_bs_prem = st.checkbox("Premium received = Black-Scholes price at day 0", value=True, key="hg_prem_bs")
premium = V0 if use_bs_prem else st.number_input("Premium received per share", min_value=0.0, value=float(round(V0, 4)),
                                                 format="%.4f", key="hg_prem")
hedge_opts = [(hedge["option_type"], hedge["K"], hedge["T"])] if hedge else []
static = solve_static_hedge(S0, K, T, r, sigma, kind, N_u, hedge_opts, mode_key if hedge else "delta", q) \
    if hedge else None
res = simulate_hedge(path, K, T, r, sigma, kind, Q, mult, mode_key, hedge, premium, q)
cmp = compare_strategies(path, K, T, r, sigma, kind, Q, mult, hedge, premium, q)
for w in res.warnings:
    st.warning(w)

# ------------------------------------------------------------------ outputs
st.subheader("Day-0 hedge")
g0 = bs_greeks(S0, K, T, r, q, sigma, kind)
t0 = res.table.iloc[0]
m = st.columns(6)
m[0].metric("Option price (day 0)", f"{V0:,.4f}")
m[1].metric("Delta / Gamma", f"{g0['delta']:.4f} / {g0['gamma']:.5f}")
m[2].metric("Premium received", fmt_money(N_u * premium))
m[3].metric("Shares to hold", f"{t0['shares_held']:,.2f}")
m[4].metric("Hedge options to hold", f"{t0['hedge_opt_held']:,.2f}" if hedge else "none")
m[5].metric("Initial cash account", fmt_money(t0["cash"]))
if hedge:
    st.caption(f"Hedge option delta {t0['hedge_delta']:.4f}, gamma {t0['hedge_gamma']:.5f}. Static HW#7 solution: "
               f"{static['q'][0]:,.2f} hedge options and {static['stock']:,.2f} shares -> portfolio delta "
               f"{static['port_delta']:.6f}, gamma {static['port_gamma']:.6f}.")

tbl = res.table.copy()
chart_tabs = st.tabs(["Spot path", "Cumulative P&L", "Positions", "Residual Greeks"])
fig_spot = go.Figure(go.Scatter(x=tbl["date"], y=tbl["spot"], mode="lines+markers", name="Spot"))
fig_spot.add_hline(y=K, line_dash="dash", line_color="grey", annotation_text="K")
fig_spot.update_layout(title=f"Daily spot values ({sim_mode})", xaxis_title="Date", yaxis_title="Spot", height=380)
fig_pnl = go.Figure()
labels = {"pnl_unhedged": "Unhedged (written option only)", "pnl_delta": "Delta hedged", "pnl_delta_gamma": "Delta-gamma hedged"}
for col, lab in labels.items():
    if col in cmp:
        fig_pnl.add_trace(go.Scatter(x=cmp["date"], y=cmp[col], mode="lines+markers", name=lab,
                                     line=dict(width=3 if col.endswith(mode_key) else 1.5)))
fig_pnl.add_hline(y=0, line_color="black", line_width=1)
fig_pnl.update_layout(title="Cumulative P&L of the written position", xaxis_title="Date", yaxis_title="P&L", height=380,
                      legend=dict(orientation="h", y=-0.25))
fig_pos = make_subplots(specs=[[{"secondary_y": True}]])
fig_pos.add_trace(go.Bar(x=tbl["date"], y=tbl["shares_traded"], name="Shares traded (day)"), secondary_y=False)
fig_pos.add_trace(go.Scatter(x=tbl["date"], y=tbl["shares_held"], name="Shares held", mode="lines+markers"), secondary_y=False)
if hedge:
    fig_pos.add_trace(go.Scatter(x=tbl["date"], y=tbl["hedge_opt_held"], name="Hedge options held", mode="lines+markers",
                                 line=dict(dash="dash")), secondary_y=True)
fig_pos.update_layout(title="Daily adjustments", height=380, legend=dict(orientation="h", y=-0.25))
fig_pos.update_yaxes(title_text="Shares", secondary_y=False)
fig_pos.update_yaxes(title_text="Hedge options", secondary_y=True)
fig_greeks = make_subplots(rows=1, cols=2, subplot_titles=("Portfolio delta after rebalancing", "Portfolio gamma after rebalancing"))
fig_greeks.add_trace(go.Scatter(x=tbl["date"], y=tbl["port_delta"], mode="lines+markers", name="Delta"), row=1, col=1)
fig_greeks.add_trace(go.Scatter(x=tbl["date"], y=tbl["port_gamma"], mode="lines+markers", name="Gamma"), row=1, col=2)
fig_greeks.add_trace(go.Scatter(x=tbl["date"], y=-N_u * tbl["gamma"], mode="lines", name="Gamma unhedged", line=dict(dash="dot")), row=1, col=2)
fig_greeks.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
for tab, fig in zip(chart_tabs, (fig_spot, fig_pnl, fig_pos, fig_greeks)):
    with tab:
        st.plotly_chart(fig, **WIDE)

st.subheader("Daily adjustment report")
for line in res.trades:
    st.markdown(f"- {line}")

st.subheader("Daily table")
cols = ["day", "date", "spot", "tau", "option_price", "delta", "gamma"]
if hedge:
    cols += ["hedge_opt_price", "hedge_delta", "hedge_gamma", "hedge_opt_held", "hedge_opt_traded"]
cols += ["shares_held", "shares_traded", "cash", "option_liability", "portfolio_value", "daily_pnl", "cum_pnl",
         "port_delta", "port_gamma"]
show_df(tbl[cols])

st.subheader("Summary")
summary = pd.DataFrame({
    "Strategy": [labels[c] for c in labels if c in cmp],
    "Final cumulative P&L": [cmp[c].iloc[-1] for c in labels if c in cmp],
    "Worst daily P&L": [cmp[c].diff().fillna(0).min() for c in labels if c in cmp],
    "Best daily P&L": [cmp[c].diff().fillna(0).max() for c in labels if c in cmp],
})
show_df(summary, decimals=2)
st.caption(f"Selected strategy: total shares traded {res.summary['total_shares_traded']:,.2f}, hedge options traded "
           f"{res.summary['total_hedge_options_traded']:,.2f}, largest daily |P&L| {res.summary['max_abs_daily_pnl']:,.2f}, "
           f"residual delta {res.summary['final_port_delta']:.6f}, residual gamma {res.summary['final_port_gamma']:.6f}.")

st.subheader("Print / export")
d1, d2, d3 = st.columns(3)
with d1:
    download_csv_button(tbl[cols], "hedging_daily_table.csv")
with d2:
    download_csv_button(cmp, "hedging_pnl_comparison.csv", label="Download P&L comparison (CSV)")
with d3:
    inputs = (f"Written {kind}: K = {K:,.2f}, T = {T:.4f} y, sigma = {sigma:.2%}, r = {r:.4%}, q = {q:.4%}, Q = {Q:g} x {mult:g} "
              f"shares, premium {premium:,.4f}/share. Hedge: {hedge_mode}"
              + (f" with {hedge['option_type']} K = {hedge['K']:,.2f}, T = {hedge['T']:.4f}." if hedge else ".")
              + f" Spot path: {sim_mode}, {n_days} days, S0 = {S0:,.4f}.")
    sections = [{"heading": "Inputs", "text": inputs},
                {"heading": "Daily adjustment report", "steps": res.trades},
                {"heading": "Summary", "table": summary},
                {"heading": "Daily table", "table": tbl[cols]},
                {"heading": "Spot path", "fig": fig_spot},
                {"heading": "Cumulative P&L", "fig": fig_pnl},
                {"heading": "Daily adjustments", "fig": fig_pos},
                {"heading": "Residual Greeks", "fig": fig_greeks}]
    download_report_button(build_html_report("Delta / Delta-Gamma Hedging Report", sections), "hedging_report.html")
