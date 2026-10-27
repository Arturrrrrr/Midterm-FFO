"""2 - Arbitrage Calculator."""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.arbitrage import put_call_parity, replicate_binomial, replicate_bs, ud_from_sigma
from core.bs import intrinsic, year_fraction
from core.market_data import get_chain, get_expiries, nearest_quote, safe
from ui.common import WIDE, fmt_money, show_df, spot_input, today
from ui.help import help_button
from ui.reports import build_html_report, download_csv_button, download_report_button

c1, c2 = st.columns([5, 2], vertical_alignment="center")
c1.title("Arbitrage Calculator")
with c2:
    help_button("arbitrage")
st.markdown("Checks whether an option is under- or overpriced by building a **synthetic option** from the stock and a "
            "bond, and lists the trades step by step.")

MODES = {
    "Put-call parity (call + bond = put + stock)": "parity",
    "Black-Scholes replication (Delta shares + bond)": "bs",
    "One-period binomial replication (u, d)": "binomial",
}
mode = MODES[st.radio("Synthetic construction", list(MODES), horizontal=True, key="arb_mode")]

# ------------------------------------------------------------------ common inputs
st.subheader("Market data")
ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "AAPL"), key="arb_ticker").upper().strip()
S, _ = spot_input(ticker, key="arb_spot", default=100.0)
i1, i2, i3, i4 = st.columns(4)
K = i1.number_input("Strike K", min_value=0.01, value=float(round(S, 0)), format="%.2f", key="arb_K")
mat_mode = i2.radio("Maturity input", ["Expiry date", "Years"], horizontal=True, key="arb_mat_mode")
if mat_mode == "Expiry date":
    expiry = i3.date_input("Expiry", value=today() + dt.timedelta(days=90), key="arb_expiry")
    T = year_fraction(expiry, today())
    i4.metric("T (years)", f"{T:.4f}")
else:
    T = i3.number_input("T (years)", min_value=0.001, value=0.25, format="%.4f", key="arb_T")
    expiry = None
j1, j2, j3, j4 = st.columns(4)
r = j1.number_input("Risk-free rate r", value=float(st.session_state.get("rf_rate", 0.05)), format="%.4f", key="arb_r")
q = j2.number_input("Dividend yield q", value=0.0, min_value=0.0, max_value=0.5, format="%.4f", key="arb_q")
threshold = j3.number_input("Transaction-cost threshold per share", value=0.05, min_value=0.0, format="%.4f", key="arb_thr",
                            help="A price gap smaller than this is not reported as arbitrage")
units = j4.number_input("Shares per trade (contract multiplier)", value=100.0, min_value=1.0, format="%.0f", key="arb_units")
if T <= 0:
    st.error("The expiry must be after the valuation date.")
    st.stop()

# ------------------------------------------------------------------ optional live quotes
with st.expander("Prefill option prices from the live option chain (Yahoo Finance)"):
    exps, err = safe(get_expiries, ticker)
    if err:
        st.warning(f"Expiries unavailable for {ticker}: {err}")
    else:
        exp_sel = st.selectbox("Listed expiry", exps, key="arb_live_exp")
        if st.button("Fetch call and put quotes at the nearest strike", key="arb_live_btn"):
            chain, err2 = safe(get_chain, ticker, exp_sel)
            if err2:
                st.warning(f"Chain unavailable: {err2}")
            else:
                qc, qp = nearest_quote(chain, K, "call"), nearest_quote(chain, K, "put")
                if qc:
                    st.session_state["arb_call"] = float(qc["mid"])
                    st.session_state["arb_vmkt"] = float(qc["mid"])
                    st.session_state["arb_sigma"] = float(qc["iv"]) if qc["iv"] > 0 else 0.25
                if qp:
                    st.session_state["arb_put"] = float(qp["mid"])
                st.success(f"Nearest listed strike {qc['strike'] if qc else '?'}: call mid {qc['mid'] if qc else 'n/a'}, "
                           f"put mid {qp['mid'] if qp else 'n/a'} (expiry {exp_sel}). Set K and the expiry to match.")

# ------------------------------------------------------------------ mode inputs and computation
st.subheader("Option prices")
try:
    if mode == "parity":
        k1, k2 = st.columns(2)
        call_mkt = k1.number_input("Market call price", min_value=0.0, value=float(st.session_state.get("arb_call", 5.0)),
                                   format="%.4f", key="arb_call_in")
        put_mkt = k2.number_input("Market put price", min_value=0.0, value=float(st.session_state.get("arb_put", 4.0)),
                                  format="%.4f", key="arb_put_in")
        res = put_call_parity(S, K, T, r, q, call_mkt, put_mkt, threshold, units)
    else:
        k1, k2, k3 = st.columns(3)
        kind = k1.selectbox("Option type", ["call", "put"], key="arb_kind")
        v_mkt = k2.number_input("Market option price", min_value=0.0, value=float(st.session_state.get("arb_vmkt", 5.0)),
                                format="%.4f", key="arb_vmkt_in")
        sigma = k3.number_input("Volatility sigma", min_value=0.001, value=float(st.session_state.get("arb_sigma", 0.25)),
                                format="%.4f", key="arb_sigma_in")
        if mode == "bs":
            res = replicate_bs(S, K, T, r, q, sigma, kind, v_mkt, threshold, units)
        else:
            u0, d0 = ud_from_sigma(sigma, T)
            u1, u2 = st.columns(2)
            u = u1.number_input("Up factor u", min_value=1.0001, value=float(round(u0, 4)), format="%.4f", key="arb_u",
                                help="Default: Cox-Ross-Rubinstein u = exp(sigma sqrt(T))")
            d = u2.number_input("Down factor d", min_value=0.0001, max_value=0.9999, value=float(round(d0, 4)), format="%.4f",
                                key="arb_d", help="Default: d = 1/u")
            res = replicate_binomial(S, K, T, r, u, d, kind, v_mkt, threshold, units)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# ------------------------------------------------------------------ outputs
st.subheader("Verdict")
(st.success if res.is_arbitrage else st.info)(res.verdict)
m = st.columns(4)
fair = res.details.get("synthetic_call", res.details.get("fair_value"))
m[0].metric("Synthetic / fair value per share", f"{fair:,.4f}")
m[1].metric("Mispricing per share (market - fair)", f"{res.mispricing:+,.4f}")
m[2].metric(f"Profit today ({units:g} shares)", fmt_money(res.profit_t0))
m[3].metric("Profit carried to T", fmt_money(res.profit_T))
if mode == "binomial":
    dd = res.details
    st.caption(f"Up node S u = {dd['Su']:,.4f} (payoff {dd['Cu']:,.4f}), down node S d = {dd['Sd']:,.4f} "
               f"(payoff {dd['Cd']:,.4f}), Delta = {dd['delta']:.4f}, bond B = {dd['bond']:,.4f}, "
               f"risk-neutral p = {dd['p']:.4f}.")
elif mode == "bs":
    dd = res.details
    st.caption(f"Black-Scholes value {dd['fair_value']:,.4f} = Delta {dd['delta']:.4f} x S + bond {dd['bond']:,.4f}. {dd['caveat']}")
else:
    dd = res.details
    st.caption(f"Synthetic call = P + S e^(-qT) - K e^(-rT) = {dd['synthetic_call']:,.4f}; synthetic put = "
               f"{dd['synthetic_put']:,.4f}; PV(K) = {dd['pv_strike']:,.4f}.")

st.subheader("Step-by-step strategy")
for i, s in enumerate(res.steps, 1):
    st.markdown(f"**{i}.** {s}")

st.subheader("Cash flows per share")
show_df(res.cashflows)

# payoff chart
grid = np.linspace(0.6 * K, 1.4 * K, 161)
fig = go.Figure()
growth = np.exp(r * T)
if mode == "parity":
    sgn = 1.0 if res.mispricing > 0 else -1.0       # +1: short call / long put / long stock / borrow
    legs = {
        "Call leg": -sgn * intrinsic(grid, K, "call"),
        "Put leg": sgn * intrinsic(grid, K, "put"),
        "Stock leg": sgn * grid,
        "Bond leg": -sgn * K * np.ones_like(grid),
    }
    for name, y in legs.items():
        fig.add_trace(go.Scatter(x=grid, y=y, name=name, line=dict(width=1, dash="dot")))
    total = sum(legs.values())
    fig.add_trace(go.Scatter(x=grid, y=total, name="Total at T (riskless)", line=dict(width=3)))
    fig.update_layout(title="Value of the arbitrage portfolio at expiry: the total does not depend on S_T")
else:
    kind = res.details.get("model")
    delta, B = res.details["delta"], res.details["bond"]
    opt = intrinsic(grid, K, st.session_state.get("arb_kind", "call"))
    rep = delta * grid + B * growth
    fig.add_trace(go.Scatter(x=grid, y=opt, name="Option payoff", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=grid, y=rep, name="Replica value Delta S_T + B e^(rT) (no rebalancing)", line=dict(width=2, dash="dash")))
    if mode == "binomial":
        for node, lab in ((res.details["Su"], "up node"), (res.details["Sd"], "down node")):
            fig.add_vline(x=node, line_dash="dot", line_color="grey", annotation_text=lab)
    fig.update_layout(title="Option payoff versus the static replicating portfolio at T")
fig.update_layout(xaxis_title="Stock price at expiry S_T", yaxis_title="Value per share", height=420,
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, **WIDE)

st.subheader("Print / export")
d1, d2 = st.columns(2)
with d1:
    download_csv_button(res.cashflows, "arbitrage_cashflows.csv")
with d2:
    inputs = (f"Mode: {[k for k, v in MODES.items() if v == mode][0]}. Ticker {ticker}, S = {S:,.4f}, K = {K:,.2f}, "
              f"T = {T:.4f} years, r = {r:.4%}, q = {q:.4%}, threshold = {threshold:.4f}, units = {units:g}.")
    sections = [{"heading": "Inputs", "text": inputs},
                {"heading": "Verdict", "text": res.verdict},
                {"heading": "Strategy", "steps": res.steps},
                {"heading": "Cash flows per share", "table": res.cashflows},
                {"heading": "Payoff at expiry", "fig": fig}]
    download_report_button(build_html_report("Arbitrage Calculator", sections), "arbitrage_report.html")
