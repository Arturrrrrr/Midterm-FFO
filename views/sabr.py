"""4 - SABR Calculator: Hagan implied volatility, calibration, smiles, pricing and arbitrage."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.bs import bs_price
from core.sabr import (SabrParams, calibrate_surface, hagan_implied_vol, infer_spot, latest_snapshot,
                       load_options_file, price_with_sabr, sabr_arbitrage, smile_grid)
from ui.common import WIDE, fmt_money, parse_number_list, show_df, spot_input
from ui.help import help_button
from ui.reports import build_html_report, download_csv_button, download_report_button

DATA = Path(__file__).resolve().parents[1] / "data"

c1, c2 = st.columns([5, 2], vertical_alignment="center")
c1.title("SABR Calculator")
with c2:
    help_button("sabr")
st.markdown("Hagan's implied volatility, calibration of (alpha, rho, nu) with beta = 1 from uploaded option data, "
            "volatility smiles across spot and forward levels, SABR option prices and arbitrage strategies.")

tab_iv, tab_cal, tab_smile, tab_arb = st.tabs(["Hagan implied volatility", "Calibrate from CSV / Excel",
                                               "Smile across S and F", "Pricing and arbitrage"])


def _set_selected(alpha, beta, rho, nu, S0, r, T, expiry=None, source="manual"):
    st.session_state["sabr_selected"] = {"alpha": float(alpha), "beta": float(beta), "rho": float(rho), "nu": float(nu),
                                         "S0": float(S0), "r": float(r), "T": float(T), "F": float(S0 * np.exp(r * T)),
                                         "expiry": str(expiry) if expiry is not None else None, "source": source}


def _selected():
    return st.session_state.get("sabr_selected")


def _params_from_selected(sel) -> SabrParams:
    return SabrParams(sel["alpha"], sel["beta"], sel["rho"], sel["nu"], sel["F"], sel["T"])


def _smile_fig(df, x="K", y="sigma_sabr", group="S", title="", market=None):
    fig = go.Figure()
    for g, sub in df.groupby(group):
        fig.add_trace(go.Scatter(x=sub[x], y=sub[y], mode="lines", name=f"{group} = {g:,.2f}"))
    if market is not None and not market.empty:
        fig.add_trace(go.Scatter(x=market["strike"], y=market["iv"], mode="markers", name="Market implied vol",
                                 marker=dict(size=7, symbol="circle-open")))
    fig.update_layout(title=title, xaxis_title="Strike K", yaxis_title="Implied volatility", height=420,
                      legend=dict(orientation="h", y=-0.25))
    return fig


# ================================================================== tab 1: Hagan IV
with tab_iv:
    st.subheader("Inputs")
    i1, i2, i3, i4 = st.columns(4)
    price_in = i1.radio("Underlying input", ["Spot S (F = S e^rT)", "Forward F"], key="iv_in")
    r1 = i2.number_input("Risk-free rate r", value=float(st.session_state.get("rf_rate", 0.05)), format="%.4f", key="iv_r")
    T1 = i3.number_input("Time to maturity T (years)", min_value=0.001, value=0.5, format="%.4f", key="iv_T")
    K1 = i4.number_input("Strike K", min_value=0.01, value=100.0, format="%.2f", key="iv_K")
    if price_in.startswith("Spot"):
        S1, _ = spot_input(st.session_state.get("ticker", "AAPL"), key="iv_spot", default=100.0)
        F1 = S1 * np.exp(r1 * T1)
    else:
        F1 = st.number_input("Forward F", min_value=0.01, value=100.0, format="%.4f", key="iv_F")
        S1 = F1 * np.exp(-r1 * T1)
    p1, p2, p3, p4 = st.columns(4)
    a1 = p1.number_input("alpha (initial vol)", min_value=0.0001, value=0.25, format="%.4f", key="iv_alpha")
    b1 = p2.number_input("beta (1 for equities)", min_value=0.0, max_value=1.0, value=1.0, format="%.2f", key="iv_beta")
    rho1 = p3.number_input("rho (correlation)", min_value=-0.999, max_value=0.999, value=-0.4, format="%.4f", key="iv_rho")
    nu1 = p4.number_input("nu (vol of vol)", min_value=0.0, value=0.6, format="%.4f", key="iv_nu")
    k1, k2, k3 = st.columns(3)
    kmin = k1.number_input("Smile: K min", min_value=0.01, value=float(round(0.7 * F1, 2)), format="%.2f", key="iv_kmin")
    kmax = k2.number_input("Smile: K max", min_value=0.01, value=float(round(1.3 * F1, 2)), format="%.2f", key="iv_kmax")
    kn = int(k3.number_input("Smile: number of strikes", min_value=3, max_value=200, value=25, step=1, key="iv_kn"))
    try:
        iv_single = float(hagan_implied_vol(F1, K1, T1, a1, b1, rho1, nu1))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    c_single = float(bs_price(S1, K1, T1, r1, 0.0, iv_single, "call"))
    p_single = float(bs_price(S1, K1, T1, r1, 0.0, iv_single, "put"))
    st.subheader("Result")
    m = st.columns(5)
    m[0].metric("Forward F", f"{F1:,.4f}")
    m[1].metric("Hagan implied vol", f"{iv_single:.4%}")
    m[2].metric("ATM vol", f"{float(hagan_implied_vol(F1, F1, T1, a1, b1, rho1, nu1)):.4%}")
    m[3].metric("Call (Black-Scholes with SABR vol)", f"{c_single:,.4f}")
    m[4].metric("Put", f"{p_single:,.4f}")
    Ks = np.linspace(min(kmin, kmax), max(kmin, kmax), kn)
    prm1 = SabrParams(a1, b1, rho1, nu1, F1, T1)
    grid1 = smile_grid(prm1, Ks, r1, F_values=[F1], T=T1)
    fig1 = _smile_fig(grid1, group="F", title="Hagan implied volatility smile")
    fig1.add_trace(go.Scatter(x=[K1], y=[iv_single], mode="markers", name=f"K = {K1:,.2f}", marker=dict(size=11)))
    st.plotly_chart(fig1, **WIDE)
    show_df(grid1[["S", "F", "T", "K", "sigma_sabr", "call_bs", "put_bs"]])
    if st.button("Use these parameters in the other tabs", key="iv_use"):
        _set_selected(a1, b1, rho1, nu1, S1, r1, T1, source="Hagan tab")
        st.success("Parameters stored for the Smile, Pricing and Monte Carlo pages.")
    e1, e2 = st.columns(2)
    with e1:
        download_csv_button(grid1, "hagan_smile.csv")
    with e2:
        secs = [{"heading": "Inputs", "text": f"F = {F1:,.4f} (S = {S1:,.4f}, r = {r1:.4%}, T = {T1:.4f}), alpha = {a1}, beta = {b1}, "
                                               f"rho = {rho1}, nu = {nu1}. K = {K1:,.2f}: implied vol {iv_single:.4%}, call {c_single:,.4f}, put {p_single:,.4f}."},
                {"heading": "Smile", "fig": fig1}, {"heading": "Table", "table": grid1}]
        download_report_button(build_html_report("Hagan Implied Volatility", secs), "hagan_iv_report.html")

# ================================================================== tab 2: calibration
with tab_cal:
    st.subheader("Historical option data")
    up = st.file_uploader("Upload a CSV or Excel file (yfinance export such as HW#8, or the HW#1 Table_1 format)",
                          type=["csv", "xlsx", "xlsm", "xls"], key="cal_upload")
    s1, s2 = st.columns(2)
    if s1.button("Use the AAPL sample (HW#8 latest snapshot)", key="cal_sample_aapl", **WIDE):
        st.session_state["options_df"] = load_options_file(DATA / "sample_aapl_options.csv")
        st.session_state["options_src"] = "sample_aapl_options.csv"
    if s2.button("Use the LLY sample (HW#1 Table_1 format)", key="cal_sample_lly", **WIDE):
        st.session_state["options_df"] = load_options_file(DATA / "sample_lly_table1.csv")
        st.session_state["options_src"] = "sample_lly_table1.csv"
    if up is not None and st.session_state.get("options_src") != up.name:
        try:
            with st.spinner("Reading the file..."):
                st.session_state["options_df"] = load_options_file(up)
            st.session_state["options_src"] = up.name
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
    df = st.session_state.get("options_df")
    if df is None or df.empty:
        st.info("Upload a file or pick a sample to calibrate.")
    else:
        snap = latest_snapshot(df)
        st.caption(f"Loaded {st.session_state.get('options_src')}: {len(df):,} quotes, {snap['ticker'].nunique()} ticker(s), "
                   f"latest quote date {pd.Timestamp(snap['quote_date'].max()).date()}.")
        tickers = sorted(snap["ticker"].unique())
        c1_, c2_, c3_, c4_ = st.columns(4)
        tk = c1_.selectbox("Ticker", tickers, key="cal_ticker")
        inferred = infer_spot(snap, tk)
        S0_key = f"cal_S0_{tk}"
        if S0_key not in st.session_state:
            st.session_state[S0_key] = float(inferred) if inferred else 100.0
        S0 = c2_.number_input("Spot S0 on the quote date", min_value=0.01, key=S0_key, format="%.4f",
                              help=f"Inferred from the file: {inferred}" if inferred else "Not available in the file: enter it")
        r2 = c3_.number_input("Risk-free rate r", value=float(st.session_state.get("rf_rate", 0.05)), format="%.4f", key="cal_r")
        beta2 = c4_.number_input("beta (fixed, 1 for equities)", min_value=0.0, max_value=1.0, value=1.0, format="%.2f", key="cal_beta")
        d1_, d2_, d3_, d4_ = st.columns(4)
        kind2 = d1_.selectbox("Quotes to fit", ["call", "put"], key="cal_kind")
        band_lo = d2_.number_input("Strike band: K/F min", min_value=0.1, max_value=1.0, value=0.7, format="%.2f", key="cal_lo")
        band_hi = d3_.number_input("Strike band: K/F max", min_value=1.0, max_value=3.0, value=1.3, format="%.2f", key="cal_hi")
        min_q = int(d4_.number_input("Min quotes per expiry", min_value=3, max_value=50, value=5, step=1, key="cal_minq"))
        if st.button("Calibrate", type="primary", key="cal_run"):
            try:
                with st.spinner("Calibrating one (alpha, rho, nu) per expiry..."):
                    params, quotes = calibrate_surface(snap[snap["ticker"] == tk], S0, r2, beta2, (band_lo, band_hi), kind2, min_q)
                st.session_state["sabr_params_df"] = params
                st.session_state["sabr_quotes_df"] = quotes
                st.session_state["sabr_cal_meta"] = {"ticker": tk, "S0": S0, "r": r2, "beta": beta2, "kind": kind2,
                                                     "valuation_date": str(params.attrs.get("valuation_date"))}
            except Exception as exc:
                st.error(f"Calibration failed: {exc}")
        params = st.session_state.get("sabr_params_df")
        meta = st.session_state.get("sabr_cal_meta")
        if params is not None and meta and meta.get("ticker") == tk:
            quotes = st.session_state["sabr_quotes_df"]
            st.subheader(f"Calibrated parameters ({meta['ticker']}, valuation date {meta['valuation_date']}, beta = {meta['beta']})")
            ptab = params.reset_index()
            show_df(ptab)
            exp_opts = [str(e) for e in params.index]
            e1_, e2_ = st.columns([3, 1])
            chosen = e1_.multiselect("Expiries to plot", exp_opts, default=exp_opts[:min(4, len(exp_opts))], key="cal_plot_exp")
            fig2 = go.Figure()
            for e in chosen:
                sub = quotes[quotes["expiry"].astype(str) == e]
                fig2.add_trace(go.Scatter(x=sub["strike"], y=sub["iv"], mode="markers", name=f"{e} market"))
                fig2.add_trace(go.Scatter(x=sub["strike"], y=sub["sabr_iv"], mode="lines", name=f"{e} SABR"))
            fig2.update_layout(title=f"{meta['ticker']}: market versus calibrated SABR smiles", xaxis_title="Strike K",
                               yaxis_title="Implied volatility", height=450, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig2, **WIDE)
            use_exp = e2_.selectbox("Expiry to use", exp_opts, key="cal_use_exp")
            if e2_.button("Use this expiry", key="cal_use_btn", **WIDE):
                row = params.loc[pd.Timestamp(use_exp).date()] if pd.Timestamp(use_exp).date() in params.index else params.iloc[exp_opts.index(use_exp)]
                _set_selected(row["alpha"], row["beta"], row["rho"], row["nu"], meta["S0"], meta["r"], row["T"], use_exp,
                              source=f"calibration {meta['ticker']}")
                st.success(f"Using {use_exp}: alpha = {row['alpha']:.4f}, rho = {row['rho']:.4f}, nu = {row['nu']:.4f}, "
                           f"F = {row['F']:,.4f}, T = {row['T']:.4f}.")
            st.subheader("Fitted quotes")
            show_df(quotes)
            x1, x2, x3 = st.columns(3)
            with x1:
                download_csv_button(ptab, "sabr_calibrated_parameters.csv", label="Download parameters (CSV)")
            with x2:
                download_csv_button(quotes, "sabr_fitted_quotes.csv", label="Download fitted quotes (CSV)")
            with x3:
                secs = [{"heading": "Inputs", "text": f"Ticker {meta['ticker']}, S0 = {meta['S0']:,.4f}, r = {meta['r']:.4%}, "
                                                       f"beta = {meta['beta']}, {meta['kind']} quotes, valuation date {meta['valuation_date']}."},
                        {"heading": "Calibrated parameters", "table": ptab},
                        {"heading": "Smiles", "fig": fig2},
                        {"heading": "Fitted quotes", "table": quotes}]
                download_report_button(build_html_report("SABR Calibration", secs), "sabr_calibration_report.html")

# ================================================================== tab 3: smile across S and F
with tab_smile:
    sel = _selected()
    if not sel:
        st.info("Set SABR parameters first: calibrate on the previous tab or use the Hagan tab.")
    else:
        prm = _params_from_selected(sel)
        st.caption(f"Parameters ({sel['source']}): alpha = {sel['alpha']:.4f}, beta = {sel['beta']:.2f}, rho = {sel['rho']:.4f}, "
                   f"nu = {sel['nu']:.4f}, S0 = {sel['S0']:,.4f}, F = {sel['F']:,.4f}, T = {sel['T']:.4f}.")
        g1, g2, g3 = st.columns(3)
        which = g1.radio("Vary", ["Spot S (F = S e^rT)", "Forward F (S = F e^-rT)"], key="sm_which")
        r3 = g2.number_input("Risk-free rate r", value=float(sel["r"]), format="%.4f", key="sm_r")
        T3 = g3.number_input("T (years)", min_value=0.001, value=float(sel["T"]), format="%.4f", key="sm_T")
        base = sel["S0"] if which.startswith("Spot") else sel["F"]
        txt = st.text_input("Levels (comma separated)", value=", ".join(f"{base * f:.2f}" for f in (0.9, 0.95, 1.0, 1.05, 1.1)), key="sm_levels")
        levels = parse_number_list(txt, [base])
        h1_, h2_, h3_ = st.columns(3)
        kmin3 = h1_.number_input("K min", min_value=0.01, value=float(round(0.7 * sel["F"], 2)), format="%.2f", key="sm_kmin")
        kmax3 = h2_.number_input("K max", min_value=0.01, value=float(round(1.3 * sel["F"], 2)), format="%.2f", key="sm_kmax")
        kn3 = int(h3_.number_input("Number of strikes", min_value=3, max_value=200, value=21, step=1, key="sm_kn"))
        Ks3 = np.linspace(min(kmin3, kmax3), max(kmin3, kmax3), kn3)
        market = None
        quotes = st.session_state.get("sabr_quotes_df")
        if quotes is not None and sel.get("expiry"):
            mq = quotes[quotes["expiry"].astype(str) == sel["expiry"]]
            market = mq if not mq.empty else None
        if which.startswith("Spot"):
            grid3 = smile_grid(prm, Ks3, r3, S_values=levels, T=T3)
            group = "S"
        else:
            grid3 = smile_grid(prm, Ks3, r3, F_values=levels, T=T3)
            group = "F"
        fig3 = _smile_fig(grid3, group=group, title="SABR implied volatility smile for several underlying levels", market=market)
        st.plotly_chart(fig3, **WIDE)
        st.subheader("Prices, forwards and Black-Scholes values")
        show_df(grid3)
        y1, y2 = st.columns(2)
        with y1:
            download_csv_button(grid3, "sabr_smile_grid.csv")
        with y2:
            secs = [{"heading": "Parameters", "table": pd.DataFrame([prm.as_dict()])},
                    {"heading": "Smile", "fig": fig3}, {"heading": "Grid", "table": grid3}]
            download_report_button(build_html_report("SABR Smile across S and F", secs), "sabr_smile_report.html")

# ================================================================== tab 4: pricing & arbitrage
with tab_arb:
    sel = _selected()
    if not sel:
        st.info("Set SABR parameters first: calibrate on the Calibrate tab or use the Hagan tab.")
    else:
        prm = _params_from_selected(sel)
        st.caption(f"Parameters ({sel['source']}): alpha = {sel['alpha']:.4f}, beta = {sel['beta']:.2f}, rho = {sel['rho']:.4f}, "
                   f"nu = {sel['nu']:.4f}, T = {sel['T']:.4f}.")
        z1, z2, z3, z4 = st.columns(4)
        S4 = z1.number_input("Spot S", min_value=0.01, value=float(sel["S0"]), format="%.4f", key="ar_S")
        r4 = z2.number_input("Risk-free rate r", value=float(sel["r"]), format="%.4f", key="ar_r")
        T4 = z3.number_input("T (years)", min_value=0.001, value=float(sel["T"]), format="%.4f", key="ar_T")
        kind4 = z4.selectbox("Option type", ["call", "put"], key="ar_kind")
        w1, w2, w3, w4 = st.columns(4)
        K4 = w1.number_input("Strike K", min_value=0.01, value=float(round(sel["S0"], 0)), format="%.2f", key="ar_K")
        quotes = st.session_state.get("sabr_quotes_df")
        prefill = None
        if quotes is not None and sel.get("expiry"):
            mq = quotes[(quotes["expiry"].astype(str) == sel["expiry"])]
            if not mq.empty:
                row = mq.iloc[(mq["strike"] - K4).abs().argmin()]
                if abs(row["strike"] - K4) < 1e-6:
                    bid, ask, last = row.get("bid"), row.get("ask"), row.get("last")
                    prefill = float((bid + ask) / 2) if bid and ask and bid > 0 and ask > 0 else (float(last) if last else None)
        v4 = w2.number_input("Market option price", min_value=0.0, value=float(prefill) if prefill else 5.0, format="%.4f", key="ar_v",
                             help="Prefilled with the market mid of the calibrated quotes when the strike is listed")
        thr4 = w3.number_input("Transaction-cost threshold per share", min_value=0.0, value=0.05, format="%.4f", key="ar_thr")
        units4 = w4.number_input("Shares per trade", min_value=1.0, value=100.0, format="%.0f", key="ar_units")
        try:
            res4 = sabr_arbitrage(S4, K4, T4, r4, prm, kind4, v4, thr4, units4)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        sig4, fair4 = price_with_sabr(S4, K4, T4, r4, prm, kind4)
        st.subheader("Verdict")
        (st.success if res4.is_arbitrage else st.info)(res4.verdict)
        m4 = st.columns(5)
        m4[0].metric("Forward F", f"{S4 * np.exp(r4 * T4):,.4f}")
        m4[1].metric("SABR implied vol", f"{sig4:.4%}")
        m4[2].metric("SABR fair price", f"{fair4:,.4f}")
        m4[3].metric("Mispricing (market - fair)", f"{res4.mispricing:+,.4f}")
        m4[4].metric(f"Profit ({units4:g} shares)", fmt_money(res4.profit_t0))
        st.subheader("Step-by-step strategy")
        for i, s in enumerate(res4.steps, 1):
            st.markdown(f"**{i}.** {s}")
        st.subheader("Cash flows per share")
        show_df(res4.cashflows)
        st.subheader("Printable pricing table (SABR vol, forward, Black-Scholes values)")
        Ks4 = np.linspace(0.8 * K4, 1.2 * K4, 9)
        tbl4 = smile_grid(prm, Ks4, r4, S_values=[S4], T=T4)
        show_df(tbl4)
        st.subheader("Calibrated parameters")
        ptab4 = pd.DataFrame([{**prm.as_dict(), "S0": S4, "r": r4, "expiry": sel.get("expiry")}])
        show_df(ptab4)
        o1, o2 = st.columns(2)
        with o1:
            download_csv_button(tbl4, "sabr_pricing_table.csv")
        with o2:
            secs = [{"heading": "Inputs", "text": f"S = {S4:,.4f}, K = {K4:,.2f}, T = {T4:.4f}, r = {r4:.4%}, {kind4}, market price {v4:,.4f}."},
                    {"heading": "Calibrated parameters", "table": ptab4},
                    {"heading": "Verdict", "text": res4.verdict},
                    {"heading": "Strategy", "steps": res4.steps},
                    {"heading": "Cash flows per share", "table": res4.cashflows},
                    {"heading": "Pricing table", "table": tbl4}]
            download_report_button(build_html_report("SABR Pricing and Arbitrage", secs), "sabr_arbitrage_report.html")
