"""5 - SABR Monte Carlo: Euler-Maruyama / log-Euler full truncation simulation of the SABR SDEs."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.sabr import SCHEMES, mc_option_price, simulate_sabr
from ui.common import WIDE, fmt_money, show_df
from ui.help import help_button
from ui.reports import build_html_report, download_csv_button, download_report_button

c1, c2 = st.columns([5, 2], vertical_alignment="center")
c1.title("SABR Monte Carlo")
with c2:
    help_button("sabr_mc")
st.markdown("Simulates the SABR stochastic differential equations path by path, reports the distribution of the final "
            "underlying price and the **present value of the average payoff** of a European option.")

sel = st.session_state.get("sabr_selected") or {}
if sel:
    st.caption(f"Prefilled from {sel['source']}: alpha = {sel['alpha']:.4f}, beta = {sel['beta']:.2f}, rho = {sel['rho']:.4f}, "
               f"nu = {sel['nu']:.4f}, S0 = {sel['S0']:,.4f}, T = {sel['T']:.4f}.")

st.subheader("Model inputs")
i1, i2, i3, i4 = st.columns(4)
in_mode = i1.radio("Underlying input", ["Spot S0 (F0 = S0 e^rT)", "Forward F0"], key="mc_in")
r = i2.number_input("Risk-free rate r", value=float(sel.get("r", st.session_state.get("rf_rate", 0.05))), format="%.4f", key="mc_r")
T = i3.number_input("Maturity T (years)", min_value=0.001, value=float(sel.get("T", 0.5)), format="%.4f", key="mc_T")
K = i4.number_input("Strike K", min_value=0.01, value=float(round(sel.get("S0", 100.0), 2)), format="%.2f", key="mc_K")
if in_mode.startswith("Spot"):
    S0 = st.number_input("Spot S0", min_value=0.01, value=float(sel.get("S0", 100.0)), format="%.4f", key="mc_S0")
    F0 = S0 * np.exp(r * T)
else:
    F0 = st.number_input("Forward F0", min_value=0.01, value=float(sel.get("F", 100.0)), format="%.4f", key="mc_F0")
    S0 = F0 * np.exp(-r * T)
p1, p2, p3, p4 = st.columns(4)
alpha = p1.number_input("alpha (initial vol)", min_value=0.0001, value=float(sel.get("alpha", 0.25)), format="%.4f", key="mc_alpha")
beta = p2.number_input("beta", min_value=0.0, max_value=1.0, value=float(sel.get("beta", 1.0)), format="%.2f", key="mc_beta")
rho = p3.number_input("rho", min_value=-0.999, max_value=0.999, value=float(sel.get("rho", -0.4)), format="%.4f", key="mc_rho")
nu = p4.number_input("nu (vol of vol)", min_value=0.0, value=float(sel.get("nu", 0.6)), format="%.4f", key="mc_nu")

st.subheader("Simulation settings")
s1, s2, s3, s4, s5 = st.columns(5)
kind = s1.selectbox("Option type", ["call", "put"], key="mc_kind")
scheme_label = s2.radio("Discretisation scheme", list(SCHEMES.values()), key="mc_scheme")
scheme = [k for k, v in SCHEMES.items() if v == scheme_label][0]
n_paths = int(s3.number_input("Number of simulations (paths)", min_value=1000, max_value=2_000_000, value=100_000, step=10_000, key="mc_n"))
n_steps = int(s4.number_input("Time steps", min_value=1, max_value=2000, value=int(max(1, round(252 * T))), step=1, key="mc_steps",
                              help="Default: one step per trading day"))
seed = int(s5.number_input("Random seed", min_value=0, value=1, step=1, key="mc_seed"))
st.caption(f"F0 = {F0:,.4f}, dt = T / steps = {T / n_steps:.6f} years, {n_paths:,} paths x {n_steps} steps.")

if st.button("Run simulation", type="primary", key="mc_run"):
    try:
        with st.spinner("Simulating..."):
            res = simulate_sabr(F0, alpha, beta, rho, nu, T, n_paths, n_steps, scheme, seed)
            out = mc_option_price(res, K, r, kind)
        st.session_state["mc_result"] = (res, out, {"S0": S0, "F0": F0, "K": K, "r": r, "T": T, "kind": kind})
    except (ValueError, MemoryError) as exc:
        st.error(str(exc))

if "mc_result" not in st.session_state:
    st.info("Set the inputs and press Run simulation.")
    st.stop()
res, out, meta = st.session_state["mc_result"]

st.subheader("Results")
m = st.columns(5)
m[0].metric("PV of the average payoff", f"{out['pv']:,.4f}", help="e^(-rT) x mean(payoff)")
m[1].metric("Standard error", f"{out['std_err']:,.4f}")
m[2].metric("95% confidence interval", f"[{out['ci95_low']:,.4f}, {out['ci95_high']:,.4f}]")
m[3].metric("Hagan + Black-76 price", f"{out['hagan_black_pv']:,.4f}", delta=f"{out['pv'] - out['hagan_black_pv']:+.4f} (MC - Hagan)")
m[4].metric("Mean final price S_T = F_T", f"{out['mean_F_T']:,.4f}")
stats = pd.DataFrame({
    "Statistic": ["Scheme", "Paths", "Time steps", "Option", "Strike K", "F0", "Discount factor e^-rT",
                  "Mean payoff (undiscounted)", "PV of average payoff", "Standard error", "95% CI low", "95% CI high",
                  "Hagan implied vol", "Hagan + Black-76 price", "z-score (MC - Hagan) / SE", "Mean F_T", "Std F_T",
                  "Mean sigma_T", "Paths absorbed at 0", "Negative-sigma steps (Euler only)", "Runtime (s)"],
    "Value": [SCHEMES[out["scheme"]], f"{out['n_paths']:,}", out["n_steps"], out["option_type"], f"{out['K']:,.4f}",
              f"{meta['F0']:,.4f}", f"{out['discount_factor']:.6f}", f"{out['mean_payoff']:,.6f}", f"{out['pv']:,.6f}",
              f"{out['std_err']:,.6f}", f"{out['ci95_low']:,.6f}", f"{out['ci95_high']:,.6f}", f"{out['hagan_vol']:.6%}",
              f"{out['hagan_black_pv']:,.6f}", f"{out['z_score']:.3f}", f"{out['mean_F_T']:,.4f}", f"{out['std_F_T']:,.4f}",
              f"{out['mean_sigma_T']:.6f}", out["n_absorbed"], out["n_neg_sigma"], f"{out['seconds']:.2f}"],
})
show_df(stats)

fig_hist = go.Figure(go.Histogram(x=res.F_T, nbinsx=120, name="F_T"))
fig_hist.add_vline(x=meta["K"], line_dash="dash", line_color="red", annotation_text="K")
fig_hist.add_vline(x=meta["F0"], line_dash="dot", line_color="grey", annotation_text="F0")
fig_hist.update_layout(title="Distribution of the final underlying price S_T = F_T", xaxis_title="F_T", yaxis_title="Paths", height=400)
fig_paths = make_subplots(rows=1, cols=2, subplot_titles=(f"{res.paths_F.shape[1]} sample paths of F", "Sample paths of sigma"))
for j in range(res.paths_F.shape[1]):
    fig_paths.add_trace(go.Scatter(x=res.t_grid, y=res.paths_F[:, j], mode="lines", line=dict(width=1), showlegend=False), row=1, col=1)
    fig_paths.add_trace(go.Scatter(x=res.t_grid, y=res.paths_sigma[:, j], mode="lines", line=dict(width=1), showlegend=False), row=1, col=2)
fig_paths.add_hline(y=meta["K"], line_dash="dash", line_color="red", row=1, col=1)
fig_paths.update_xaxes(title_text="Time (years)")
fig_paths.update_layout(height=400)
t1, t2 = st.tabs(["Final price distribution", "Sample paths"])
with t1:
    st.plotly_chart(fig_hist, **WIDE)
with t2:
    st.plotly_chart(fig_paths, **WIDE)

st.subheader("Print / export")
d1, d2 = st.columns(2)
with d1:
    download_csv_button(stats, "sabr_mc_summary.csv", label="Download summary (CSV)")
with d2:
    secs = [{"heading": "Inputs", "text": f"S0 = {meta['S0']:,.4f}, F0 = {meta['F0']:,.4f}, K = {meta['K']:,.2f}, T = {meta['T']:.4f}, "
                                           f"r = {meta['r']:.4%}, alpha = {res.alpha}, beta = {res.beta}, rho = {res.rho}, nu = {res.nu}."},
            {"heading": "Summary", "table": stats},
            {"heading": "Final price distribution", "fig": fig_hist},
            {"heading": "Sample paths", "fig": fig_paths}]
    download_report_button(build_html_report("SABR Monte Carlo", secs), "sabr_mc_report.html")
