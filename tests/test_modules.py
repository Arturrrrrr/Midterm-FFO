"""Tests for arbitrage, hedging, SABR (Hagan, calibration, loader, Monte Carlo) and the SymPy formulas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import arbitrage, bs, formulas, hedging, sabr  # noqa: E402


# ---------------------------------------------------------------- arbitrage
def test_parity_arbitrage_and_cashflows():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    c = bs.bs_price(S, K, T, r, q, 0.2, "call")
    p = bs.bs_price(S, K, T, r, q, 0.2, "put")
    res = arbitrage.put_call_parity(S, K, T, r, q, c + 0.5, p, threshold=0.0, units=100)
    assert res.is_arbitrage and res.mispricing == pytest.approx(0.5, abs=1e-9)
    assert res.profit_t0 == pytest.approx(50.0, abs=1e-6)
    total = res.cashflows.iloc[-1]
    assert total.iloc[1] == pytest.approx(0.5, abs=1e-9)          # t = 0
    assert total.iloc[2] == pytest.approx(0.0, abs=1e-9)          # S_T < K
    assert total.iloc[3] == pytest.approx(0.0, abs=1e-9)          # S_T >= K
    fair = arbitrage.put_call_parity(S, K, T, r, q, c, p, threshold=0.01)
    assert not fair.is_arbitrage
    cheap = arbitrage.put_call_parity(S, K, T, r, q, c - 0.7, p)
    assert cheap.is_arbitrage and cheap.mispricing < 0 and "UNDERPRICED" in cheap.verdict


def test_binomial_replication():
    S, K, T, r, u, d = 100.0, 100.0, 1.0, 0.05, 1.2, 0.8
    growth = np.exp(r * T)
    p = (growth - d) / (u - d)
    V = np.exp(-r * T) * (p * 20.0 + (1 - p) * 0.0)
    res = arbitrage.replicate_binomial(S, K, T, r, u, d, "call", V, threshold=1e-9)
    assert not res.is_arbitrage and res.details["fair_value"] == pytest.approx(V)
    over = arbitrage.replicate_binomial(S, K, T, r, u, d, "call", V + 1.0)
    tot = over.cashflows.iloc[-1]
    assert over.is_arbitrage and tot.iloc[1] == pytest.approx(1.0) and tot.iloc[2] == pytest.approx(0.0, abs=1e-9)
    assert tot.iloc[3] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        arbitrage.replicate_binomial(S, K, T, r, 1.03, 0.9, "call", 5.0)


def test_bs_replication():
    res = arbitrage.replicate_bs(100, 100, 1, 0.05, 0, 0.2, "put", 5.0)
    assert res.is_arbitrage and res.mispricing == pytest.approx(5.0 - 5.5735, abs=1e-4)
    assert res.details["delta"] < 0 and res.details["bond"] > 0
    assert res.cashflows.iloc[-1, 1] == pytest.approx(0.5735, abs=1e-4)


# ---------------------------------------------------------------- hedging
def test_static_hedge_hw7_vector():
    s = hedging.solve_static_hedge(100, 100, 1, 0.05, 0.30, "call", 50, [("put", 95, 0.5), ("call", 105, 0.5)])
    assert s["q"] == pytest.approx([-18.88, 50.56], abs=0.01)
    assert s["stock"] == pytest.approx(0.0, abs=1e-9)
    assert s["port_delta"] == pytest.approx(0.0, abs=1e-9) and s["port_gamma"] == pytest.approx(0.0, abs=1e-9)


def test_simulate_hedge_bookkeeping():
    path = hedging.simulate_spot(100, 5, "gbm", sigma=0.25, seed=7)
    assert list(path["day"]) == [0, 1, 2, 3, 4, 5] and path["spot"].iloc[0] == 100
    args = dict(K=100, T=0.5, r=0.03, sigma=0.25, option_type="call", Q=10, multiplier=100)
    hedge = dict(K=105, T=0.4, option_type="call")
    for mode in ("none", "delta", "delta_gamma"):
        res = hedging.simulate_hedge(path, mode=mode, hedge=hedge, **args)
        t = res.table
        assert t["portfolio_value"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        g = np.exp(args["r"] / 252)
        N_u = 1000.0
        for i in range(1, len(t)):
            pre_trade = (-N_u * t["option_price"].iloc[i] + t["shares_held"].iloc[i - 1] * t["spot"].iloc[i]
                         + t["hedge_opt_held"].iloc[i - 1] * t["hedge_opt_price"].iloc[i] + t["cash"].iloc[i - 1] * g)
            assert t["portfolio_value"].iloc[i] == pytest.approx(pre_trade, abs=1e-6)   # self-financing
        if mode == "none":
            V0 = t["option_price"].iloc[0]
            closed = N_u * (V0 * g ** t["day"] - t["option_price"])
            assert np.allclose(t["cum_pnl"], closed, atol=1e-6)
        if mode == "delta_gamma":
            assert np.allclose(t["port_delta"], 0, atol=1e-8) and np.allclose(t["port_gamma"], 0, atol=1e-8)
        if mode == "delta":
            assert np.allclose(t["port_delta"], 0, atol=1e-8)
    cmp = hedging.compare_strategies(path, hedge=hedge, **args)
    assert abs(cmp["pnl_delta"].iloc[-1]) < abs(cmp["pnl_unhedged"].iloc[-1])


def test_manual_and_historical_spot_modes():
    man = hedging.simulate_spot(100, 3, "manual", manual=[101, 99, 102])
    assert list(man["spot"]) == [100, 101, 99, 102]
    hist = pd.DataFrame({"date": pd.date_range("2026-01-05", periods=8, freq="B").date, "close": np.linspace(90, 97, 8)})
    h = hedging.simulate_spot(100, 5, "historical", hist=hist)
    assert len(h) == 6 and h["spot"].iloc[-1] == 97


# ---------------------------------------------------------------- SABR
def test_hagan_formula():
    assert sabr.hagan_implied_vol(100, 100, 0.5, 0.2, 1.0, -0.3, 0.4) == pytest.approx(0.2005533, abs=1e-7)
    F, T, a, b, rho, nu = 100.0, 0.5, 2.0, 0.5, -0.3, 0.4
    atm = sabr.hagan_implied_vol(F, F, T, a, b, rho, nu)
    corr = 1 + ((1 - b) ** 2 / 24 * a**2 / F ** (2 - 2 * b) + rho * b * nu * a / (4 * F ** (1 - b)) + (2 - 3 * rho**2) / 24 * nu**2) * T
    assert atm == pytest.approx(a / F ** (1 - b) * corr, rel=1e-12)
    Ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vec = sabr.hagan_implied_vol(100, Ks, 0.5, 0.2, 1.0, -0.3, 0.4)

    def hw8(K, alpha, rho, nu, F=100, tau=0.5):  # HW#8 SABR() with F, tau as globals
        z = nu / alpha * np.log(F / K)
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        return alpha * z / x * (1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * tau)

    mask = Ks != 100
    assert np.allclose(vec[mask], hw8(Ks[mask], 0.2, -0.3, 0.4), rtol=1e-12)
    lo = sabr.hagan_implied_vol(100, 100 * (1 - 1e-6), 0.5, 0.2, 0.7, -0.3, 0.4)
    hi = sabr.hagan_implied_vol(100, 100 * (1 + 1e-6), 0.5, 0.2, 0.7, -0.3, 0.4)
    assert lo == pytest.approx(hi, rel=1e-4)


def test_calibration_round_trip():
    Ks = np.linspace(80, 120, 15)
    ivs = sabr.hagan_implied_vol(100, Ks, 0.5, 0.25, 1.0, -0.4, 0.6)
    prm = sabr.calibrate_smile(Ks, ivs, 100, 0.5, beta=1.0)
    assert prm.alpha == pytest.approx(0.25, abs=1e-3) and prm.rho == pytest.approx(-0.4, abs=1e-3)
    assert prm.nu == pytest.approx(0.6, abs=1e-3) and prm.rmse < 1e-6


def test_loader_both_formats_and_surface():
    d8 = sabr.load_options_file(ROOT / "data" / "sample_aapl_options.csv")
    assert set(sabr.SCHEMA) <= set(d8.columns) and set(d8["option_type"]) == {"call", "put"}
    assert pd.api.types.is_datetime64_any_dtype(d8["expiry"])
    snap = sabr.latest_snapshot(d8)
    S0 = sabr.infer_spot(snap, "AAPL")
    assert S0 == pytest.approx(266.25)
    params, quotes = sabr.calibrate_surface(snap, S0, 0.05985, beta=1.0)
    assert len(params) >= 10 and (params["rho"] < 0).all() and params["rmse"].median() < 0.05
    assert {"sabr_iv", "iv", "strike"} <= set(quotes.columns)
    d1 = sabr.load_options_file(ROOT / "data" / "sample_lly_table1.csv")
    assert set(sabr.SCHEMA) <= set(d1.columns) and d1["spot"].notna().all()
    assert sabr.infer_spot(d1, "LLY") == pytest.approx(1184.78)
    assert d1["expiry"].min() > pd.Timestamp("2026-09-28")
    with pytest.raises(ValueError):
        sabr.normalize_options(pd.DataFrame({"foo": [1]}))


def test_smile_grid_and_pricing():
    prm = sabr.SabrParams(0.25, 1.0, -0.4, 0.6, F=100 * np.exp(0.05 * 0.5), T=0.5)
    grid = sabr.smile_grid(prm, [90, 100, 110], 0.05, S_values=[95, 100])
    assert len(grid) == 6 and {"S", "F", "K", "sigma_sabr", "call_bs", "put_bs"} <= set(grid.columns)
    row = grid[(grid["S"] == 100) & (grid["K"] == 100)].iloc[0]
    sig, price = sabr.price_with_sabr(100, 100, 0.5, 0.05, prm, "call")
    assert row["sigma_sabr"] == pytest.approx(sig) and row["call_bs"] == pytest.approx(price)
    assert price == pytest.approx(bs.bs_price(100, 100, 0.5, 0.05, 0, sig, "call"))
    arb = sabr.sabr_arbitrage(100, 100, 0.5, 0.05, prm, "call", price + 1.0)
    assert arb.is_arbitrage and arb.mispricing == pytest.approx(1.0, abs=1e-9)


def test_monte_carlo_matches_hagan_and_schemes_agree():
    kw = dict(F0=100, alpha=0.2, beta=1.0, rho=-0.3, nu=0.4, T=0.5, n_paths=200_000, n_steps=100, seed=1)
    e = sabr.mc_option_price(sabr.simulate_sabr(scheme="euler", **kw), 100, 0.05, "call")
    l = sabr.mc_option_price(sabr.simulate_sabr(scheme="log_euler_full_truncation", **kw), 100, 0.05, "call")
    for out in (e, l):
        assert abs(out["pv"] - out["hagan_black_pv"]) < max(4 * out["std_err"], 0.01 * out["hagan_black_pv"])
        assert out["n_absorbed"] == 0
    assert abs(e["pv"] - l["pv"]) < 3 * np.sqrt(e["std_err"] ** 2 + l["std_err"] ** 2)
    res = sabr.simulate_sabr(100, 0.2, 1.0, -0.3, 0.4, 0.5, 2000, 20, "log_euler_full_truncation", seed=3)
    assert (res.F_T > 0).all() and (res.sigma_T > 0).all() and res.paths_F.shape == (21, 50)
    with pytest.raises(ValueError):
        sabr.simulate_sabr(100, 0.2, 1.0, -0.3, 0.4, 0.5, 10, 5000)


# ---------------------------------------------------------------- SymPy formulas
def test_formulas_render():
    s = formulas.latex("bs.call")
    assert s and "N" in s
    g = formulas.latex("bs.delta_call")
    assert "d_{1}" in g and "N" in g
    for name in formulas.names():
        assert isinstance(formulas.latex(name), str) and formulas.latex(name)
