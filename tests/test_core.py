"""Sanity tests for the pure-python core package (run: pytest -q from the midterm folder)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import bs  # noqa: E402


# ---------------------------------------------------------------- Black-Scholes
def test_bs_reference_values():
    assert bs.bs_price(100, 100, 1, 0.05, 0, 0.2, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs.bs_price(100, 100, 1, 0.05, 0, 0.2, "put") == pytest.approx(5.5735, abs=1e-4)
    assert bs.bs_price(100, 90, 0, 0.05, 0, 0.2, "call") == 10.0          # expired -> intrinsic


def test_implied_vol_round_trip():
    p = bs.bs_price(100, 105, 0.5, 0.03, 0.01, 0.35, "put")
    assert bs.implied_vol(p, 100, 105, 0.5, 0.03, 0.01, "put") == pytest.approx(0.35, abs=1e-6)


def test_parity_and_greeks_vs_finite_differences():
    S, K, tau, r, q, sig = 100.0, 95.0, 0.75, 0.04, 0.02, 0.25
    c = bs.bs_price(S, K, tau, r, q, sig, "call")
    p = bs.bs_price(S, K, tau, r, q, sig, "put")
    assert bs.parity_gap(S, K, tau, r, q, c, p) == pytest.approx(0.0, abs=1e-10)
    assert bs.put_call_parity(S, K, tau, r, q, put=p) == pytest.approx(c, abs=1e-10)
    g = bs.bs_greeks(S, K, tau, r, q, sig, "call")
    h = 1e-4

    def f(s):
        return bs.bs_price(s, K, tau, r, q, sig, "call")

    def fv(v):
        return bs.bs_price(S, K, tau, r, q, v, "call")

    def ft(t):
        return bs.bs_price(S, K, t, r, q, sig, "call")

    assert g["delta"] == pytest.approx((f(S + h) - f(S - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((f(S + h) - 2 * f(S) + f(S - h)) / h**2, abs=1e-5)
    assert g["vega"] == pytest.approx((fv(sig + h) - fv(sig - h)) / (2 * h), abs=1e-5)
    assert g["theta"] == pytest.approx(-(ft(tau + h) - ft(tau - h)) / (2 * h), abs=1e-5)
    assert bs.black76_price(S * np.exp((r - q) * tau), K, tau, r, sig, "call") == pytest.approx(c, abs=1e-10)


def test_bs_vectorised():
    Ks = np.array([80.0, 100.0, 120.0])
    out = bs.bs_price(100, Ks, 1, 0.05, 0, 0.2, "call")
    assert out.shape == (3,) and np.all(np.diff(out) < 0)
