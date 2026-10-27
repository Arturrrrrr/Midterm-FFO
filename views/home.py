import streamlit as st

st.title("Options Analytics Suite")
st.markdown("""
A web application for European stock options, built for the Financial Derivatives midterm. Pick a calculator in the
sidebar; every module has a dedicated **Help** button with the theory, the operational assumptions and the formulas
(rendered with SymPy).

| Module | What it does |
|---|---|
| **1 · Portfolio Evaluation** | Market and intrinsic value of a portfolio of calls, puts and stock, with spot prices fetched online or entered by hand. |
| **2 · Arbitrage Calculator** | Detects mispriced options with synthetic positions (put-call parity, Black-Scholes and binomial replication with stock and bonds) and spells out the trades step by step. |
| **3 · Delta / Delta-Gamma Hedging** | Daily hedge adjustments for written options, a spot price simulator (random, manual or historical) and projected P&L over up to 5 days, with printable tables, charts and reports. |
| **4 · SABR Calculator** | Hagan implied volatility, calibration of alpha, rho, nu (beta = 1) from uploaded CSV / Excel option data, volatility smiles across spot and forward levels, SABR option pricing and arbitrage strategies. |
| **5 · SABR Monte Carlo** | Simulation of the SABR stochastic differential equations (Euler-Maruyama or log-Euler / full truncation), distribution of the final price and the present value of the average payoff. |

**Global settings** (sidebar): default ticker, risk-free rate (fetch the 13-week T-bill or type a value) and valuation date.

**Online data** comes from Yahoo Finance through `yfinance`. If the connection fails, every input can be typed manually.
""")
