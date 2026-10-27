# Options Analytics Suite (FFD midterm)

A Streamlit web application for European stock options with five calculators. Every page has two buttons at the
top right: **Help** opens an overlay explaining how to use the page (what each button and field does, how to read the
results) and **Theory** opens the theoretical background, operational assumptions and formulas rendered with SymPy.

| Module                        | File                   | What it does                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 Portfolio Evaluation        | `views/portfolio.py` | Market and intrinsic value of a portfolio of calls, puts and stock. Spot prices fetched from Yahoo Finance or entered manually; unit prices from a manual override, the live option-chain mid or Black-Scholes. Payoff diagram, CSV and printable HTML report.                                                                                     |
| 2 Arbitrage Calculator        | `views/arbitrage.py` | Detects under/overpriced options with synthetic positions built from the stock and a bond: put-call parity, Black-Scholes replication (Delta shares + bond) and one-period binomial replication (u, d). Verdict, numbered step-by-step strategy, cash-flow table and payoff chart.                                                                 |
| 3 Delta / Delta-Gamma Hedging | `views/hedging.py`   | Daily hedge adjustments for a written option (delta or delta-gamma with a second option), self-financing cash account and P&L. Spot Price Simulator: geometric Brownian motion, manual daily spots or historical closes (1 to 5 days). Charts, daily table, adjustment report, summary, CSV and HTML report.                                       |
| 4 SABR Calculator             | `views/sabr.py`      | Hagan (2002) implied volatility (general beta, ATM limit), calibration of alpha, rho, nu with beta = 1 from an uploaded CSV / Excel file (yfinance export like HW#8 or the HW#1 Table_1 format), smiles across spot and forward levels, SABR option prices (Black-Scholes with Hagan's vol), mispricing / arbitrage strategy and printable tables. |
| 5 SABR Monte Carlo            | `views/sabr_mc.py`   | Simulation of the SABR SDEs with standard Euler-Maruyama or log-transform / full truncation, user-chosen number of paths and steps, distribution of the final price and the present value of the average payoff with standard error, compared with the Hagan + Black-76 price.                                                                     |

## Run

```bash
cd midterm
pip install -r requirements.txt
streamlit run app.py
```

Tests (pure-python core, no Streamlit needed):

```bash
pytest -q
```

## Layout

```
app.py              entry point: sidebar (ticker, risk-free rate, valuation date) + st.navigation
views/              one Streamlit script per calculator (+ home.py)
core/bs.py          Black-Scholes-Merton price, Greeks, implied vol, parity, Black-76 (ported from HW#5)
core/market_data.py yfinance helpers (HW#1 pattern), cached, failure-safe
core/portfolio.py   positions, valuation, payoff at expiry (HW#2)
core/arbitrage.py   put-call parity, Black-Scholes and binomial replication strategies (HW#4)
core/hedging.py     spot simulator, static hedge solver (HW#7) and daily rebalancing loop
core/sabr.py        Hagan formula, calibration (HW#8), data loader, smile grid, Monte Carlo
core/formulas.py    SymPy expressions for the Help dialogs (HW#6 engine)
ui/                 shared widgets, Help dialogs, HTML / CSV report builders
data/               sample data (AAPL snapshot from HW#8, LLY rows from HW#1, demo portfolio) + make_sample.py
tests/              pytest suite
```

## Conventions and assumptions

* Option type `call` / `put`, side `long` (+1) / `short` (-1); one contract = `multiplier` shares (100 by default).
* Time in years: calendar days / 365 for valuation dates; the hedging loop uses one trading day = 1/252.
* Continuous compounding. The sidebar can fetch the 13-week T-bill (`^IRX`) and converts it to ln(1 + y/100).
* Online data comes from Yahoo Finance through `yfinance`; every online field has a manual fallback, so the app works offline.
* SABR calibration follows HW#8: calls (or puts) with iv > 0, bid > 0, volume > 0, 0.7 F <= K <= 1.3 F, F = S0 e^{rT}, beta fixed at 1.
* Printing: use the browser print command on any page (the sidebar and buttons are hidden by the print CSS) or download the standalone HTML report.
