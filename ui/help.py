"""Dedicated Help button for every calculator: theory, assumptions and SymPy-rendered formulas."""
from __future__ import annotations

import streamlit as st

from core import formulas
from ui.common import WIDE

HELP = {
    "portfolio": {
        "title": "Help: Portfolio Evaluation",
        "theory": """
A **European option** gives its holder the right, not the obligation, to buy (**call**) or sell (**put**)
one share of the underlying stock at the strike price K on the expiry date T.

* **Intrinsic value** is what the option would pay if it were exercised right now: max(S-K, 0) for a call and
  max(K-S, 0) for a put. It can never be negative.
* **Market value** is the price at which the position could be closed today. It equals intrinsic value plus
  **time value**, the premium the market pays for the chance that the option ends deeper in the money.
* A **long** position (you bought the option) is an asset, a **short** position (you wrote the option) is a
  liability, so its value enters the portfolio with a negative sign.
* The **portfolio market value** is the sum over positions of sign x contracts x multiplier x unit price, and the
  **profit and loss (P&L)** compares that value with the premiums paid or received.

When no live quote is available the unit price is estimated with the Black-Scholes-Merton formula, which is
the arbitrage-free price of a European option when the stock follows a geometric Brownian motion with
constant volatility.
""",
        "assumptions": """
* Options are European on stocks; dividends, if any, are a continuous yield q.
* One contract covers `multiplier` shares (100 for US listed options); stock rows use a multiplier of 1.
* The spot price comes from Yahoo Finance (last close) or from your manual entry.
* Unit market price priority: your manual override, then the live option-chain mid price (bid+ask)/2,
  then the Black-Scholes value with the volatility you entered (default 25%).
* Time to expiry is measured in calendar days / 365 from the valuation date; the risk-free rate is continuous.
""",
        "formulas": [
            ("Intrinsic value of a call", "pf.intrinsic_call"),
            ("Intrinsic value of a put", "pf.intrinsic_put"),
            ("Market value of a position (s = +1 long, -1 short, Q contracts, m shares per contract)", "pf.market_value"),
            ("Time value", "pf.time_value"),
            ("Profit and loss versus the premium", "pf.pnl"),
            ("Black-Scholes d1", "bs.d1"),
            ("Black-Scholes d2", "bs.d2"),
            ("Black-Scholes call price (N = standard normal CDF)", "bs.call"),
            ("Black-Scholes put price", "bs.put"),
            ("Payoff of a call position at expiry", "pf.payoff"),
        ],
    },
    "arbitrage": {
        "title": "Help: Arbitrage Calculator",
        "theory": """
An **arbitrage** is a trading strategy that costs nothing today, can never lose money and makes a profit in at
least one state of the world. Option prices must rule it out, which gives two tests:

**1. Put-call parity (model free).** A call plus a zero-coupon bond paying K at T has exactly the same payoff
as a put plus the stock. Both are worth max(S_T, K) at expiry, so today C + K e^(-rT) = P + S e^(-qT).
The **synthetic call** is therefore "long put + long stock + borrow K e^(-rT)". If the market call is more
expensive than the synthetic call, sell the call and buy the synthetic (and vice versa): the payoffs cancel
at T whatever happens and the price difference is a riskless profit.

**2. Replication with shares and a bond.** An option can be manufactured from Delta shares of stock and a
bond B (a loan when B < 0). In the **one-period binomial model** the stock can only go up to S u or down to
S d, and the two unknowns Delta and B are chosen so the replica matches the option payoff in both nodes: the
value Delta S + B is then the only arbitrage-free price. In the **Black-Scholes model** Delta = dV/dS and the
replica must be rebalanced continuously, so a mispricing is a model-dependent (statistical) arbitrage rather
than a riskless one.

In every case the recipe is the same: **sell what is expensive, buy what is cheap, and hold until T**.
""",
        "assumptions": """
* Borrowing and lending happen at the same continuous risk-free rate r; short selling is allowed.
* Call and put used in the parity test share the same strike and expiry and are European.
* Dividends are a continuous yield q (buy e^(-qT) shares today and reinvest the dividends).
* The "threshold" input is the round-trip transaction cost per share: gaps smaller than it are not arbitrage.
* The binomial test needs d < e^(rT) < u, otherwise the tree itself admits an arbitrage between stock and bond.
* The Black-Scholes test assumes the volatility you enter is the true (constant) volatility.
""",
        "formulas": [
            ("Put-call parity", "parity.identity"),
            ("Synthetic call", "parity.synthetic_call"),
            ("Synthetic put", "parity.synthetic_put"),
            ("Mispricing gap (positive: call overpriced)", "parity.gap"),
            ("Replication with Delta shares and a bond", "rep.bs"),
            ("Bond leg", "rep.bs_bond"),
            ("Black-Scholes delta of a call", "bs.delta_call"),
            ("Black-Scholes delta of a put", "bs.delta_put"),
            ("Binomial delta (C_u, C_d = payoffs in the up and down nodes)", "rep.bin_delta"),
            ("Binomial bond", "rep.bin_bond"),
            ("Risk-neutral probability", "rep.bin_p"),
            ("Binomial option value", "rep.bin_value"),
            ("Cox-Ross-Rubinstein up move (d = 1/u)", "rep.crr"),
        ],
    },
    "hedging": {
        "title": "Help: Delta and Delta-Gamma Hedging",
        "theory": """
A trader who has **written** (sold) options is exposed to moves in the stock. **Delta** (dV/dS) measures how
much the option value changes per 1 unit move of the stock; **gamma** (d2V/dS2) measures how fast delta itself
changes.

* **Delta hedging**: hold N_u x Delta shares against N_u written options so that the combined delta is zero.
  Because delta drifts with the stock and with time, the share position has to be **rebalanced** every day.
* **Delta-gamma hedging**: stock has no gamma, so a second option is added to cancel the gamma of the written
  options; the leftover delta is then covered with stock. The hedge survives larger moves with fewer trades.
* **Dynamic bookkeeping**: the premium received funds the initial hedge; each rebalance is paid from (or
  credited to) a **cash account** that accrues interest at r. The hedged portfolio value is the sum of the
  option liability, the stock, the hedge options and cash; its daily change is the **P&L**. A perfect hedge
  keeps that P&L near zero; the unhedged writer instead feels the full option move.
* The **Spot Price Simulator** produces the daily stock path: random (geometric Brownian motion), typed in by
  hand, or the actual last closes from Yahoo Finance for a historical back-test.
""",
        "assumptions": """
* Black-Scholes world: constant volatility sigma and rate r, no transaction costs, fractional shares allowed.
* One trading day = 1/252 year; time to maturity shrinks by one day per row; positions are rebalanced once a day
  at the close. Discrete rebalancing leaves a residual P&L of order gamma x (daily move)^2.
* The written option premium is the Black-Scholes price at day 0 unless you enter another premium.
* Projections cover 1 to 5 days; the hedge option used for delta-gamma hedging is priced with the same sigma.
""",
        "formulas": [
            ("Delta of a call", "bs.delta_call"),
            ("Delta of a put", "bs.delta_put"),
            ("Gamma (same for calls and puts)", "bs.gamma"),
            ("Theta of a call (time decay per year)", "bs.theta_call"),
            ("Delta hedge: shares to hold against N_u written options", "hedge.delta_shares"),
            ("Delta-gamma hedge: number of hedge options", "hedge.dg_n"),
            ("Delta-gamma hedge: shares after the option leg", "hedge.dg_shares"),
            ("Initial cash account (premium received minus hedge cost)", "hedge.cash0"),
            ("Daily cash account (interest, then trades)", "hedge.cash"),
            ("Hedged portfolio value (= cumulative P&L)", "hedge.portfolio_value"),
            ("Remaining time to maturity on day d", "hedge.tau"),
            ("Spot simulator: geometric Brownian motion step", "hedge.gbm_step"),
        ],
    },
    "sabr": {
        "title": "Help: SABR Calculator",
        "theory": """
Market implied volatilities are not flat across strikes: they form a **smile** or **skew**. The **SABR model**
(Hagan, Kumar, Lesniewski and Woodward, 2002) captures this with a stochastic volatility for the forward price F:

* dF = sigma F^beta dW1 and d sigma = nu sigma dW2 with correlation rho between the two Brownian motions.
* **alpha** is the initial volatility level, **beta** the backbone (beta = 1 is lognormal, the standard choice for
  equities), **rho** controls the skew (negative for stocks) and **nu** the volatility of volatility, which bends
  the smile.

Hagan's asymptotic formula gives the Black implied volatility sigma_B(K, F) of the model in closed form, so an
option is priced by plugging sigma_B into the Black / Black-Scholes formula. For a non-dividend stock the forward
is F = S e^(rT).

**Calibration** fixes beta = 1 and finds alpha, rho and nu that minimise the squared difference between market
and model implied volatilities, one set per expiry. The **smile tab** then draws sigma_B across strikes for
several spot / forward levels, and the **mispricing tab** compares a quoted option with its SABR fair value and
proposes the replication trade (sell the expensive side, replicate with Delta shares and a bond).
""",
        "assumptions": """
* Non-dividend-paying stock, so F = S e^(rT); rates continuous; European options.
* Market implied volatilities are taken from the uploaded file (Yahoo Finance column) as given.
* Quotes used for calibration: calls (or puts) with iv > 0, bid > 0, volume > 0 and 0.7 F <= K <= 1.3 F
  (HW#8 filters); at least 5 strikes per expiry; beta fixed at 1 (alpha, rho, nu are fitted).
* Hagan's formula is an expansion accurate for moderate maturities and strikes not too far from the forward.
* The arbitrage strategy is a Black-Scholes replication: model-dependent, needs rebalancing.
""",
        "formulas": [
            ("SABR forward dynamics", "sabr.sde_F"),
            ("SABR volatility dynamics", "sabr.sde_sigma"),
            ("Correlation of the Brownian motions", "sabr.corr"),
            ("Forward of a non-dividend stock", "sabr.forward"),
            ("Hagan z", "sabr.z"),
            ("Hagan x(z)", "sabr.x"),
            ("Hagan implied volatility, general beta (eq. 2.17a)", "sabr.hagan_general"),
            ("Hagan implied volatility for beta = 1 (HW#8)", "sabr.hagan_beta1"),
            ("At-the-money volatility (eq. 2.18)", "sabr.hagan_atm"),
            ("Black-76 call price on the forward", "sabr.black76"),
            ("Black-76 d1 (d2 = d1 - sigma sqrt(T))", "sabr.black76_d1"),
            ("Calibration objective (least squares)", "sabr.calibration_objective"),
            ("Replication of a mispriced option", "rep.bs"),
        ],
    },
    "sabr_mc": {
        "title": "Help: SABR Monte Carlo",
        "theory": """
Instead of Hagan's approximation, the SABR price can be computed by **simulating** the two stochastic differential
equations. Time is cut into small steps dt and each path is advanced with random normal shocks:

* **Standard Euler-Maruyama**: F and sigma are each moved by drift x dt + diffusion x sqrt(dt) x Z. Simple, but F
  can be pushed below zero (it is absorbed at 0) and sigma can in principle turn negative for coarse steps.
* **Log-transform / full truncation** (production standard): sigma is a geometric Brownian motion, so its log is
  stepped exactly and stays positive; with beta = 1 the log of F is stepped the same way; for beta < 1 the
  diffusion uses max(F, 0)^beta ("full truncation"). Positivity is preserved and the discretisation bias is smaller.
* The two Brownian motions are correlated with a **Cholesky** step: W2 = rho Z1 + sqrt(1 - rho^2) Z2.
* Under the forward measure the forward has no drift, and at maturity the forward equals the stock price,
  so S_T = F_T. The option value is the **present value of the average payoff**, e^(-rT) x mean(max(F_T - K, 0)),
  with a standard error that shrinks like 1/sqrt(number of paths).
""",
        "assumptions": """
* Simulation under the T-forward measure with F0 = S0 e^(rT); discounting at the constant rate r.
* The number of paths controls the statistical error, the number of steps the discretisation error.
* Only 50 sample paths are stored for plotting; the histogram and statistics use every path.
* The Hagan + Black-76 price shown for comparison is itself an approximation, so small systematic gaps are normal.
""",
        "formulas": [
            ("SABR forward dynamics", "sabr.sde_F"),
            ("SABR volatility dynamics", "sabr.sde_sigma"),
            ("Correlated shocks (Cholesky)", "mc.cholesky"),
            ("Euler-Maruyama step for F", "mc.euler_F"),
            ("Euler-Maruyama step for sigma", "mc.euler_sigma"),
            ("Log-Euler step for F (beta = 1)", "mc.logeuler_F"),
            ("Exact lognormal step for sigma", "mc.logeuler_sigma"),
            ("Present value of the average payoff (call)", "mc.pv"),
            ("Standard error of the estimate (s = sample std of the payoff)", "mc.stderr"),
        ],
    },
}


# ---------------------------------------------------------------------------------------------------
# User guides: how to use each page (what every button and field does). Opened by the "Help" button.
# ---------------------------------------------------------------------------------------------------
SIDEBAR_GUIDE = """
**Sidebar (left, shared by every page)**
* **Default ticker**: the stock symbol proposed on every page (for example AAPL, MSFT). Type a Yahoo Finance symbol.
* **Fetch ^IRX (13-week T-bill)**: downloads today's US T-bill yield and converts it to a continuous rate.
  **Risk-free rate r**: or type your own rate (0.05 = 5%). Every page starts from this value.
* **Valuation date**: the "today" used to compute time to expiry (defaults to the real date).
* Navigation: click a module name to open it. The URL changes, so you can bookmark a page.
"""

GUIDE = {
    "portfolio": """
**What this page does**: values a list of option and stock positions. You type the positions, the app gets the stock
prices and computes intrinsic value, market value, time value and P&L.

**Step 1: fill the positions table**
* **Load sample portfolio**: fills the table with 4 demo positions (edit or delete them freely).
* **Clear table**: empties the table.
* Click the empty row at the bottom (or press **+**) to add a position; select a row and press **Delete** to remove it.
* Columns: `ticker` (Yahoo symbol), `kind` (call / put / stock), `side` (long = you bought, short = you sold),
  `quantity` (number of contracts, or number of shares for a stock row), `multiplier` (shares per contract, 100 in the US;
  put 1 for a stock row), `strike`, `expiry` (click the cell to pick a date), `premium` (price you paid or received per share),
  `vol` (volatility used if no market quote is available), `market_price` (optional: force a price per share).

**Step 2: spot prices** (one box per ticker in the table)
* **Fetch XXX online**: downloads the last close from Yahoo Finance into the box. The caption shows the source.
* Or simply type the price in the box.

**Step 3: valuation settings**
* **Risk-free rate r**, **Dividend yield q**, **Default volatility**: used by the Black-Scholes fallback price.
* **Use live option-chain mid prices**: tick it to price each option with the real bid/ask mid from Yahoo
  (needs a listed strike and expiry; otherwise Black-Scholes is used and a message tells you).

**Reading the results**
* The 5 big numbers are the portfolio totals (short positions count negatively).
* In the table, `price_source` tells where each price came from: manual, market mid or Black-Scholes.
* **Payoff at expiry**: one tab per ticker; dotted lines are the single positions, the thick line the whole portfolio,
  the dashed line the profit after premiums. The grey vertical line is the current spot.

**Print / export**: **Download table (CSV)** saves the valuation table; **Download printable report (HTML)** saves a
standalone page with totals, table and charts. Ctrl+P prints the page without the sidebar.
""",
    "arbitrage": """
**What this page does**: compares a market option price with a synthetic option made of stock and a bond, tells you
whether an arbitrage exists and lists the trades to do.

**Step 1: choose the construction** (radio buttons at the top)
* **Put-call parity**: you need a call and a put price with the same strike and expiry.
* **Black-Scholes replication**: one option price plus a volatility.
* **One-period binomial replication**: one option price plus the up/down factors u and d.

**Step 2: market data**
* **Ticker** and **Fetch XXX online** (last close) or type the **Spot price S**.
* **Strike K**; **Maturity input**: pick *Expiry date* (a calendar) or *Years* (a number such as 0.25).
* **Risk-free rate r** (prefilled from the sidebar), **Dividend yield q**.
* **Transaction-cost threshold per share**: gaps smaller than this are reported as "no arbitrage".
* **Shares per trade**: how many shares one contract covers (100); used to scale the profit.
* **Prefill option prices from the live option chain** (expander): choose a listed expiry, press
  **Fetch call and put quotes at the nearest strike** and the price boxes below are filled with Yahoo mid prices.

**Step 3: option prices**
* Parity: type the **Market call price** and **Market put price**.
* Replication modes: choose the **Option type**, type the **Market option price** and the **Volatility sigma**;
  in binomial mode you can also edit **u** and **d** (defaults come from sigma).

**Reading the results**
* A green box means an arbitrage was found, a blue box means prices are consistent.
* The 4 numbers: fair / synthetic value, mispricing per share, profit today and profit at T.
* **Step-by-step strategy**: execute the numbered trades in order.
* **Cash flows per share**: what you pay or receive today and at expiry in each scenario; the *Total* row shows the
  profit today and zero at expiry when the strategy is riskless.
* The chart shows the value of the legs at expiry (parity) or the option payoff versus the replica.

**Print / export**: CSV of the cash-flow table, HTML report with verdict, steps, table and chart.
""",
    "hedging": """
**What this page does**: you have sold (written) options; the page tells you how many shares (and optionally hedge
options) to hold each day, simulates the stock over 1 to 5 days and shows the P&L with and without hedging.

**Step 1: written option**
* **Ticker**, **Fetch XXX online** or type the **Spot price S**.
* **Option type**, **Strike K**, **Maturity input** (*Years* or *Expiry date*).
* **Risk-free rate r**, **Volatility sigma**, **Dividend yield q**.
* **Contracts written Q** and **Multiplier**: Q x multiplier = number of option units you are short.

**Step 2: hedge**
* **Delta hedge**: shares only.
* **Delta-gamma hedge**: also uses a second option; enter its **type**, **strike** and **T** in the boxes that appear.

**Step 3: Spot Price Simulator**
* **Forecast: random walk**: choose **Number of days**, **Drift**, **Simulated volatility** and a **Random seed**
  (change the seed to get another path).
* **Forecast: manual daily spots**: type the stock price of each future day in the small table.
* **Historical: actual last closes**: downloads the last closes of the ticker from Yahoo Finance; day 0 becomes the
  oldest close (the spot box is updated) so the hedge is back-tested on real prices.
* **Premium received = Black-Scholes price at day 0**: untick to type the premium you really received.

**Reading the results**
* **Day-0 hedge**: option price, Greeks, premium received, shares and hedge options to hold today, initial cash.
* Chart tabs: *Spot path*, *Cumulative P&L* (hedged versus unhedged), *Positions* (shares traded and held each day),
  *Residual Greeks* (should be near zero when the hedge works).
* **Daily adjustment report**: one line per day telling you what to BUY or SELL and the cash balance.
* **Daily table**: every number used (price, delta, gamma, shares held / traded, cash, P&L).
* **Summary**: final P&L of each strategy and best / worst day.

**Print / export**: CSV of the daily table, CSV of the P&L comparison, HTML report with all tables and charts.
""",
    "sabr": """
**What this page does**: computes SABR implied volatilities, calibrates the model on real option quotes, draws
volatility smiles and finds mispriced options. Work through the tabs from left to right.

**Tab 1: Hagan implied volatility** (manual parameters)
* **Underlying input**: give the *Spot S* (fetch online or type it; the forward is computed) or the *Forward F*.
* **r**, **T**, **Strike K**, then the SABR parameters **alpha**, **beta** (keep 1 for stocks), **rho**, **nu**.
* **Smile: K min / K max / number of strikes** define the curve drawn below.
* Results: forward, implied vol at K, ATM vol, call and put prices, the smile chart and its table.
* **Use these parameters in the other tabs**: stores alpha, beta, rho, nu, S, r, T for tabs 3, 4 and the Monte Carlo page.

**Tab 2: Calibrate from CSV / Excel**
* **Upload** a file (yfinance export like the HW#8 database, or the HW#1 Table_1 format), or press
  **Use the AAPL sample** / **Use the LLY sample**.
* Choose the **Ticker**; the **Spot S0** is inferred from the file when possible (edit it if needed); set **r**,
  **beta** (1), which **quotes** to fit (calls or puts), the **strike band** and the **minimum quotes per expiry**.
* Press **Calibrate**: one line per expiry appears with alpha, rho, nu and the fit error (rmse).
* **Expiries to plot**: pick the smiles to draw (dots = market, lines = SABR).
* **Expiry to use** then **Use this expiry**: sends that expiry's parameters to tabs 3, 4 and the Monte Carlo page.
* Downloads: parameters CSV, fitted quotes CSV, HTML report.

**Tab 3: Smile across S and F**
* **Vary**: spot levels or forward levels; type the **Levels** separated by commas; set the strike range.
* The chart draws one smile per level (market dots are added when the expiry was calibrated); the table lists S, F, T,
  K, SABR vol and the Black-Scholes call and put prices.

**Tab 4: Pricing and arbitrage**
* Enter **S**, **r**, **T**, **Option type**, **Strike K**, the **Market option price** (prefilled from the calibrated
  quotes when the strike exists), the **threshold** and **shares per trade**.
* Results: SABR vol, fair price, mispricing, verdict, numbered strategy, cash flows, a printable pricing table around K
  and the parameter table. Downloads: CSV and HTML report.
""",
    "sabr_mc": """
**What this page does**: prices an option by simulating the SABR model instead of using Hagan's formula.

**Inputs** (prefilled from the SABR page when you pressed *Use this expiry* or *Use these parameters*)
* **Underlying input**: *Spot S0* (the forward F0 = S0 e^rT is computed) or *Forward F0*.
* **r**, **Maturity T**, **Strike K**, **alpha**, **beta**, **rho**, **nu**: edit any value by hand.

**Simulation settings**
* **Option type**: call or put.
* **Discretisation scheme**: *Standard Euler-Maruyama* or *Log-transform / full truncation* (recommended).
* **Number of simulations (paths)**: more paths = smaller statistical error, longer run (100,000 takes a few seconds).
* **Time steps**: steps per path (default one per trading day).
* **Random seed**: same seed = same random numbers; change it to re-draw.
* Press **Run simulation**. The results stay on screen until you run again.

**Reading the results**
* **PV of the average payoff** is the Monte Carlo option price; **Standard error** and the **95% confidence interval**
  measure its precision; **Hagan + Black-76 price** is the closed-form comparison and the *delta* under it the gap.
* The summary table lists every statistic (final price mean and std, absorbed paths, runtime).
* Chart tabs: histogram of the final underlying price (red line = strike) and 50 sample paths of F and sigma.

**Print / export**: CSV of the summary and an HTML report with the charts.
""",
}


def render_guide(module_key: str) -> None:
    st.markdown(GUIDE[module_key])
    st.markdown("---")
    st.markdown(SIDEBAR_GUIDE)


def render_help(module_key: str) -> None:
    h = HELP[module_key]
    st.markdown("#### Theoretical background")
    st.markdown(h["theory"])
    st.markdown("#### Operational assumptions")
    st.markdown(h["assumptions"])
    st.markdown("#### Mathematical formulas (rendered with SymPy)")
    for caption, name in h["formulas"]:
        st.caption(caption)
        st.latex(formulas.latex(name))


def _open(title: str, renderer, module_key: str) -> None:
    if hasattr(st, "dialog"):
        st.dialog(title, width="large")(renderer)(module_key)
    else:  # pragma: no cover
        with st.expander(title, expanded=True):
            renderer(module_key)


def help_button(module_key: str) -> None:
    """Two buttons side by side: Help (how to use the page, as an overlay) and Theory (background + SymPy formulas)."""
    h = HELP[module_key]
    c1, c2 = st.columns(2)
    if c1.button("❔ Help", key=f"help_{module_key}", help="How to use this page: buttons, inputs and results", **WIDE):
        _open(h["title"].replace("Help:", "How to use:"), render_guide, module_key)
    if c2.button("📘 Theory", key=f"theory_{module_key}", help="Theoretical background, assumptions and formulas", **WIDE):
        _open(h["title"].replace("Help:", "Theory:"), render_help, module_key)
