"""SABR model: Hagan implied volatility, calibration (HW#8), option-data loader, smile grid,
mispricing / arbitrage and Monte Carlo simulation of the SABR SDEs.

    dF     = sigma F^beta dW1
    dsigma = nu sigma dW2,      dW1 dW2 = rho dt,     F(0) = F0, sigma(0) = alpha
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from core.arbitrage import ArbResult, replicate_bs
from core.bs import black76_price, bs_price, normalize_type

P0 = (0.15, -0.5, 0.7)                                        # HW#8 initial alpha, rho, nu
BOUNDS = ([0.0, -0.999, 0.0], [np.inf, 0.999, np.inf])        # HW#8 bounds for alpha, rho, nu
SCHEMA = ["ticker", "option_type", "strike", "expiry", "last", "bid", "ask",
          "volume", "open_interest", "iv", "spot", "quote_date"]
SCHEMES = {"euler": "Standard Euler-Maruyama",
           "log_euler_full_truncation": "Log-transform / full truncation (production standard)"}


def _ret(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


# ------------------------------------------------------------------ Hagan (2002) implied volatility
def hagan_implied_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) eq. 2.17a, lognormal implied volatility, vectorised over K (and F).

    With lk = ln(F/K) and FK = (F K)^{(1-beta)/2}:
      z = (nu/alpha) FK lk,   x(z) = ln[(sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)]
      sigma_B = alpha / (FK [1 + (1-beta)^2/24 lk^2 + (1-beta)^4/1920 lk^4]) * z/x(z)
                * {1 + [(1-beta)^2/24 alpha^2/FK^2 + rho beta nu alpha/(4 FK) + (2-3rho^2)/24 nu^2] T}
    z/x(z) -> 1 at the money (eq. 2.18).  beta = 1 reduces to the HW#8 formula.
    """
    F, K = np.broadcast_arrays(np.asarray(F, dtype=float), np.asarray(K, dtype=float))
    if alpha <= 0 or nu < 0 or not (-1 < rho < 1):
        raise ValueError("alpha must be > 0, nu >= 0 and -1 < rho < 1")
    omb = 1.0 - beta
    lk = np.log(F / K)
    FK = (F * K) ** (omb / 2.0)
    z = (nu / alpha) * FK * lk
    x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    small = np.abs(z) < 1e-7
    ratio = np.where(small, 1.0, z / np.where(small, 1.0, x))
    pre = alpha / (FK * (1.0 + omb**2 / 24.0 * lk**2 + omb**4 / 1920.0 * lk**4))
    corr = 1.0 + (omb**2 / 24.0 * alpha**2 / FK**2 + rho * beta * nu * alpha / (4.0 * FK)
                  + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * T
    return _ret(pre * ratio * corr)


def hagan_atm_vol(F, T, alpha, beta, rho, nu) -> float:
    """Hagan eq. 2.18: at-the-money volatility."""
    return float(hagan_implied_vol(F, F, T, alpha, beta, rho, nu))


# ------------------------------------------------------------------ calibration
@dataclass
class SabrParams:
    alpha: float
    beta: float
    rho: float
    nu: float
    F: float
    T: float
    rmse: float = float("nan")
    n: int = 0

    def iv(self, K, F=None, T=None):
        return hagan_implied_vol(self.F if F is None else F, K, self.T if T is None else T,
                                 self.alpha, self.beta, self.rho, self.nu)

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "rho": self.rho, "nu": self.nu,
                "F": self.F, "T": self.T, "rmse": self.rmse, "n": self.n}


def calibrate_smile(strikes, ivs, F, T, beta: float = 1.0, p0=None, bounds=BOUNDS) -> SabrParams:
    """Fit alpha, rho, nu (beta fixed) to market implied vols of one maturity by least squares (HW#8 curve_fit)."""
    K = np.asarray(strikes, dtype=float)
    iv = np.asarray(ivs, dtype=float)
    ok = np.isfinite(K) & np.isfinite(iv) & (iv > 0) & (K > 0)
    K, iv = K[ok], iv[ok]
    if len(K) < 3:
        raise ValueError("at least 3 valid quotes are needed to calibrate a smile")

    def model(k, alpha, rho, nu):
        return hagan_implied_vol(F, k, T, alpha, beta, rho, nu)

    guesses = [tuple(p0)] if p0 is not None else [P0, (float(np.median(iv)), -0.3, 0.5)]
    best = None
    for g in guesses:
        try:
            popt, _ = curve_fit(model, K, iv, p0=g, bounds=bounds, max_nfev=20000)
        except (RuntimeError, ValueError):
            continue
        rmse = float(np.sqrt(np.mean((model(K, *popt) - iv) ** 2)))
        if best is None or rmse < best[1]:
            best = (popt, rmse)
    if best is None:
        raise RuntimeError("SABR calibration did not converge")
    popt, rmse = best
    return SabrParams(float(popt[0]), float(beta), float(popt[1]), float(popt[2]), float(F), float(T), rmse, int(len(K)))


def calibrate_surface(df: pd.DataFrame, S0: float, r: float, beta: float = 1.0, band=(0.7, 1.3),
                      option_type: str = "call", min_quotes: int = 5, asof=None):
    """HW#8 procedure: one (alpha, rho, nu) per expiry.

    Filters: option_type, expiry > valuation date, iv > 0, bid > 0 (when available), volume > 0,
    band[0] F <= K <= band[1] F with F = S0 e^{rT}, T = (expiry - valuation date) / 365.
    Returns (params DataFrame indexed by expiry, filtered quotes DataFrame with the SABR fit).
    """
    kind = normalize_type(option_type)
    d = df[df["option_type"] == kind].copy()
    if d.empty:
        raise ValueError(f"no {kind} quotes in the data")
    T0 = pd.Timestamp(asof) if asof is not None else pd.to_datetime(d["quote_date"]).max()
    T0 = T0.normalize()
    d["expiry"] = pd.to_datetime(d["expiry"])
    d = d[d["expiry"] > T0]
    d["T"] = (d["expiry"] - T0).dt.days / 365.0
    d["F"] = S0 * np.exp(r * d["T"])
    mask = (d["iv"] > 0) & (d["strike"] >= band[0] * d["F"]) & (d["strike"] <= band[1] * d["F"])
    if "bid" in d and d["bid"].notna().any():
        mask &= d["bid"].fillna(0) > 0
    if "volume" in d and d["volume"].notna().any():
        mask &= d["volume"].fillna(0) > 0
    d = d[mask]
    rows, fitted = [], []
    for expiry, g in d.groupby("expiry"):
        if len(g) < min_quotes:
            continue
        T, F = float(g["T"].iloc[0]), float(g["F"].iloc[0])
        try:
            prm = calibrate_smile(g["strike"], g["iv"], F, T, beta)
        except (RuntimeError, ValueError):
            continue
        rows.append({"expiry": expiry.date(), "T": T, "F": F, "n": prm.n, "alpha": prm.alpha, "beta": prm.beta,
                     "rho": prm.rho, "nu": prm.nu, "rmse": prm.rmse})
        gg = g[["strike", "iv", "last", "bid", "ask", "volume"]].copy()
        gg.insert(0, "expiry", expiry.date())
        gg.insert(1, "T", T)
        gg.insert(2, "F", F)
        gg["sabr_iv"] = prm.iv(gg["strike"].to_numpy())
        fitted.append(gg)
    if not rows:
        raise ValueError("no expiry had enough quotes to calibrate (try a wider band or another option type)")
    params = pd.DataFrame(rows).set_index("expiry")
    quotes = pd.concat(fitted, ignore_index=True).sort_values(["expiry", "strike"]).reset_index(drop=True)
    params.attrs["valuation_date"] = T0.date()
    return params, quotes


# ------------------------------------------------------------------ data loader
def _read(file, sheet_name=0) -> pd.DataFrame:
    name = str(getattr(file, "name", file)).lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")
    return pd.read_csv(file)


def normalize_options(raw: pd.DataFrame) -> pd.DataFrame:
    """Map an uploaded table to SCHEMA. Supports the HW#8 / yfinance export and the HW#1 Table_1 format."""
    cols = set(raw.columns)
    if "contractSymbol" in cols:                                   # HW#8 / yfinance
        d = raw.dropna(subset=["contractSymbol"]).copy()
        ext = d["contractSymbol"].astype(str).str.extract(r"^([A-Z]+)\d{6}([CP])")
        out = pd.DataFrame({
            "ticker": ext[0],
            "option_type": ext[1].map({"C": "call", "P": "put"}),
            "strike": pd.to_numeric(d["strike"], errors="coerce"),
            "expiry": pd.to_datetime(d["expirationDate"], errors="coerce"),
            "last": pd.to_numeric(d.get("lastPrice"), errors="coerce"),
            "bid": pd.to_numeric(d.get("bid"), errors="coerce"),
            "ask": pd.to_numeric(d.get("ask"), errors="coerce"),
            "volume": pd.to_numeric(d.get("volume"), errors="coerce"),
            "open_interest": pd.to_numeric(d.get("openInterest"), errors="coerce"),
            "iv": pd.to_numeric(d["impliedVolatility"], errors="coerce"),
            "spot": np.nan,
            "quote_date": pd.to_datetime(d["lastTradeDate"], errors="coerce").dt.normalize(),
        })
        if "inTheMoney" in cols:
            out["in_the_money"] = d["inTheMoney"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    elif "Ticker Symbol" in cols:                                   # HW#1 Table_1.csv
        d = raw.copy()
        out = pd.DataFrame({
            "ticker": d["Ticker Symbol"].astype(str).str.upper(),
            "option_type": d["Option Type"].astype(str).str.lower().str.strip(),
            "strike": pd.to_numeric(d["Strike Price"], errors="coerce"),
            "expiry": pd.to_datetime(d["Term"].astype(str).str.slice(0, 10), format="%d/%m/%Y", errors="coerce"),
            "last": pd.to_numeric(d["Last Price"], errors="coerce"),
            "bid": pd.to_numeric(d["Bid"], errors="coerce"),
            "ask": pd.to_numeric(d["Ask"], errors="coerce"),
            "volume": pd.to_numeric(d["Volume"], errors="coerce"),
            "open_interest": pd.to_numeric(d["Open Interest"], errors="coerce"),
            "iv": pd.to_numeric(d["Implied Volatility"], errors="coerce"),
            "spot": pd.to_numeric(d["Spot Price"], errors="coerce"),
            "quote_date": pd.to_datetime(d["Date"], format="%d/%m/%Y", errors="coerce"),
        })
    elif {"strike", "impliedVolatility"} <= cols:                  # raw yfinance chain with extra columns
        d = raw.copy()
        if "ticker" not in cols or not ({"expiry", "expirationDate"} & cols):
            raise ValueError("a raw yfinance chain needs ticker and expiry columns")
        exp_col = "expiry" if "expiry" in cols else "expirationDate"
        out = pd.DataFrame({
            "ticker": d["ticker"].astype(str).str.upper(),
            "option_type": d.get("option_type", d.get("type", "call")).astype(str).str.lower(),
            "strike": pd.to_numeric(d["strike"], errors="coerce"),
            "expiry": pd.to_datetime(d[exp_col], errors="coerce"),
            "last": pd.to_numeric(d.get("lastPrice"), errors="coerce"),
            "bid": pd.to_numeric(d.get("bid"), errors="coerce"),
            "ask": pd.to_numeric(d.get("ask"), errors="coerce"),
            "volume": pd.to_numeric(d.get("volume"), errors="coerce"),
            "open_interest": pd.to_numeric(d.get("openInterest"), errors="coerce"),
            "iv": pd.to_numeric(d["impliedVolatility"], errors="coerce"),
            "spot": pd.to_numeric(d.get("spot"), errors="coerce"),
            "quote_date": pd.to_datetime(d.get("quote_date", d.get("lastTradeDate")), errors="coerce"),
        })
    else:
        raise ValueError("Unrecognised file format. Expected yfinance columns (contractSymbol, strike, "
                         "impliedVolatility, expirationDate, ...) or HW#1 columns (Ticker Symbol, Strike Price, ...).")
    out = out.dropna(subset=["ticker", "option_type", "strike", "expiry", "iv"])
    out = out[out["option_type"].isin(["call", "put"])].reset_index(drop=True)
    return out


def load_options_file(file, sheet_name=0) -> pd.DataFrame:
    """Read a CSV / Excel upload and normalise it to SCHEMA."""
    return normalize_options(_read(file, sheet_name))


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent row of every contract; the valuation date becomes the latest quote date."""
    d = df.sort_values("quote_date").drop_duplicates(subset=["ticker", "option_type", "strike", "expiry"], keep="last")
    return d.reset_index(drop=True)


def infer_spot(df: pd.DataFrame, ticker: str):
    """Spot from a spot column, else the HW#8 ITM/OTM call-strike midpoint, else None."""
    d = df[df["ticker"] == ticker]
    if d.empty:
        return None
    if "spot" in d and d["spot"].notna().any():
        return float(d["spot"].dropna().median())
    if "in_the_money" in d:
        c = d[d["option_type"] == "call"]
        itm, otm = c.loc[c["in_the_money"], "strike"], c.loc[~c["in_the_money"], "strike"]
        if len(itm) and len(otm):
            return float((itm.max() + otm.min()) / 2)
    return None


# ------------------------------------------------------------------ smiles, pricing, arbitrage
def smile_grid(params: SabrParams, K_values, r: float, S_values=None, F_values=None, T=None,
               market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Implied-vol smile and Black-Scholes prices for combinations of spot/forward and strike.

    Either S_values (F = S e^{rT}) or F_values (S = F e^{-rT}) must be given.  For a non-dividend
    stock F = S e^{rT}.  market: optional DataFrame with strike, option_type, mid -> adds market
    prices and mispricing = market - model.
    """
    T = params.T if T is None else float(T)
    K = np.asarray(K_values, dtype=float)
    if S_values is not None:
        S_list = [float(s) for s in np.atleast_1d(S_values)]
        pairs = [(s, s * np.exp(r * T)) for s in S_list]
    elif F_values is not None:
        pairs = [(float(f) * np.exp(-r * T), float(f)) for f in np.atleast_1d(F_values)]
    else:
        raise ValueError("give S_values or F_values")
    frames = []
    for S, F in pairs:
        sig = hagan_implied_vol(F, K, T, params.alpha, params.beta, params.rho, params.nu)
        frames.append(pd.DataFrame({
            "S": S, "F": F, "T": T, "K": K, "sigma_sabr": sig,
            "call_bs": bs_price(S, K, T, r, 0.0, sig, "call"),
            "put_bs": bs_price(S, K, T, r, 0.0, sig, "put"),
        }))
    out = pd.concat(frames, ignore_index=True)
    if market is not None and not market.empty:
        for kind in ("call", "put"):
            m = market[market["option_type"] == kind][["strike", "mid"]].rename(columns={"strike": "K", "mid": f"mkt_{kind}"})
            out = out.merge(m.drop_duplicates("K"), on="K", how="left")
            out[f"{kind}_mispricing"] = out[f"mkt_{kind}"] - out[f"{kind}_bs"]
    return out


def price_with_sabr(S, K, T, r, params: SabrParams, option_type="call"):
    """(Hagan implied vol, Black-Scholes price) for a non-dividend stock with F = S e^{rT}."""
    F = S * np.exp(r * T)
    sig = float(hagan_implied_vol(F, K, T, params.alpha, params.beta, params.rho, params.nu))
    return sig, float(bs_price(S, K, T, r, 0.0, sig, option_type))


def sabr_arbitrage(S, K, T, r, params: SabrParams, option_type, market_price, threshold=0.0, units=100) -> ArbResult:
    """Mispricing versus the SABR/Hagan fair value with a Delta-shares + bond replication strategy."""
    sig, _ = price_with_sabr(S, K, T, r, params, option_type)
    res = replicate_bs(S, K, T, r, 0.0, sig, option_type, market_price, threshold, units)
    res.mode = "sabr_replication"
    res.details.update({"sigma_sabr": sig, "F": S * np.exp(r * T), "sabr_params": params.as_dict()})
    return res


# ------------------------------------------------------------------ Monte Carlo
@dataclass
class MCResult:
    F_T: np.ndarray
    sigma_T: np.ndarray
    paths_F: np.ndarray
    paths_sigma: np.ndarray
    t_grid: np.ndarray
    scheme: str
    n_paths: int
    n_steps: int
    n_absorbed: int
    n_neg_sigma: int
    seconds: float
    F0: float
    alpha: float
    beta: float
    rho: float
    nu: float
    T: float


def simulate_sabr(F0, alpha, beta, rho, nu, T, n_paths, n_steps, scheme="euler", seed=None, n_store=50) -> MCResult:
    """Simulate the SABR SDEs, vectorised over paths (loop over time steps only).

    euler (standard Euler-Maruyama):
        F_{i+1} = F_i + sigma_i (F_i^+)^beta sqrt(dt) W1,   sigma_{i+1} = sigma_i + nu sigma_i sqrt(dt) W2,
        then F <- max(F, 0) (absorption at zero so F^beta stays defined).
    log_euler_full_truncation (production standard):
        sigma_{i+1} = sigma_i exp(-nu^2 dt/2 + nu sqrt(dt) W2)           (exact lognormal step, sigma > 0)
        beta = 1:  ln F_{i+1} = ln F_i - sigma_i^2 dt/2 + sigma_i sqrt(dt) W1   (F > 0)
        beta < 1:  F_{i+1} = F_i + sigma_i (F_i^+)^beta sqrt(dt) W1, F <- max(F, 0)   (full truncation)
    W2 = rho Z1 + sqrt(1 - rho^2) Z2 (Cholesky).  Only n_store sample paths are kept in memory.
    """
    n_paths, n_steps = int(n_paths), int(n_steps)
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme}")
    if not (1 <= n_paths <= 2_000_000) or not (1 <= n_steps <= 2000) or n_paths * n_steps > 5e8:
        raise ValueError("limits: 1 <= n_paths <= 2,000,000, 1 <= n_steps <= 2000 and n_paths*n_steps <= 5e8")
    if alpha <= 0 or F0 <= 0 or T <= 0 or not (-1 < rho < 1) or nu < 0 or not (0 <= beta <= 1):
        raise ValueError("need F0 > 0, alpha > 0, T > 0, -1 < rho < 1, nu >= 0 and 0 <= beta <= 1")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sq = np.sqrt(dt)
    rho_c = np.sqrt(1.0 - rho * rho)
    n_store = int(min(n_store, n_paths))
    F = np.full(n_paths, float(F0))
    sig = np.full(n_paths, float(alpha))
    use_log = scheme == "log_euler_full_truncation" and beta == 1.0
    logF = np.log(F) if use_log else None
    paths_F = np.empty((n_steps + 1, n_store))
    paths_sigma = np.empty((n_steps + 1, n_store))
    paths_F[0], paths_sigma[0] = F[:n_store], sig[:n_store]
    n_neg_sigma = 0
    for i in range(1, n_steps + 1):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = z1
        w2 = rho * z1 + rho_c * z2
        if scheme == "euler":
            Fpos = np.maximum(F, 0.0)
            F = F + sig * Fpos**beta * sq * w1
            sig = sig + nu * sig * sq * w2
            n_neg_sigma += int(np.count_nonzero(sig < 0))
            F = np.maximum(F, 0.0)
        else:
            if use_log:
                logF = logF - 0.5 * sig * sig * dt + sig * sq * w1
            else:
                Fpos = np.maximum(F, 0.0)
                F = np.maximum(F + sig * Fpos**beta * sq * w1, 0.0)
            sig = sig * np.exp(-0.5 * nu * nu * dt + nu * sq * w2)
        paths_F[i] = np.exp(logF[:n_store]) if use_log else F[:n_store]
        paths_sigma[i] = sig[:n_store]
    if use_log:
        F = np.exp(logF)
    return MCResult(F_T=F, sigma_T=sig, paths_F=paths_F, paths_sigma=paths_sigma,
                    t_grid=np.linspace(0.0, T, n_steps + 1), scheme=scheme, n_paths=n_paths, n_steps=n_steps,
                    n_absorbed=int(np.count_nonzero(F <= 0)), n_neg_sigma=n_neg_sigma,
                    seconds=time.perf_counter() - t0, F0=float(F0), alpha=float(alpha), beta=float(beta),
                    rho=float(rho), nu=float(nu), T=float(T))


def mc_option_price(res: MCResult, K, r, option_type="call") -> dict:
    """Present value of the average payoff: PV = e^{-rT} mean(max(+/-(F_T - K), 0)), with the standard
    error, a 95% confidence interval and the Hagan + Black-76 reference price."""
    kind = normalize_type(option_type)
    payoff = np.maximum(res.F_T - K, 0.0) if kind == "call" else np.maximum(K - res.F_T, 0.0)
    disc = float(np.exp(-r * res.T))
    mean_payoff = float(payoff.mean())
    pv = disc * mean_payoff
    se = disc * float(payoff.std(ddof=1)) / np.sqrt(res.n_paths) if res.n_paths > 1 else float("nan")
    hagan_vol = float(hagan_implied_vol(res.F0, K, res.T, res.alpha, res.beta, res.rho, res.nu))
    ref = float(black76_price(res.F0, K, res.T, r, hagan_vol, kind))
    return {
        "option_type": kind, "K": float(K), "r": float(r), "discount_factor": disc,
        "mean_F_T": float(res.F_T.mean()), "std_F_T": float(res.F_T.std(ddof=1)),
        "mean_sigma_T": float(res.sigma_T.mean()),
        "mean_payoff": mean_payoff, "pv": pv, "std_err": se,
        "ci95_low": pv - 1.96 * se, "ci95_high": pv + 1.96 * se,
        "hagan_vol": hagan_vol, "hagan_black_pv": ref, "z_score": (pv - ref) / se if se > 0 else float("nan"),
        "n_paths": res.n_paths, "n_steps": res.n_steps, "scheme": res.scheme,
        "n_absorbed": res.n_absorbed, "n_neg_sigma": res.n_neg_sigma, "seconds": res.seconds,
    }
