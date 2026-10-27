"""Options Analytics Suite - FFD midterm web app (Streamlit).

Run from this folder:  streamlit run app.py
"""
import datetime as dt
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.common import inject_print_css, rate_input  # noqa: E402

st.set_page_config(page_title="Options Analytics Suite", page_icon="📈", layout="wide")
inject_print_css()

PAGES = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/portfolio.py", title="1 · Portfolio Evaluation", icon="💼", url_path="portfolio"),
    st.Page("views/arbitrage.py", title="2 · Arbitrage Calculator", icon="⚖️", url_path="arbitrage"),
    st.Page("views/hedging.py", title="3 · Delta / Delta-Gamma Hedging", icon="🛡️", url_path="hedging"),
    st.Page("views/sabr.py", title="4 · SABR Calculator", icon="📉", url_path="sabr"),
    st.Page("views/sabr_mc.py", title="5 · SABR Monte Carlo", icon="🎲", url_path="sabr-mc"),
]

with st.sidebar:
    st.title("Options Analytics Suite")
    st.caption("Global settings shared by every calculator")
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "AAPL"
    st.text_input("Default ticker", key="ticker", help="Used to fetch spot prices and option chains online")
    rate_input()
    if "asof" not in st.session_state:
        st.session_state["asof"] = dt.date.today()
    st.date_input("Valuation date", key="asof")
    st.caption("Tip: use the browser print command (Ctrl+P) on any page to print the tables and charts, "
               "or download the HTML report.")

st.navigation(PAGES).run()
