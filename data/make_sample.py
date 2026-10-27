"""One-off script that builds the small sample datasets shipped with the app.

Run from the midterm folder:  python data/make_sample.py

Sources (course homework folders, one level up):
  ../HW#8/Final Database Options .xlsx  -> data/sample_aapl_options.csv (latest AAPL snapshot, calls + puts)
  ../HW#1/Table_1.csv                   -> data/sample_lly_table1.csv   (LLY rows of the first scrape date)
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # the FFD course folder


def build_aapl() -> None:
    src = ROOT / "HW#8" / "Final Database Options .xlsx"
    d = pd.read_excel(src, sheet_name=0, engine="openpyxl")
    d = d.dropna(subset=["contractSymbol"])
    d[["ticker", "type"]] = d["contractSymbol"].str.extract(r"^([A-Z]+)\d{6}([CP])")
    # the file stacks daily snapshots, each starting with MSFT (same trick as HW#8)
    pull = ((d["ticker"] == "MSFT") & (d["ticker"].shift() != "MSFT")).cumsum()
    d = d[(pull == pull.max()) & (d["ticker"] == "AAPL")]
    cols = [c for c in d.columns if c not in ("ticker", "type", "Call_Put")]
    d[cols].to_csv(HERE / "sample_aapl_options.csv", index=False)
    print("AAPL rows written:", len(d))


def build_lly() -> None:
    src = ROOT / "HW#1" / "Table_1.csv"
    d = pd.read_csv(src)
    d = d[d["Ticker Symbol"] == "LLY"]
    d = d[d["Date"] == d["Date"].iloc[0]]
    d.to_csv(HERE / "sample_lly_table1.csv", index=False)
    print("LLY rows written:", len(d))


if __name__ == "__main__":
    build_lly()
    build_aapl()
