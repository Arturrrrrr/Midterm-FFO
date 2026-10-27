"""Printable HTML reports (Jinja2 + embedded Plotly) and CSV downloads."""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import streamlit as st
from jinja2 import Template

_TEMPLATE = Template("""<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
 body { font-family: "Segoe UI", Arial, sans-serif; margin: 28px; color: #222; }
 h1 { font-size: 22px; margin-bottom: 4px; } h2 { font-size: 17px; margin-top: 30px; border-bottom: 1px solid #ccc; }
 .meta { color: #666; font-size: 12px; } .notes { font-size: 13px; line-height: 1.45; }
 table.tbl { border-collapse: collapse; font-size: 12px; margin: 8px 0; }
 table.tbl th, table.tbl td { border: 1px solid #bbb; padding: 4px 8px; text-align: right; }
 table.tbl th { background: #f0f0f0; } table.tbl td:first-child, table.tbl th:first-child { text-align: left; }
 ol li { margin-bottom: 4px; }
 @media print { .page-break { page-break-before: always; } }
</style></head>
<body>
<h1>{{ title }}</h1>
<p class="meta">Generated {{ generated }}{% for k, v in meta.items() %} &middot; {{ k }}: {{ v }}{% endfor %}</p>
{% for s in sections %}
<h2>{{ s.heading }}</h2>
{% if s.text %}<div class="notes">{{ s.text }}</div>{% endif %}
{% if s.steps %}<ol>{% for x in s.steps %}<li>{{ x }}</li>{% endfor %}</ol>{% endif %}
{% if s.table %}{{ s.table }}{% endif %}
{% if s.fig %}{{ s.fig }}{% endif %}
{% endfor %}
</body></html>""", autoescape=False)


def _text_html(text: str) -> str:
    """Minimal markdown-ish to HTML: blank lines separate paragraphs, **bold** supported."""
    paras = [p.strip() for p in str(text).split("\n\n") if p.strip()]
    out = []
    for p in paras:
        p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p).replace("\n", "<br>")
        out.append(f"<p>{p}</p>")
    return "".join(out)


def build_html_report(title: str, sections: list, meta: dict | None = None) -> str:
    """sections: dicts with heading and any of text, steps (list), table (DataFrame), fig (plotly Figure)."""
    rendered, first_fig = [], True
    for s in sections:
        d = {"heading": s.get("heading", ""), "text": None, "steps": s.get("steps"), "table": None, "fig": None}
        if s.get("text"):
            d["text"] = _text_html(s["text"])
        tbl = s.get("table")
        if tbl is not None and isinstance(tbl, pd.DataFrame) and not tbl.empty:
            d["table"] = tbl.to_html(index=s.get("index", False), classes="tbl", border=0,
                                     float_format=lambda x: f"{x:,.4f}", na_rep="")
        fig = s.get("fig")
        if fig is not None:
            d["fig"] = fig.to_html(full_html=False, include_plotlyjs="cdn" if first_fig else False,
                                   default_width="100%", default_height="420px")
            first_fig = False
        rendered.append(d)
    return _TEMPLATE.render(title=title, generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            meta=meta or {}, sections=rendered)


def _key(prefix: str, filename: str) -> str:
    return prefix + "_" + re.sub(r"[^a-z0-9]+", "_", filename.lower())


def download_report_button(html: str, filename: str, label: str = "Download printable report (HTML)") -> None:
    st.download_button(label, data=html.encode("utf-8"), file_name=filename, mime="text/html",
                       key=_key("dl_html", filename))


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "Download table (CSV)") -> None:
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv",
                       key=_key("dl_csv", filename))
