"""Build a cross-sectional equity panel from raw daily price dumps.

Provenance: the daily price series are pulled from Financial Modeling Prep
(financialmodelingprep.com) via the FMP MCP tool — endpoint
``historical-price-eod-light`` (split-adjusted close + volume). This script reads
a directory of those JSON dumps (each a list of {symbol, date, price, volume})
and writes a ``real_panel.parquet`` in the schema the ``--data`` hook of
``gbrt_realistic_validation.py`` expects: ``date, symbol, <features...>,
fwd_ret, adv``.

HONEST CAVEATS (printed and meant to be read):
  * The universe is a fixed set of names that *survived* to the pull date, so it
    carries **survivorship/selection bias** — this is a methods demonstration on
    real prices, not a clean research backtest.
  * ``historical-price-eod-light`` is split-adjusted, not total-return
    (dividends not reinvested); a second-order effect over this horizon, but a
    real one, especially for high-yield names.
  * Features use only data up to each rebalance date; ``fwd_ret`` is strictly
    future (no look-ahead).

    python examples/build_real_panel.py --src <dir-of-json> --out real_panel.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def load_prices(src: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(src, "*.txt")) + glob.glob(os.path.join(src, "*.json"))):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if isinstance(data, list) and data and "symbol" in data[0]:
            rows.extend(data)
    if not rows:
        raise SystemExit(f"no parseable FMP price dumps found in {src}")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = (df.dropna(subset=["price"])
            .drop_duplicates(["symbol", "date"])
            .sort_values(["symbol", "date"]).reset_index(drop=True))
    return df


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="date", columns="symbol", values="price").sort_index()
    vol = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    ret = wide.pct_change()
    dollar_vol = (wide * vol)

    # month-end rebalance dates with >=252 trading days of prior history
    me = wide.resample("ME").last().index
    out = []
    idx = wide.index
    for d in me:
        pos = idx.searchsorted(d, side="right") - 1     # last trading day <= month-end
        if pos < 252 or pos + 21 >= len(idx):           # need history + a forward month
            continue
        t = idx[pos]
        p_t = wide.iloc[pos]
        feat = pd.DataFrame({
            "mom_12_1": wide.iloc[pos - 21] / wide.iloc[pos - 252] - 1.0,
            "mom_6":    p_t / wide.iloc[pos - 126] - 1.0,
            "mom_3":    p_t / wide.iloc[pos - 63] - 1.0,
            "rev_1m":  -(p_t / wide.iloc[pos - 21] - 1.0),          # short-term reversal
            "vol_3m":   ret.iloc[pos - 63:pos].std() * np.sqrt(252),
            "vol_12m":  ret.iloc[pos - 252:pos].std() * np.sqrt(252),
        })
        feat["fwd_ret"] = wide.iloc[pos + 21] / p_t - 1.0          # strictly future ~1m
        feat["adv"] = dollar_vol.iloc[pos - 21:pos].mean() / 1e6   # $mm avg daily
        feat["date"] = t
        feat["symbol"] = feat.index
        out.append(feat.dropna())
    panel = pd.concat(out, ignore_index=True)
    cols = ["date", "symbol", "mom_12_1", "mom_6", "mom_3", "rev_1m",
            "vol_3m", "vol_12m", "fwd_ret", "adv"]
    return panel[cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of FMP JSON price dumps")
    ap.add_argument("--out", default="real_panel.parquet")
    args = ap.parse_args()
    prices = load_prices(args.src)
    panel = build_panel(prices)
    if args.out.endswith(".csv"):
        panel.to_csv(args.out, index=False)
    else:
        panel.to_parquet(args.out, index=False)
    print(f"universe: {panel.symbol.nunique()} names, "
          f"{panel.date.nunique()} monthly cross-sections, {len(panel)} rows")
    print(f"window  : {panel.date.min().date()} .. {panel.date.max().date()}")
    print(f"wrote {args.out}")
    print("CAVEAT: survivorship-selected universe + split-adjusted (not total-return) "
          "prices — a real-data methods demonstration, not a clean research backtest.")


if __name__ == "__main__":
    main()
