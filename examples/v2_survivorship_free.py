"""v2.0 — the survivorship-free re-run, one command.

This is the payoff of the data-first ingestion backbone. It runs the structural signal
through ``honest_backtest`` twice — **survivors only** vs **survivors + delisted** — and
shows whether adding the death leg degrades or *inverts* the edge. That is the single
test the whole study turns on.

Data, in order of honesty:

* ``--lake PATH``         — read a real delisted-inclusive parquet lake via DataLakeSource
                            (the real answer; build one with --source below).
* ``--source fmp|polygon``— materialize a lake first from a *paid* feed (needs a key and a
                            plan that includes delisted history), then run on it.
* (default)               — a synthetic lake (offline): survivors are near-random-walks,
                            "delisted" names decline to pennies. This proves the PIPELINE
                            end-to-end now; the dead names are a *sensitivity estimate*,
                            not real history, until you point --lake/--source at real data.

    python examples/v2_survivorship_free.py                       # offline, synthetic
    python examples/v2_survivorship_free.py --lake /data/eod      # real lake
    FMP_API_KEY=... python examples/v2_survivorship_free.py --source fmp --materialize /tmp/lake
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vpts import build_structural_dataset, honest_backtest  # noqa: E402
from vpts.data import (  # noqa: E402
    DataLakeSource,
    KNOWN_DELISTED,
    materialize_lake,
    synthetic_delisted_ohlcv,
    synthetic_survivor_ohlcv,
)


def _frames_from_lake(root: str) -> tuple[dict, dict]:
    """Split a DataLakeSource universe into {survivor frames} and {delisted frames}."""
    src = DataLakeSource(root)
    uni = src.build_universe()
    surv = {s: src.get_bars(s, period="max") for s in uni.survivors()}
    dead = {s: src.get_bars(s, period="max") for s in uni.delisted()}
    return surv, dead


def _synthetic_frames(n_surv: int, n_dead: int) -> tuple[dict, dict]:
    surv = {f"SURV{i}": synthetic_survivor_ohlcv(820, seed=i, start_date="2014-01-01")
            for i in range(n_surv)}
    dead = {f"DEAD{i}": synthetic_delisted_ohlcv(520, seed=900 + i, start_date="2014-01-01",
                                                 terminal_frac=0.08)
            for i in range(n_dead)}
    return surv, dead


def _materialize_real(source_name: str, root: str, n_surv: int, start: str, end: str) -> str:
    """Pull a real universe (survivors + KNOWN_DELISTED) into a parquet lake; return root."""
    if source_name == "fmp":
        from vpts.data import FMPSource
        source = FMPSource(delisted_capable=True)
    elif source_name == "polygon":
        from vpts.data import PolygonSource
        source = PolygonSource()
    else:
        raise SystemExit(f"unknown --source {source_name!r} (use fmp or polygon)")
    survivors = ["AAPL", "MSFT", "JPM", "XOM", "KO", "PG", "JNJ", "WMT", "GE", "T"][:n_surv]
    dead = [t.symbol for t in KNOWN_DELISTED]
    caps = {t.symbol: f"{t.year}-12-31" for t in KNOWN_DELISTED}
    rep = materialize_lake(root, source, survivors + dead, start=start, end=end,
                           delist_caps=caps, delisted=dead)
    print(rep.summary(), "\n")
    return root


def _book(frames: dict, lookback: int, horizon: int, stride: int) -> list:
    out = []
    for sym, f in frames.items():
        try:
            out.append(build_structural_dataset(f, lookback=lookback, horizon=horizon,
                                                 stride=stride, symbol=sym, interval="1d"))
        except Exception as exc:  # noqa: BLE001 - too-short / degenerate frame
            print(f"  (skip {sym}: {type(exc).__name__})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lake", default=None, help="path to an existing parquet data lake")
    ap.add_argument("--source", default=None, choices=["fmp", "polygon"],
                    help="materialize a lake from a paid feed first")
    ap.add_argument("--csv-dir", default=None,
                    help="directory of OHLCV CSVs to ingest (e.g. ariva/broker exports)")
    ap.add_argument("--materialize", default=None,
                    help="lake root to write when using --source/--csv-dir")
    ap.add_argument("--n-survivors", type=int, default=6)
    ap.add_argument("--n-delisted", type=int, default=4)
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default="2018-12-31")
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.source:
        root = args.materialize or tempfile.mkdtemp(prefix="vpts_lake_")
        surv, dead = _frames_from_lake(_materialize_real(args.source, root, args.n_survivors,
                                                         args.start, args.end))
        provenance = f"REAL feed via {args.source} → lake {root}"
    elif args.csv_dir:
        from vpts.data import CsvSource
        root = args.materialize or tempfile.mkdtemp(prefix="vpts_lake_")
        csv_src = CsvSource(args.csv_dir)
        known_dead = [t.symbol for t in KNOWN_DELISTED]
        caps = {t.symbol: f"{t.year}-12-31" for t in KNOWN_DELISTED}
        rep = materialize_lake(root, csv_src, csv_src.available_symbols(),
                               delist_caps=caps, delisted=known_dead)
        print(rep.summary(), "\n")
        surv, dead = _frames_from_lake(root)
        provenance = f"CSV dir {args.csv_dir} → lake {root}"
    elif args.lake:
        surv, dead = _frames_from_lake(args.lake)
        provenance = f"REAL parquet lake {args.lake}"
    else:
        surv, dead = _synthetic_frames(args.n_survivors, args.n_delisted)
        provenance = "SYNTHETIC (offline) — dead names are a sensitivity estimate, not real history"

    print("=" * 84)
    print("v2.0 — survivorship-free re-run of the structural signal")
    print("=" * 84)
    print(f"Data: {provenance}")
    print(f"Universe: {len(surv)} survivors, {len(dead)} delisted\n")
    if not surv or not dead:
        print("Need at least one survivor AND one delisted name — nothing to compare.")
        return 1

    kw = dict(lookback=args.lookback, horizon=args.horizon, stride=args.stride)
    surv_ds = _book(surv, **kw)
    dead_ds = _book(dead, **kw)
    if not surv_ds or not dead_ds:
        print("Could not build datasets (frames too short for lookback+horizon).")
        return 1

    rep_surv = honest_backtest(surv_ds, perms=args.perms, n_trials=11, seed=args.seed)
    rep_all = honest_backtest(surv_ds + dead_ds, perms=args.perms, n_trials=11, seed=args.seed)

    print("--- SURVIVORS ONLY " + "-" * 65)
    print(rep_surv.summary())
    print("\n--- SURVIVORS + DELISTED " + "-" * 59)
    print(rep_all.summary())

    d_ic = rep_all.oos_ic_mean - rep_surv.oos_ic_mean
    d_ls = rep_all.long_short_net_pct - rep_surv.long_short_net_pct
    inverted = (np.isfinite(rep_surv.long_short_net_pct) and np.isfinite(rep_all.long_short_net_pct)
                and rep_surv.long_short_net_pct > 0 and rep_all.long_short_net_pct < 0)
    print("\n" + "=" * 84)
    print(f"Δ OOS IC      : {d_ic:+.3f}   ({rep_surv.oos_ic_mean:+.3f} → {rep_all.oos_ic_mean:+.3f})")
    print(f"Δ L/S net/bet : {d_ls:+.3f}%  ({rep_surv.long_short_net_pct:+.3f}% → "
          f"{rep_all.long_short_net_pct:+.3f}%)")
    print(f"Reading       : {'INVERTS — survivorship mirage confirmed on this data' if inverted else 'decays / holds (no sign flip)'}")
    print("Note: a SYNTHETIC run is a sensitivity estimate. Point --lake/--source at real")
    print("delisted data for the verdict that actually settles the question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
