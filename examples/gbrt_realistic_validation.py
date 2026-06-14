"""Real-world-scale validation of the defensible recipe.

Purged + embargoed CPCV **and** a cost-aware, capacity-limited, time-ordered
long/short walk-forward, on a point-in-time-style equity panel.

HONEST DATA NOTE
----------------
CRSP / WRDS point-in-time data is paywalled and not reachable from this
execution environment (no network egress to a data vendor), so this script runs
on a **realistic *simulated*** date x name panel calibrated to documented
stylized facts, NOT on real CRSP returns:

  * point-in-time entry/exit (names list and delist; the cross-section is a
    survivorship-free, time-varying universe — no look-ahead survivorship);
  * a market + style factor structure with heterogeneous betas;
  * volatility regimes — a 2020 COVID-scale vol spike and an elevated 2022 —
    so the regime split has real contrast;
  * a weak, **decaying** cross-sectional signal (alpha that fades over the
    sample, in the spirit of McLean-Pontiff post-publication decay), so the
    decay diagnostic has something true to find.

The pipeline is data-agnostic. Point it at a real panel with
``--data PANEL.parquet`` (columns: ``date``, ``symbol``, ``f0..fk``,
``fwd_ret``; optionally ``adv`` for the capacity cap) and the identical
analysis runs on real data.

WHAT IT REPORTS (all out-of-sample)
-----------------------------------
  1. Purged CPCV: directional accuracy vs the **up-rate** (never 50%), OOS
     rank-IC dispersion, deflated & probabilistic Sharpe on the IC stream, and
     feature-importance stability across folds.
  2. A cost-aware walk-forward L/S book: turnover, net Sharpe / PSR / deflated
     Sharpe, and cumulative P&L under a **cost sweep** (bps per unit traded),
     with capacity limits and per-name position caps.
  3. Regime splits: pre-/post-2020 and calm/stressed volatility.
  4. Baselines: always-long, single-factor momentum, and a shuffled null.
  5. Performance decay across the OOS timeline (thirds).

    python examples/gbrt_realistic_validation.py
    python examples/gbrt_realistic_validation.py --names 1000 --rebalance 21
    python examples/gbrt_realistic_validation.py --no-decay   # stationary alpha
    python examples/gbrt_realistic_validation.py --null       # alpha switched off
    python examples/gbrt_realistic_validation.py --data my_panel.parquet
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vpts.ml.gbrt import GradientBoostedTrees
from vpts.ml.models import CrossSectionalPanel
from vpts.ml.significance import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from vpts.validation.cpcv import CombinatorialPurgedCV

N_FEATURES = 8
FEATURE_NAMES = tuple(["f0_alpha", "f1_alpha_weak"] + [f"f{i}_noise" for i in range(2, N_FEATURES)])


# --------------------------------------------------------------------------- #
# 1. A realistic, point-in-time-style simulated panel
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Panel:
    """A CrossSectionalPanel plus the auxiliary arrays the backtest needs."""

    panel: CrossSectionalPanel
    row_date: np.ndarray        # (n_rows,) int date index
    row_name: np.ndarray        # (n_rows,) int name id
    row_adv: np.ndarray         # (n_rows,) per-name liquidity proxy (ADV, $mm)
    date_ts: pd.DatetimeIndex   # (n_dates,) rebalance timestamps
    date_vol: np.ndarray        # (n_dates,) realized vol regime level
    stressed: np.ndarray        # (n_dates,) bool: high-vol regime


def _safe_ic(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman IC, returning NaN (not a warning) when either input is constant."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 5 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    rho = spearmanr(a, b).correlation
    return float(rho) if rho is not None and np.isfinite(rho) else float("nan")


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")


def _xs_rank(v: np.ndarray) -> np.ndarray:
    """Cross-sectional rank in [-0.5, 0.5] (ties broken by order)."""
    n = v.size
    if n < 2:
        return np.zeros(n)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(n)
    r[order] = np.arange(n)
    return r / (n - 1) - 0.5


def generate_panel(
    n_names: int, start: str, end: str, rebalance: int, *,
    seed: int, decay: bool, null: bool,
) -> Panel:
    """A survivorship-free date x name panel with factor structure, vol regimes,
    and a (decaying) cross-sectional alpha embedded in ``f0``/``f1``."""
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, end)
    reb = bdays[::rebalance]
    n_dates = len(reb)

    # A potential-name superset; each name lists and (hazard-rate) delists, so
    # the alive cross-section varies in time and there is no survivorship.
    n_super = int(round(n_names * 1.4))
    list_date = rng.integers(0, max(1, int(n_dates * 0.6)), size=n_super)
    life = rng.integers(int(n_dates * 0.25), n_dates + 1, size=n_super)
    delist_date = np.minimum(list_date + life, n_dates)
    beta = rng.normal(1.0, 0.35, size=n_super).clip(0.2, 2.2)        # market beta
    s_load = rng.normal(0.0, 1.0, size=n_super)                      # style loading
    adv = np.exp(rng.normal(3.0, 1.1, size=n_super))                 # ADV ($mm), lognormal

    # Volatility regime path: calm baseline, COVID spike (2020 Q1-Q2), 2022 stress.
    base_vol = 0.045 + 0.01 * np.sin(np.arange(n_dates) / 9.0)
    yr = reb.year.to_numpy()
    mo = reb.month.to_numpy()
    covid = (yr == 2020) & (mo >= 2) & (mo <= 5)
    y2022 = (yr == 2022) & (mo >= 1) & (mo <= 10)
    vol = base_vol.copy()
    vol[covid] *= 3.2
    vol[y2022] *= 1.7
    vol += rng.normal(0, 0.004, size=n_dates).clip(-0.01, 0.05)
    vol = vol.clip(0.02, 0.30)
    stressed = vol > np.quantile(vol, 0.80)

    # Decaying alpha strength: full early, fading to ~20% by the end if decay on.
    if null:
        alpha_t = np.zeros(n_dates)
    elif decay:
        alpha_t = np.linspace(1.0, 0.2, n_dates)
    else:
        alpha_t = np.ones(n_dates)
    alpha0 = 0.012                                   # base per-rank alpha (tuned to a
    #                                                  realistic GKX-scale OOS IC ~0.02-0.04
    #                                                  at the default universe — not a toy
    #                                                  signal, and small enough that costs bite)

    X_rows, y_rows, d_rows, n_rows_, adv_rows = [], [], [], [], []
    for t in range(n_dates):
        alive = np.flatnonzero((list_date <= t) & (t < delist_date))
        if alive.size < 20:
            continue
        m = alive.size
        raw = rng.standard_normal((m, N_FEATURES))
        ranks = np.column_stack([_xs_rank(raw[:, j]) for j in range(N_FEATURES)])
        sig = alpha_t[t] * alpha0 * (ranks[:, 0] + 0.4 * ranks[:, 1])  # f0 strong, f1 weak
        mkt = rng.normal(0.004, vol[t])                                # market factor
        sty = rng.normal(0.0, vol[t] * 0.6)                            # style factor
        idio = rng.standard_normal(m) * vol[t]
        y = sig + beta[alive] * mkt + s_load[alive] * sty + idio
        X_rows.append(ranks)
        y_rows.append(y)
        d_rows.append(np.full(m, t))
        n_rows_.append(alive)
        adv_rows.append(adv[alive])

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    row_date = np.concatenate(d_rows)
    row_name = np.concatenate(n_rows_)
    row_adv = np.concatenate(adv_rows)
    horizon = rebalance
    panel = CrossSectionalPanel(
        X=X, y=y, date_id=row_date, feature_names=FEATURE_NAMES,
        horizon=horizon, rebalance=rebalance, n_dates=n_dates,
        symbols=tuple(f"S{i:04d}" for i in range(n_super)), dates=reb,
    )
    return Panel(panel, row_date, row_name, row_adv, reb, vol, stressed)


def load_panel(path: str, rebalance: int) -> Panel:
    """Load a real panel: columns date, symbol, f0..fk, fwd_ret [, adv]."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    feats = [c for c in df.columns if c not in ("date", "symbol", "fwd_ret", "adv")]
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    dmap = {d: i for i, d in enumerate(dates)}
    syms = sorted(df["symbol"].unique())
    smap = {s: i for i, s in enumerate(syms)}
    # cross-sectionally rank features within each date (matches the synthetic schema)
    X = np.zeros((len(df), len(feats)))
    for d, idx in df.groupby("date").groups.items():
        rows = df.loc[idx]
        for j, f in enumerate(feats):
            X[rows.index, j] = _xs_rank(rows[f].to_numpy())
    row_date = df["date"].map(dmap).to_numpy()
    row_name = df["symbol"].map(smap).to_numpy()
    adv = df["adv"].to_numpy() if "adv" in df.columns else np.ones(len(df))
    panel = CrossSectionalPanel(
        X=X, y=df["fwd_ret"].to_numpy(float), date_id=row_date,
        feature_names=tuple(feats), horizon=rebalance, rebalance=rebalance,
        n_dates=len(dates), symbols=tuple(syms), dates=dates,
    )
    # vol regime from realized cross-sectional dispersion per date
    vol = df.groupby("date")["fwd_ret"].std().reindex(dates).to_numpy()
    stressed = vol > np.nanquantile(vol, 0.80)
    return Panel(panel, row_date, row_name, adv, dates, vol, stressed)


# --------------------------------------------------------------------------- #
# 2. Purged CPCV: IC, directional-vs-up-rate, significance, importance stability
# --------------------------------------------------------------------------- #
def _date_rows(date_id: np.ndarray, n_dates: int) -> list[np.ndarray]:
    groups = [np.empty(0, int)] * n_dates
    order = np.argsort(date_id, kind="mergesort")
    sid = date_id[order]
    for ch in np.split(order, np.flatnonzero(np.diff(sid)) + 1):
        if ch.size:
            groups[int(date_id[ch[0]])] = ch
    return groups


def cpcv_block(P: Panel, gbrt_kw: dict, n_trials: int) -> dict:
    panel = P.panel
    X, y = panel.X, panel.y
    groups = _date_rows(panel.date_id, panel.n_dates)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2,
                               purge=panel.purge_dates, embargo_pct=0.02)
    oos_ic, importances = [], []
    correct = total = up = 0
    for split in cv.split(panel.n_dates):
        tr = (np.concatenate([groups[d] for d in split.train_idx])
              if split.train_idx.size else np.empty(0, int))
        if tr.size < max(200, 8 * X.shape[1]):
            continue
        model = GradientBoostedTrees(**gbrt_kw).fit(X[tr], y[tr])
        importances.append(model.feature_importances_)
        for d in split.test_idx:
            g = groups[d]
            if g.size < 5:
                continue
            pred, real = model.predict(X[g]), y[g]
            rho = spearmanr(pred, real).correlation
            if rho is not None and np.isfinite(rho):
                oos_ic.append(float(rho))
            correct += int(np.sum((pred > np.median(pred)) == (real > 0)))
            total += int(real.size)
            up += int(np.sum(real > 0))
    ic = np.array(oos_ic)
    imp = np.vstack(importances)                       # (folds, features)
    up_rate = up / total
    dir_acc = correct / total
    majority = max(up_rate, 1 - up_rate)
    dsr = deflated_sharpe_ratio(ic, n_trials=n_trials, sr_std=0.05)
    # importance stability: per-feature mean/std + mean pairwise Spearman of fold rankings
    rank_corrs = [spearmanr(imp[a], imp[b]).correlation
                  for a in range(len(imp)) for b in range(a + 1, len(imp))]
    rank_corrs = [c for c in rank_corrs if c is not None and np.isfinite(c)]
    return dict(
        ic_mean=float(ic.mean()), ic_median=float(np.median(ic)), ic_std=float(ic.std()),
        pct_pos=float((ic > 0).mean() * 100), n_dates_oos=int(ic.size),
        up_rate=up_rate, dir_acc=dir_acc, majority=majority,
        edge_pp=(dir_acc - majority) * 100,
        psr=dsr.psr_vs_zero, dsr=dsr.deflated_sr, dsr_sig=dsr.significant,
        imp_mean=imp.mean(0), imp_std=imp.std(0),
        imp_rank_stability=float(np.mean(rank_corrs)) if rank_corrs else float("nan"),
    )


# --------------------------------------------------------------------------- #
# 3. Cost-aware, capacity-limited walk-forward long/short
# --------------------------------------------------------------------------- #
def _book_weights(score: np.ndarray, adv: np.ndarray, capital: float,
                  w_max: float, cap_frac: float) -> np.ndarray:
    """Dollar-neutral, gross=1 L/S weights with per-name and ADV capacity caps."""
    s = score - score.mean()
    if not np.any(np.abs(s) > 0):
        return np.zeros_like(s)
    w = s / np.sum(np.abs(s))
    adv_cap = np.maximum(cap_frac * adv / max(capital, 1e-9), 1e-6)   # |w| <= ADV share
    cap = np.minimum(w_max, adv_cap)
    w = np.clip(w, -cap, cap)
    w = w - w.mean()                                                 # restore neutrality
    g = np.sum(np.abs(w))
    return w / g if g > 0 else w


def _make_scorer(kind: str, gbrt_kw: dict, rng: np.random.Generator):
    if kind == "momentum":
        return lambda Xtr, ytr, Xte: Xte[:, 0]            # single-factor: f0 rank
    if kind == "long":
        return lambda Xtr, ytr, Xte: np.ones(Xte.shape[0])
    if kind == "null":
        return lambda Xtr, ytr, Xte: rng.standard_normal(Xte.shape[0])
    # "gbrt": refit handled by caller; placeholder
    return None


def walk_forward(P: Panel, gbrt_kw: dict, *, train_window: int, refit_every: int,
                 cost_bps_grid: list[float], capital: float, w_max: float,
                 cap_frac: float, seed: int) -> dict:
    panel = P.panel
    X, y = panel.X, panel.y
    groups = _date_rows(panel.date_id, panel.n_dates)
    purge = panel.purge_dates
    warmup = max(train_window, 24)
    rng = np.random.default_rng(seed + 7)

    def run_strategy(kind: str) -> dict:
        scorer = _make_scorer(kind, gbrt_kw, np.random.default_rng(seed + 11))
        long_only = kind == "long"
        model = None
        prev_w = {}                                       # name -> weight (for turnover)
        recs = []                                         # per-period records
        for t in range(warmup, panel.n_dates):
            g = groups[t]
            if g.size < 10:
                continue
            if kind == "gbrt":
                if model is None or (t - warmup) % refit_every == 0:
                    lo = max(0, t - train_window)
                    tr_dates = range(lo, t - purge)
                    tr = np.concatenate([groups[d] for d in tr_dates
                                         if groups[d].size]) if t - purge > lo else np.empty(0, int)
                    if tr.size < max(200, 8 * X.shape[1]):
                        continue
                    model = GradientBoostedTrees(**gbrt_kw).fit(X[tr], y[tr])
                score = model.predict(X[g])
            else:
                score = scorer(None, None, X[g])
            adv = P.row_adv[g]
            if long_only:
                w = np.ones(g.size) / g.size
            else:
                w = _book_weights(score, adv, capital, w_max, cap_frac)
            names = P.row_name[g]
            ret = float(np.sum(w * y[g]))
            ic = _safe_ic(score, y[g])
            # turnover vs previous book (names absent => prior weight 0)
            cur = dict(zip(names.tolist(), w.tolist()))
            alln = set(cur) | set(prev_w)
            traded = sum(abs(cur.get(n, 0.0) - prev_w.get(n, 0.0)) for n in alln)
            prev_w = cur
            recs.append((t, ret, traded, ic))
        if not recs:
            return {}
        tt = np.array([r[0] for r in recs])
        gross = np.array([r[1] for r in recs])
        traded = np.array([r[2] for r in recs])          # two-way notional turned over
        icw = np.array([r[3] for r in recs])
        turnover_1w = 0.5 * traded
        ppy = 252.0 / panel.rebalance
        out = dict(t=tt, gross=gross, turnover=turnover_1w, ic=icw, ppy=ppy,
                   nets={})
        for c in cost_bps_grid:
            net = gross - c * 1e-4 * traded
            out["nets"][c] = net
        return out

    res = {k: run_strategy(k) for k in ("gbrt", "momentum", "long", "null")}
    return res


def _perf(net: np.ndarray, ppy: float, n_trials: int) -> dict:
    sr = sharpe_ratio(net)
    sr_ann = sr * np.sqrt(ppy) if np.isfinite(sr) else float("nan")
    mu, sd = net.mean(), net.std()
    sk = float(((net - mu) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    ku = float(((net - mu) ** 4).mean() / sd ** 4) if sd > 0 else 3.0
    psr = probabilistic_sharpe_ratio(sr, net.size, skew=sk, kurt=ku)
    dsr = deflated_sharpe_ratio(net, n_trials=n_trials, sr_std=0.05)
    cum = float(np.prod(1.0 + net) - 1.0)
    return dict(sr_obs=sr, sr_ann=sr_ann, ann_ret=float(mu * ppy),
                psr=psr, dsr=dsr.deflated_sr, cum=cum, n=int(net.size))


# --------------------------------------------------------------------------- #
# 4. Reporting
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--names", type=int, default=500, help="approx universe size")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--rebalance", type=int, default=21, help="trading days between rebalances")
    ap.add_argument("--cost-bps", type=float, nargs="+", default=[0.05, 0.5, 5.0],
                    help="per-trade cost(s) in bps of traded notional")
    ap.add_argument("--capital", type=float, default=100.0, help="book size ($mm) for the ADV cap")
    ap.add_argument("--w-max", type=float, default=0.02, help="per-name gross weight cap")
    ap.add_argument("--cap-frac", type=float, default=0.05, help="max position as fraction of name ADV")
    ap.add_argument("--train-window", type=int, default=36, help="rolling train window (rebalances)")
    ap.add_argument("--refit-every", type=int, default=3, help="GBRT refit cadence (rebalances)")
    ap.add_argument("--n-trials", type=int, default=50, help="declared search size for deflated Sharpe")
    ap.add_argument("--no-decay", action="store_true", help="stationary alpha (no signal decay)")
    ap.add_argument("--null", action="store_true", help="switch the alpha off entirely")
    ap.add_argument("--data", default=None, help="real panel parquet/csv (date,symbol,f*,fwd_ret[,adv])")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="optional path to dump the results dict")
    args = ap.parse_args()

    gbrt_kw = dict(n_estimators=60, learning_rate=0.05, max_depth=3,
                   min_samples_leaf=20, subsample=0.7, random_state=args.seed)

    if args.data:
        P = load_panel(args.data, args.rebalance)
        src = f"real panel: {args.data}"
    else:
        P = generate_panel(args.names, args.start, args.end, args.rebalance,
                           seed=args.seed, decay=not args.no_decay, null=args.null)
        src = (f"SIMULATED point-in-time panel (CRSP unavailable in this environment) — "
               f"decay={'off' if args.no_decay else 'on'}, alpha={'OFF' if args.null else 'on'}")

    panel = P.panel
    avg_alive = len(panel) / panel.n_dates
    print("=" * 78)
    print("REAL-WORLD-SCALE VALIDATION OF THE DEFENSIBLE RECIPE")
    print("=" * 78)
    print(f"Source        : {src}")
    print(f"Universe      : {panel.n_names} potential names, {avg_alive:.0f} alive/date avg, "
          f"{len(panel):,} rows")
    print(f"Window        : {P.date_ts[0].date()} .. {P.date_ts[-1].date()}  "
          f"({panel.n_dates} rebalances, every {panel.rebalance}td, horizon {panel.horizon}td)")
    print(f"Stressed dates: {int(P.stressed.sum())}/{panel.n_dates} "
          f"(vol regime; incl. 2020 COVID + 2022)")

    # --- 1. CPCV ---------------------------------------------------------- #
    print("\n" + "-" * 78)
    print("1. PURGED + EMBARGOED CPCV  (robustness / significance)")
    print("-" * 78)
    cp = cpcv_block(P, gbrt_kw, args.n_trials)
    print(f"  OOS rank-IC    : mean {cp['ic_mean']:+.3f}  median {cp['ic_median']:+.3f}  "
          f"σ {cp['ic_std']:.3f}  ({cp['pct_pos']:.0f}% dates >0, {cp['n_dates_oos']} OOS dates)")
    print(f"  Dir. accuracy  : {cp['dir_acc']*100:.1f}%  vs up-rate {cp['up_rate']*100:.1f}% / "
          f"majority {cp['majority']*100:.1f}%  -> edge {cp['edge_pp']:+.2f}pp")
    print(f"  PSR vs 0       : {cp['psr']:.3f}     Deflated SR ({args.n_trials} trials): "
          f"{cp['dsr']:.3f}  -> {'SIGNIFICANT' if cp['dsr_sig'] else 'not significant'}")
    print(f"  Importance     : " + ", ".join(
        f"{n} {m:.2f}±{s:.2f}" for n, m, s in
        zip(FEATURE_NAMES, cp["imp_mean"], cp["imp_std"])))
    print(f"  Imp. stability : mean pairwise rank-corr across folds {cp['imp_rank_stability']:+.2f}  "
          f"(1.0 = identical ordering every fold)")

    # --- 2/3/4. Walk-forward --------------------------------------------- #
    print("\n" + "-" * 78)
    print("2. COST-AWARE WALK-FORWARD L/S  (economic P&L, turnover, capacity)")
    print("-" * 78)
    wf = walk_forward(P, gbrt_kw, train_window=args.train_window,
                      refit_every=args.refit_every, cost_bps_grid=args.cost_bps,
                      capital=args.capital, w_max=args.w_max, cap_frac=args.cap_frac,
                      seed=args.seed)
    gb = wf["gbrt"]
    print(f"  GBRT book      : {gb['t'].size} periods, avg 1-way turnover "
          f"{gb['turnover'].mean()*100:.1f}%/rebalance, ppy={gb['ppy']:.0f}")
    print(f"  {'cost (bps)':>10} | {'ann.ret':>8} | {'net Sharpe (ann)':>16} | "
          f"{'PSR':>5} | {'DSR':>5} | {'cum P&L':>8}")
    for c in args.cost_bps:
        pf = _perf(gb["nets"][c], gb["ppy"], args.n_trials)
        print(f"  {c:>10.2f} | {pf['ann_ret']*100:>7.1f}% | {pf['sr_ann']:>16.2f} | "
              f"{pf['psr']:>5.2f} | {pf['dsr']:>5.2f} | {pf['cum']*100:>7.0f}%")
    mid_cost = args.cost_bps[len(args.cost_bps) // 2]

    # baselines at the mid cost
    print("\n" + "-" * 78)
    print(f"3. BASELINES  (same sizing/costs, cost={mid_cost} bps)")
    print("-" * 78)
    print(f"  {'strategy':>12} | {'ann.ret':>8} | {'Sharpe(ann)':>11} | {'DSR':>5} | {'avg IC':>7}")
    for kind, label in (("gbrt", "GBRT L/S"), ("momentum", "1-factor mom"),
                        ("long", "always-long"), ("null", "shuffled null")):
        w = wf[kind]
        if not w:
            continue
        pf = _perf(w["nets"][mid_cost], w["ppy"], args.n_trials)
        ic = _nanmean(w["ic"])
        print(f"  {label:>12} | {pf['ann_ret']*100:>7.1f}% | {pf['sr_ann']:>11.2f} | "
              f"{pf['dsr']:>5.2f} | {ic:>+7.3f}")

    # regimes (GBRT, mid cost)
    print("\n" + "-" * 78)
    print(f"4. REGIME SPLITS  (GBRT L/S, cost={mid_cost} bps)")
    print("-" * 78)
    net = gb["nets"][mid_cost]
    tt = gb["t"]
    ts = P.date_ts[tt]
    pre = ts < pd.Timestamp("2020-01-01")
    calm = ~P.stressed[tt]
    for label, mask in (("pre-2020", pre), ("2020+", ~pre),
                        ("calm vol", calm), ("stressed vol", ~calm)):
        if mask.sum() < 3:
            continue
        pf = _perf(net[mask], gb["ppy"], args.n_trials)
        print(f"  {label:>14}: n={mask.sum():>3}  ann.ret {pf['ann_ret']*100:>6.1f}%  "
              f"Sharpe {pf['sr_ann']:>5.2f}  avg IC {_nanmean(gb['ic'][mask]):+.3f}")

    # decay across OOS thirds (GBRT)
    print("\n" + "-" * 78)
    print("5. PERFORMANCE DECAY  (GBRT, OOS timeline in thirds)")
    print("-" * 78)
    thirds = np.array_split(np.arange(net.size), 3)
    for k, idx in enumerate(thirds):
        seg = net[idx]
        ics = gb["ic"][idx]
        pf = _perf(seg, gb["ppy"], args.n_trials)
        yr0, yr1 = P.date_ts[tt[idx[0]]].year, P.date_ts[tt[idx[-1]]].year
        print(f"  third {k+1} ({yr0}-{yr1}): avg IC {_nanmean(ics):+.3f}  "
              f"ann.ret {pf['ann_ret']*100:>6.1f}%  Sharpe {pf['sr_ann']:>5.2f}")
    print("\n  (With --no-decay the IC is flat across thirds; with decay on it fades — the\n"
          "   signature of an alpha being arbitraged away, the McLean-Pontiff effect.)")

    print("\n" + "=" * 78)
    print("READING: the GBRT book earns a real but small gross edge that survives tiny\n"
          "costs and erodes as costs and turnover rise; the deflated Sharpe is the honest\n"
          "gate; baselines and the shuffled null bound what 'skill' means; the regime and\n"
          "decay splits show where and when the edge lives. On a NULL/real-dead panel all\n"
          "of this collapses to the up-rate and DSR~0. Swap in real data via --data.")
    print("=" * 78)

    if args.json:
        dump = dict(source=src, n_names=panel.n_names, avg_alive=avg_alive,
                    n_dates=panel.n_dates, cpcv={k: (v.tolist() if isinstance(v, np.ndarray)
                                                     else v) for k, v in cp.items()})
        with open(args.json, "w") as fh:
            json.dump(dump, fh, indent=2, default=float)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
