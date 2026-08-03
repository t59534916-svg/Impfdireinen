"""Fundamentals → the project's one evaluation contract.

A new feature family only counts here once it can be judged by the same
machinery as everything else: a no-look-ahead :class:`~vpts.ml.models.FactorDataset`
(or :class:`~vpts.ml.models.CrossSectionalPanel`) fed to purged CPCV, a
block-permutation null, the Deflated Sharpe / PBO and the survivorship-injection
sweep. This module is that adapter for fundamental data.

Three builders:

* :func:`build_fundamental_dataset` — per-name time series: fundamental ratios
  at bar *t* → the strictly-future ``horizon``-bar return.
* :func:`build_combined_dataset` — the structural microstructure features (the
  study's strongest signal) **stacked with** the fundamental ratios, to test
  whether slow accounting data adds anything to fast price/volume structure.
* :func:`build_fundamental_panel` — the frame fundamentals are actually used in:
  a date×name cross-section of *ranked* value/quality factors, market-neutral by
  construction.

Point-in-time is enforced, not assumed
--------------------------------------
Features come from :func:`fundamental_feature_frame`, which aligns snapshots
as-of their **filing date**. Every builder then re-audits the sampled rows with
:func:`~vpts.analysis.fundamentals.audit_point_in_time` and **raises** if a
single bar would have used a filing before it was public. A leak here would not
produce a subtle bias — it produces a spectacular fake edge — so it is a hard
error rather than a warning.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from vpts.analysis.fundamentals import (
    _safe_div,
    align_asof,
    audit_point_in_time,
    piotroski_f_score,
)
from vpts.analysis.models import FUNDAMENTAL_FEATURES, FundamentalSeries
from vpts.ml.models import CrossSectionalPanel, FactorDataset
# Reuse the study's existing cross-sectional ranking so a fundamental panel is
# ranked exactly like the price-based one (no second, subtly-different scheme).
from vpts.ml.cross_sectional import _rank_centered

_EPS = 1e-12
_NAN = float("nan")

#: Columns carried alongside the features for auditing/derivation.
_AUDIT_COLS = ("period_end", "available_at")


def _vdiv(a: pd.Series, b: pd.Series) -> pd.Series:
    """Vectorised safe division: NaN wherever the denominator is ~0 or missing."""
    b = b.where(b.abs() > _EPS)
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _snapshot_table(series: FundamentalSeries) -> pd.DataFrame:
    """Per-snapshot, **price-independent** ratios + the raw inputs price needs.

    Indexed by ``available_at``, so :func:`~vpts.analysis.fundamentals.align_asof`
    can broadcast it onto any price index without re-deriving the point-in-time
    logic.
    """
    rows = []
    for snap in series.snapshots:
        prior = series.prior_year_of(snap)
        ta = float(snap.total_assets)
        wc = float(snap.current_assets) - float(snap.current_liabilities)
        # Altman Z minus its market-cap term (0.6·MVE/TL), added per-bar later.
        z_base = (1.2 * _safe_div(wc, ta) + 1.4 * _safe_div(snap.retained_earnings, ta)
                  + 3.3 * _safe_div(snap.operating_income, ta) + 1.0 * _safe_div(snap.revenue, ta))
        rows.append({
            "available_at": snap.available_at,
            "period_end": snap.period_end,
            # price-independent features
            "gross_margin": _safe_div(snap.gross_profit, snap.revenue),
            "operating_margin": _safe_div(snap.operating_income, snap.revenue),
            "roe": _safe_div(snap.net_income, snap.total_equity),
            "roa": _safe_div(snap.net_income, ta),
            "debt_to_equity": _safe_div(snap.total_debt, snap.total_equity),
            "current_ratio": _safe_div(snap.current_assets, snap.current_liabilities),
            "interest_coverage": _safe_div(snap.operating_income, abs(float(snap.interest_expense))),
            "accruals": _safe_div(float(snap.net_income) - float(snap.operating_cash_flow), ta),
            "asset_growth": _NAN if prior is None else _safe_div(
                ta - float(prior.total_assets), abs(float(prior.total_assets))),
            "revenue_growth": _NAN if prior is None else _safe_div(
                float(snap.revenue) - float(prior.revenue), abs(float(prior.revenue))),
            "earnings_growth": _NAN if prior is None else _safe_div(
                float(snap.net_income) - float(prior.net_income), abs(float(prior.net_income))),
            "piotroski_f": piotroski_f_score(snap, prior),
            # inputs for the price-dependent features
            "_net_income": float(snap.net_income),
            "_total_equity": float(snap.total_equity),
            "_revenue": float(snap.revenue),
            "_fcf": snap.free_cash_flow,
            "_total_liabilities": float(snap.total_liabilities),
            "_shares": float(snap.shares_diluted),
            "_z_base": z_base,
        })
    if not rows:
        return pd.DataFrame(columns=["period_end", "_shares"])
    return pd.DataFrame(rows).set_index("available_at").sort_index()


def fundamental_feature_frame(
    frame: pd.DataFrame,
    series: FundamentalSeries,
    *,
    audit: bool = True,
) -> pd.DataFrame:
    """Per-bar :data:`~vpts.analysis.models.FUNDAMENTAL_FEATURES`, point-in-time aligned.

    Accounting ratios come from the latest filing public at bar *t*; valuation
    ratios pair those accounting numbers with bar *t*'s **own** close (that is
    the correct pairing — the market re-prices continuously, the balance sheet
    does not). Rows before the first filing are ``NaN``.

    Raises ``ValueError`` when *audit* is on and the alignment used any filing
    before its publication date.
    """
    if "Close" not in frame.columns:
        raise ValueError("fundamental_feature_frame needs a 'Close' column.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("fundamental_feature_frame needs a DatetimeIndex (filing dates).")

    aligned = align_asof(_snapshot_table(series), frame.index)
    if audit and "available_at" in aligned.columns:
        report = audit_point_in_time(aligned)
        if report["violations"]:
            raise ValueError(
                f"point-in-time violation: {report['violations']} bar(s) would use a filing "
                "before it was public. Refusing to build a leaked dataset."
            )

    close = frame["Close"].astype(float)
    shares = aligned.get("_shares", pd.Series(np.nan, index=frame.index))
    mcap = close * shares

    out = pd.DataFrame(index=frame.index)
    out["earnings_yield"] = _vdiv(aligned["_net_income"], mcap)
    out["book_to_price"] = _vdiv(aligned["_total_equity"], mcap)
    out["sales_to_price"] = _vdiv(aligned["_revenue"], mcap)
    out["fcf_yield"] = _vdiv(aligned["_fcf"], mcap)
    for col in ("gross_margin", "operating_margin", "roe", "roa", "debt_to_equity",
                "current_ratio", "interest_coverage", "accruals", "asset_growth",
                "revenue_growth", "earnings_growth", "piotroski_f"):
        out[col] = aligned[col].astype(float)
    out["altman_z"] = aligned["_z_base"] + 0.6 * _vdiv(mcap, aligned["_total_liabilities"])

    out = out[list(FUNDAMENTAL_FEATURES)]
    for c in _AUDIT_COLS:
        out[c] = aligned[c]
    return out


# --------------------------------------------------------------------------- #
# Per-name time series → FactorDataset
# --------------------------------------------------------------------------- #
def _filing_blocks(period_end: pd.Series, valid: np.ndarray) -> list[np.ndarray]:
    """Group valid row positions by the filing they descend from."""
    pe = period_end.to_numpy()
    blocks: dict = {}
    for i in np.where(valid)[0]:
        blocks.setdefault(pe[i], []).append(i)
    return [np.array(v, dtype=int) for _, v in sorted(blocks.items(), key=lambda kv: kv[1][0])]


def build_fundamental_dataset(
    frame: pd.DataFrame,
    series: FundamentalSeries,
    *,
    horizon: int = 20,
    rows_per_filing: int = 1,
    features: Sequence[str] = FUNDAMENTAL_FEATURES,
    symbol: Optional[str] = None,
    min_samples: int = 30,
) -> FactorDataset:
    """Fundamental ratios at bar *t* → the strictly-future ``horizon``-bar return.

    The single-feature ``baseline`` is the **earnings yield** — the canonical
    value factor — so the ridge combination is measured against the obvious
    one-factor alternative rather than against nothing.

    Sampling: one row per filing, by default
    -----------------------------------------
    This is the load-bearing decision, and it costs a lot of apparent sample
    size on purpose. Accounting features only change when a company files;
    between filings they are constant (the valuation ratios drift only through
    price). Sampling them daily produces thousands of rows that are *one*
    observation wearing many hats, and it breaks the significance test in a way
    that is easy to miss: :func:`~vpts.stats.recommend_block_size` sizes the
    permutation block from the **label** horizon, so with ``horizon=20,
    stride=10`` it keeps blocks of 3 samples together — while the *feature* is
    unchanged across ~25 consecutive samples. The null is then far too tight and
    over-rejects. Measured on synthetic data with fundamentals generated
    independently of price, daily sampling reported ``IC = +0.16, p = 0.005``
    — a confident verdict on pure noise.

    Emitting ``rows_per_filing`` rows per filing (default 1, placed at the first
    usable bar) makes consecutive rows genuinely far apart, so the derived
    ``stride`` — the median bar gap between emitted rows — feeds an honest block
    size and purge. Raise it only if you also raise the block size by hand and
    can say why.

    The honest consequence: an annual filer over ten years contributes ~10 rows.
    That is the real sample size of annual fundamental data, and it is why a
    single name can never settle a fundamental question here.
    """
    feats = fundamental_feature_frame(frame, series)
    close = frame["Close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0                    # strictly FUTURE label

    cols = list(features)
    valid = feats[cols].notna().all(axis=1).to_numpy() & fwd.notna().to_numpy()
    n_per = max(1, int(rows_per_filing))
    picks: list[int] = []
    for block in _filing_blocks(feats["period_end"], valid):
        if block.size <= n_per:
            picks.extend(block.tolist())
        else:                                    # evenly spread within the filing window
            sel = np.linspace(0, block.size - 1, n_per).round().astype(int)
            picks.extend(block[np.unique(sel)].tolist())
    idx = np.array(sorted(picks), dtype=int)
    if idx.size < min_samples:
        raise ValueError(
            f"{symbol or series.symbol}: only {idx.size} usable fundamental samples "
            f"(need >= {min_samples}). Fundamentals are sampled once per filing, so "
            "this needs many years of history — or many names pooled."
        )

    gaps = np.diff(idx)
    stride = int(np.median(gaps)) if gaps.size else 1
    X = feats[cols].to_numpy(float)[idx]
    y = fwd.to_numpy(float)[idx]
    base_col = "earnings_yield" if "earnings_yield" in cols else cols[0]
    baseline = feats[base_col].to_numpy(float)[idx]
    return FactorDataset(
        X=X, y=y, baseline=baseline, feature_names=tuple(cols),
        horizon=horizon, stride=max(1, stride),
        timestamps=frame.index[idx], symbol=symbol or series.symbol,
    )


def build_combined_dataset(
    frame: pd.DataFrame,
    series: FundamentalSeries,
    *,
    horizon: int = 20,
    stride: int = 5,
    lookback: int = 120,
    symbol: Optional[str] = None,
    fundamental_features: Sequence[str] = FUNDAMENTAL_FEATURES,
    **structural_kwargs,
) -> FactorDataset:
    """Structural microstructure features **plus** fundamental ratios, same bars.

    The point is a direct comparison, not a bigger model: does slow accounting
    data add anything on top of the fast price/volume structure that
    ``RESEARCH.md`` identifies as the study's strongest (if survivorship-fragile)
    signal? Structural rows come from
    :func:`~vpts.structure.dataset.build_structural_dataset`; the fundamental
    block is aligned onto those exact timestamps, and bars where either block is
    incomplete are dropped.

    Two honest caveats, both about how the result may be read:

    * Every added feature is another degree of freedom, so ``n_trials`` for the
      Deflated Sharpe must go **up** when this dataset is compared against its
      two halves.
    * Sampling follows the *structural* cadence (every ``stride`` bars), because
      that is what the structural features need. The fundamental block is
      therefore piecewise-constant across consecutive rows, and the permutation
      block size derived from the label horizon understates that dependence —
      exactly the over-rejection :func:`build_fundamental_dataset` avoids by
      sampling once per filing. **Do not read this dataset's p-value as clean
      evidence for the fundamental block.** The question it can answer is a
      nested comparison: OOS IC here versus structural-only on the same bars.
    """
    from vpts.structure.dataset import build_structural_dataset

    struct = build_structural_dataset(
        frame, lookback=lookback, horizon=horizon, stride=stride,
        symbol=symbol, **structural_kwargs)
    if struct.timestamps is None:
        raise ValueError("combined dataset needs a DatetimeIndex to align fundamentals.")

    feats = fundamental_feature_frame(frame, series)
    cols = list(fundamental_features)
    fund = feats.reindex(struct.timestamps)[cols].to_numpy(float)

    keep = np.all(np.isfinite(fund), axis=1) & np.all(np.isfinite(struct.X), axis=1)
    if int(keep.sum()) < 30:
        raise ValueError(
            f"{symbol or series.symbol}: only {int(keep.sum())} bars have both structural "
            "and fundamental features."
        )
    return FactorDataset(
        X=np.hstack([struct.X[keep], fund[keep]]),
        y=struct.y[keep],
        baseline=struct.baseline[keep],
        feature_names=tuple(struct.feature_names) + tuple(cols),
        horizon=horizon,
        stride=max(1, stride),
        timestamps=pd.DatetimeIndex(struct.timestamps)[keep],
        symbol=symbol or series.symbol,
    )


# --------------------------------------------------------------------------- #
# Cross-section → CrossSectionalPanel
# --------------------------------------------------------------------------- #
def build_fundamental_panel(
    frames: Mapping[str, pd.DataFrame],
    series_map: Mapping[str, FundamentalSeries],
    *,
    horizon: int = 20,
    rebalance: int = 20,
    features: Sequence[str] = FUNDAMENTAL_FEATURES,
    min_names: int = 5,
) -> CrossSectionalPanel:
    """Date×name panel of cross-sectionally **ranked** fundamental factors.

    This is how value and quality factors are actually deployed: not "is this
    company cheap in absolute terms" but "is it cheaper than its peers today".
    Ranking is contemporaneous across names and centred to ``[-0.5, 0.5]``, so
    the bet is market-neutral and scale-free, and the level of a ratio (which
    drifts with the whole market's valuation) cannot masquerade as signal.

    ``rebalance`` defaults to 20 bars rather than the price-panel's 5: sampling
    slow-moving accounting data every week mostly manufactures duplicate rows.
    """
    syms = tuple(sorted(s for s in frames if s in series_map))
    if len(syms) < min_names:
        raise ValueError(f"need >= {min_names} names with fundamentals; got {len(syms)}.")

    cols = list(features)
    per_name: dict[str, pd.DataFrame] = {}
    for s in syms:
        f = frames[s]
        ff = fundamental_feature_frame(f, series_map[s])[cols]
        close = f["Close"].astype(float)
        ff["fwd"] = close.shift(-horizon) / close - 1.0
        per_name[s] = ff

    all_dates = per_name[syms[0]].index
    for s in syms[1:]:
        all_dates = all_dates.union(per_name[s].index)
    sampled = pd.DatetimeIndex(all_dates)[::max(1, rebalance)]

    wide = {c: pd.concat({s: per_name[s][c] for s in syms}, axis=1)[list(syms)]
            for c in (*cols, "fwd")}
    fmat = np.column_stack([wide[c].reindex(sampled).to_numpy(float) for c in cols])
    fmat = fmat.reshape(len(sampled), len(cols), len(syms))
    ymat = wide["fwd"].reindex(sampled).to_numpy(float)

    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    d_rows: list[int] = []
    kept: list = []
    for di in range(len(sampled)):
        feat_d, y_d = fmat[di], ymat[di]
        valid = np.isfinite(y_d) & np.all(np.isfinite(feat_d), axis=0)
        n_valid = int(valid.sum())
        if n_valid < min_names:
            continue
        ranked = np.column_stack([_rank_centered(feat_d[j, valid]) for j in range(len(cols))])
        date_id = len(kept)
        yv = y_d[valid]
        for r in range(n_valid):
            X_rows.append([float(v) for v in ranked[r]])
            y_rows.append(float(yv[r]))
            d_rows.append(date_id)
        kept.append(sampled[di])

    if not kept:
        raise ValueError("no fundamental cross-section met the min_names requirement.")
    return CrossSectionalPanel(
        X=np.array(X_rows, dtype=float).reshape(-1, len(cols)),
        y=np.array(y_rows, dtype=float),
        date_id=np.array(d_rows, dtype=int),
        feature_names=tuple(cols),
        horizon=horizon,
        rebalance=max(1, rebalance),
        n_dates=len(kept),
        symbols=syms,
        dates=pd.DatetimeIndex(kept),
    )
