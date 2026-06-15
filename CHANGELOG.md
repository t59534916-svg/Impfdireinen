# Changelog

All notable changes to `vpts`, by version. The project grew in two acts — **a product** (Phases 1–6, `v0.1`→`v1.0`) and then **its adversarial validation** (`v1.1`→`v1.7`). Format loosely follows [Keep a Changelog](https://keepachangelog.com); research findings are noted where a version produced one.

The canonical research narrative is [`RESEARCH.md`](RESEARCH.md); experiment numbers below refer to it.

---

## Act III — production hardening

### `1.13.0` — Base-rate-calibrated (loser-heavy) survivorship injection
- **`SurvivorshipInjector`** gains `delisted_fraction` (target share of the *augmented* universe that is delisted — can exceed the survivors, making the population **loser-heavy** to match the empirical reality that most stocks underperform/delist over their lifetime, Bessembinder 2018) and `terminal_frac` (calibrated death severity). `synthetic_delisted_ohlcv` gains `terminal_frac` (deterministic drift to ≈ `frac×start`, a slow decline — delistings take months, not a week).
- **`examples/survivorship_baserate.py`** — sweeps the loser:winner ratio on the *real* free stocknet survivors. Two views: (1) pooled OOS IC erodes from **+0.082 (p=0.024, significant)** to **+0.031 (p=0.098)** as the population goes loser-heavy; (2) the **cost-aware conviction-bucket** view surfaces the **directional sign inversion** — the most-bullish-flagged bars go from **+1.49% (survivors) → −1.51% (loser-heavy)** while the market-neutral tails L/S only erodes (+1.05% → +0.38%). Faithfully reproduces `RESEARCH.md`'s nuance (direction = survivorship mirage; selectivity = resilient thread) as a function of the base rate. Dead names synthetic (no free delisted prices); survivors real.

### `1.12.1` — Review fixes (correctness double-check)
A 7-angle adversarial code review of the Act III diff (math verified against references) surfaced edge-case fixes, all now tested:
- **`vpts.insight`** — `VALIDATED` (which licenses edge-claims) now requires the deflated-Sharpe bar to have been **tested and passed**; a missing DSR downgrades to `weak_unvalidated`. Closes a hole where edge-language could be licensed with no selection control.
- **`vpts.stats`** — the multiple-testing haircut now **preserves the sign** of a significant *negative* Sharpe (was zeroed out); `block_permutation_test` **raises** instead of returning a degenerate `p≈1` when `block_size` leaves < 2 blocks (and precomputes the partition); `min_track_record_length` returns `inf` for inconsistent moments; PBO no longer recomputes the IS-best statistic in a second pass.
- **`vpts.data`** — `PolygonSource` returns a **tz-naive** index (matching the other sources, so `Universe.members_asof` doesn't crash on tz mismatch) and handles `period="ytd"` correctly; `SyntheticSource` honors a **partial** `start`/`end`; `SurvivorshipInjector` uses the **true median** survivor length.
- **`vpts.altdata`** — `StaticAltSource` no longer forward-fills a stale value **past the data span** (NaN outside it, per the contract).
- Demo now uses the **block-permutation** null (honest regardless of `--stride`); behavioral-frame warm-up caveat documented.
- **`PolygonSource` verified live** against the real API: recent OHLCV parses correctly and returns a tz-naive index; `list_delisted()` enumerates real delisted names. `get_bars` now **surfaces Polygon's own error** (e.g. `NOT_AUTHORIZED` → "upgrade your plan" for delisted/old data) instead of a generic "no bars"; `is_delisted` documents that a free single-ticker lookup returns `NOT_FOUND` for delisted symbols (use `list_delisted`). Confirms the documented wall: the **delisted/point-in-time timeframe requires a paid plan** — the adapter is ready for it.

### `1.12.0` — Delisted-capable data source, an honest null, and alt-data hooks
- **Added** `PolygonSource` (`vpts.data`): a `DataSource` over Polygon.io that serves **delisted** history and point-in-time reference status (`is_delisted`, `list_delisted`) — the survivorship escape hatch. Activates on `POLYGON_API_KEY` and is placed first in `default_registry()` when present; the HTTP layer is injectable, so parsing is unit-tested with no key/network.
- **Added** the **block-permutation** test (`vpts.stats`): `block_permutation_test` / `block_shuffle_indices` / `recommend_block_size`. The per-row label shuffle is anti-conservative for **overlapping** labels (it destroys autocorrelation); the block null preserves it and gives an honest p-value — demonstrated by a spurious-regression test where per-row falsely rejects and the block null does not.
- **Added** `vpts.altdata`: integration points for **options-flow** (dealer gamma/skew positioning) and **sentiment** (news/social/retail) signals — `AltSignalSource` ABC, `NullAltSource`, `StaticAltSource`, and a causal `merge_alt_features`. Interfaces only (no live feed); alt signals plug into the same harness as hypotheses.

### `1.11.0` — LLM insight layer that cannot fabricate an edge (`vpts.insight`)
- **Added** `vpts.insight`: turns validated harness statistics into a human-readable behavioral-finance explanation via Claude (`AnthropicClient`, default `claude-opus-4-8`) — with a structural honesty guarantee rather than a trust-the-model hope. The **verdict** (no-edge / survivorship-fragile / overfit / weak / validated) is computed *in code* (`assess`) from the same bars as `RESEARCH.md`; the LLM only narrates it; the output is **scanned** for edge-claims the verdict forbids (`scan_for_overclaims`) and corrected if it overclaims; with no client (or on failure) a faithful, non-overclaiming **template** is rendered, so the layer is fully offline-capable.
- **Added** the optional `llm` extra (`pip install vpts[llm]`). 16 tests, all offline (incl. an adversarial mock model that *tries* to claim a tradeable edge and is caught + corrected).

### `1.10.0` — Multi-timeframe behavioral-dynamics features (`vpts.features`)
- **Added** `vpts.features`: causal behavioral proxies — participation surge (RVOL), absorption (effort-without-result), liquidity grabs (stop-runs), accumulation/distribution pressure and its acceleration (conviction shift), FOMO extension/thrust, wick rejection asymmetry, and coil-to-expansion trend emergence — each computed at multiple horizons. `build_behavioral_dataset` emits a `FactorDataset` straight into the CPCV + permutation + survivorship harness; `multi_timeframe_feature` adds look-ahead-safe calendar resampling.
- **Honest scope:** these are *hypotheses*, not validated edges — `RESEARCH.md` already found single-timeframe richness didn't beat the wall. They are wired to be judged by the harness, with negatives reported. The load-bearing test is **truncation invariance** (a feature at bar *t* is identical with or without future bars), proving no look-ahead for the whole set at once.

### `1.9.0` — Provider-agnostic data layer + point-in-time universe (`vpts.data`)
- **Added** a `DataSource` abstraction with honest `DataSourceCapabilities` (notably `provides_delisted`), a `YFinanceSource` (free, survivor-only) and an offline, deterministic `SyntheticSource` that *can* mint delisted paths, and a capability-aware `SourceRegistry` with priority fallback (`default_registry()`).
- **Added** `Universe` — point-in-time membership with **delist dates** (`members_asof`, `survivors`/`delisted`, `survivorship_free`) — and `SurvivorshipInjector`, which promotes the `RESEARCH.md` synthetic-delisted generator into the library so any experiment can be re-run "with injection" and an augmented universe in one call.
- **Why:** the project's documented binding constraint is survivorship-free / point-in-time data. The pipeline was welded to one survivor-only feed; this is the abstraction that lets a delisted-capable source (or the injector) drop in and the rest of the system ask "who was tradable *as of* date *t*?". Fully offline-testable (13 tests, incl. end-to-end injection through the structural harness).

### `1.8.0` — Anti-overfitting statistics suite (`vpts.stats`)
- **Added** `vpts.stats`: selection-aware significance tests that sit on top of the CPCV + permutation harness — **Probabilistic** and **Deflated Sharpe Ratio** (Bailey & López de Prado), **minimum track-record length**, **Probability of Backtest Overfitting** via Combinatorial Symmetric CV (Bailey–Borwein–López de Prado–Zhu), the **Harvey–Liu–Zhu** multiple-testing Sharpe haircut (Bonferroni/Holm/BH/BY adjusted p-values), and **Lo's** autocorrelation-corrected annualized Sharpe.
- **Why:** the harness tells you whether *one* signal clears its shuffled null; these tell you whether a *selected-best* strategy survives the multiplicity and short-sample corrections — the controls that turn the repo's "in-sample 0.94 / OOS 0.49" XGBoost trap into a measured PBO. Prerequisite for an honest edge-hunt.
- **Fixed** tail-cancellation in the haircut path (`1 − cdf` → survival function), so large t-stats no longer explode to an infinite haircut.

## Act II — the validation

### `1.7.0` — Structural microstructure analytics *(the strongest, and most instructive, signal)*
- **Added** `vpts.structure`: synthetic delta (CLV×volume), profile skew/kurtosis, P/b/B/D shape, ledges, poor highs, value‑area‑compression z‑score, time‑decayed cost‑basis migration → `FactorDataset` **and** `MetaDataset` (MFE/MAE triple‑barrier).
- **Added** `cpcv_factor_quantile_returns` (long/short/**flat** conviction buckets) and a relative `select_top` mode for the meta‑eval (act on the best‑rated fraction).
- **Added** a swing setup‑rater and the survivorship‑injection / decomposition / selectivity stress harnesses.
- **Findings (experiments 6–11):** a real OOS correlation (IC +0.035, p=0.005) that survives universe‑widening — but decomposition shows it is a **survivorship mirage** (the conviction edge *inverts* off survivors, +0.26 → −1.07%/bet), carried by dip‑buying features. The most resilient thread, meta‑labeling **selectivity**, is robust on survivors (9/9 params, p=0.023) yet not significant once delisted names are injected (p=0.106). XGBoost on an MFE/MAE target overfits to a sub‑0.5 OOS AUC. **No survivorship‑robust edge.**

### `1.6.0` — Cross‑sectional rank factors
- **Added** `build_cross_sectional_panel` + `cross_sectional_ic_eval` — the standard equity‑alpha construction, scored **within‑date** with a within‑date permutation null.
- **Finding (experiment 5):** a cross‑sectional near‑miss that vanished when properly powered — no edge.

### `1.5.0` — Enriched features + factor permutation test
- **Added** `build_enriched_factor_dataset` (richer per‑name inputs beyond the four coarse confluence factors) and `permutation_test_factor`.
- **Finding (experiment 4):** richer inputs, still OOS IC ≈ 0 — no edge.

### `1.4.0` — Cost‑aware meta‑eval + permutation significance
- **Added** per‑trade cost to `cpcv_meta_eval` and `permutation_test_meta` (label‑shuffle null for AUC and return‑lift), plus the survivorship stress harness for meta‑labeling.
- **Finding (experiment 3):** the meta‑labeling “edge” was **survivorship** — significant on survivors (p≈0.005), gone under injection (p≈0.80).

### `1.3.0` — Triple‑barrier meta‑labeling
- **Added** `triple_barrier_labels` (first‑touch profit/stop/vertical = MFE/MAE outcome), `build_meta_dataset`, `LogisticMetaModel`, `cpcv_meta_eval`.

### `1.2.0` — Learned factor weights
- **Added** `vpts.ml`: `RidgeFactorModel` + `cpcv_factor_eval` — the first *fitted* model, scored as a distribution of OOS IC across CPCV paths, with the hand‑weighted baseline for comparison.
- **Finding (experiment 2):** learned weights on the confluence factors → OOS IC ≈ 0.

### `1.1.0` — Validation harness *(the turning point)*
- **Added** `vpts.validation`: `CombinatorialPurgedCV` with purging + embargo, and immutable split results.
- **Fixed** 5 correctness bugs surfaced by a max‑effort review (regression‑tested).

---

## Act I — the product

### `1.0.0` — Phases 1–6 complete *(the trading system)*
- **Added** `vpts.backtest`: walk‑forward, no‑look‑ahead engine with realistic free costs (slippage + spread + commission), fixed‑fractional sizing, equity curve + blotter + stats. *A truth‑teller, not a money‑printer.*

### `0.5.0` — Phase 5 · Dashboard
- **Added** `vpts.dashboard`: pure Plotly figure builders (unit‑tested headless) + a thin Streamlit app (deep‑dive + watchlist scanner); deployable free on Streamlit Community Cloud.

### `0.4.0` — Phase 4 · Signals
- **Added** `vpts.signals`: `SignalGenerator` with reversion/breakout styles, structure‑based entry/stop/targets, minimum‑R:R gating, fixed‑fractional sizing, and journal‑ready `explain()`.

### `0.3.0` — Phase 3 · Confluence scoring
- **Added** `vpts.scoring`: `ConfluenceScorer` → `setup_quality` (0–100) + signed `bias_score`, from four transparent weighted components.

### `0.2.0` — Phase 2 · Regime
- **Added** `vpts.regime`: `QuietPhaseDetector` (percentile‑ranked vol/volume/compression) + `VolumePatternDetector` (dry‑up, accumulation, divergence, climax), on dependency‑free indicators.

### `0.1.0` — Phase 1 · Volume Profile
- **Added** `vpts.profile`: `VolumeProfileCalculator` (POC, VAH/VAL, HVN/LVN) with volume‑conserving intra‑bar distribution and volatility‑aware auto‑binning; `vpts.data` robust fetcher.

---

*Versions are tracked in `vpts.__version__`. Each minor bump corresponds to one snap‑in module or one validation milestone.*
