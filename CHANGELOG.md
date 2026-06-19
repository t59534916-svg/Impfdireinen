# Changelog

All notable changes to `vpts`, by version. The project grew in two acts — **a product** (Phases 1–6, `v0.1`→`v1.0`) and then **its adversarial validation** (`v1.1`→`v1.7`). Format loosely follows [Keep a Changelog](https://keepachangelog.com); research findings are noted where a version produced one.

The canonical research narrative is [`RESEARCH.md`](RESEARCH.md); experiment numbers below refer to it.

---

## Act IV — data-first

### `2.1.0` — Classic indicators on trial + a harness small-sample calibration guard
- **Added** `build_indicator_dataset` (`vpts.features`) — the newsletter-staple swing toolkit (**RSI, MACD, a moving-average crossover, momentum, a Fibonacci-retracement position**) as a no-look-ahead `FactorDataset`, plus `examples/indicator_swing_eval.py` to run it through the same harness (CPCV OOS IC · block-permutation p · DSR · PBO · survivorship sweep). On synthetic survivors it clears nothing (**NO EDGE**, DSR ≈ 0.1) — the same class of input experiments 2–6 already showed doesn't beat the wall. No-look-ahead unit-tested.
- **Found & guarded (honesty):** validating the above surfaced that `honest_backtest`'s block-permutation p **over-rejects true i.i.d. nulls in a small-sample regime** — ≈45% false-positive at ~48 samples/dataset, because the small-sample OOS IC is biased/unstable; it is well-calibrated at ~170+ samples/dataset (false-positive ≈ 5%). Added `MIN_SAMPLES_FOR_PERM` (120): below it, a "significant" result is flagged as **unreliable** with a warning. **This does *not* affect the eleven published experiments** — those use the standalone, review-verified `block_permutation_test` on the full 88-name / ~1,300-bar sample, not this convenience path.

### `2.0.0` — The delisted-inclusive ingestion backbone
The study's binding constraint is survivorship-free data, **not** the model. v2.0 builds the production path to feed the harness real delisted names the instant a source exists — and re-confirms the wall with a fresh live probe.
- **Added** `FMPSource` (`vpts.data`) — a real `DataSource` over Financial Modeling Prep's stable EOD API. Survivor history on the free tier; **delisted on a paid tier**, opt-in via `FMPSource(delisted_capable=True)` (or `VPTS_FMP_DELISTED=1`). Key from `FMP_API_KEY`, injectable HTTP, surfaces FMP's plan-gate message, added to `default_registry()` when the key is set. Offline-tested.
- **Added** `materialize_lake()` + `LakeBuildReport` (`vpts.data.lake`) — pull a universe (survivors + `KNOWN_DELISTED`) from **any** `DataSource` and write the Hive-partitioned parquet lake `DataLakeSource` reads, capping each name at its delist date (pre-delisting history only) and reporting death-leg coverage. Source-agnostic; offline round-trip tested through `DataLakeSource`.
- **Added** `CsvSource` + `read_ohlcv_csv` (`vpts.data`) — ingest OHLCV from **CSV files you have the rights to** (broker/vendor/logged-in export), handling real-world mess: `,` or `;` delimiters, decimal-comma + thousands-dot (German, e.g. ariva), `DD.MM.YY` dates, and English/German column synonyms (or an explicit `column_map`). Composes with `materialize_lake` → `DataLakeSource`; `examples/v2_survivorship_free.py --csv-dir DIR` runs the survivorship-free comparison straight from a CSV folder. (Note: keep one venue/currency — a German EUR listing is not the US USD series.)
- **Added** `examples/v2_survivorship_free.py` — one command: materialize/read a lake → structural `honest_backtest` **survivors-only vs survivors+delisted**, printing the IC / long-short-net delta and whether the signal inverts. Runs offline on a synthetic lake today (reproduces the mirage: L/S +0.21% → −0.46%/bet); becomes the real survivorship-free verdict the instant `--lake`/`--source` points at real data.
- **Probe (fresh evidence):** FMP serves survivors (AAPL 2017, MSFT 2023 ✓) but **paywalls delisted** (LEH, SIVB → "requires a higher plan") and the delisted screener. The wall stands — real delisted OHLCV needs a paid Polygon/FMP plan or a local lake. The backbone is built, tested, and ready for it; **the headline conclusion is unchanged.**

---

## Act III — production hardening

### `1.16.0` — Rotating proxies, a free delisted feed, the one-call harness, and the external-review hardening
A large branch: the Tier 0→3 response to an external quant review, two new free-data capabilities, the harness packaged as a product, and a full adversarial code-review pass. **No finding overturns the headline** (*no survivorship-robust tradeable edge; the binding constraint is data, not the model*) — it sharpens the harness and its honesty.
- **Added** `ProxyPool` (`vpts.data.proxy`): rotating proxy pool for the rate-limited free feeds (Yahoo v8 audit + yfinance) — round-robin rotation, per-proxy exponential-backoff cooldown on 403/429, request jitter, User-Agent rotation. Credentials load from `$VPTS_PROXIES` / `$VPTS_PROXY_FILE` / a git-ignored `proxies.txt` and are never committed (`proxies.example.txt` template). Injectable transport → fully offline-tested; no config ⇒ direct fetch (backward compatible). *Caveat: a proxy changes the IP only — it does not defeat a JS proof-of-work wall like Stooq's live CSV.*
- **Added** `StooqSource` (`vpts.data`): the first **free** `DataSource` that retains delisted US names (`provides_delisted=True`). Live single-symbol CSV (JS-walled — detected and refused, never parsed as data) plus an offline bulk-export reader with delimiter sniffing.
- **Added** `vpts.harness.honest_backtest()`: the one-call skeptic's checklist — pooled OOS-IC, block-permutation p, conviction-bucket curve, Deflated Sharpe, PBO, optional survivorship-injection sweep, and a one-line verdict. Plus `vpts.insight.evidence_from_report` / `explain_report` to narrate a `HonestReport` through the LLM layer in one step.
- **Added** the Tier 0→3 external-review response: a **block-aware null** on the structural headline; a **rally-mode** synthetic decliner (the inversion is not a monotone-decline artifact); **DSR/PBO** on the +0.26%/bet survivor book (DSR 0.884 < 0.95); reversal/momentum **orthogonalization** (not generic reversal) and an **out-of-regime** crypto run; and a coverage audit (`audit_coverage` / `KNOWN_DELISTED`) quantifying the free-data ceiling (≈0% bankruptcy coverage).
- **Changed/Fixed (code review):** the harness verdict now **gates on PBO** (an overfit selection no longer "PASSES") and bases **inversion on the traded long/short net**, not the noisiest bucket mean; the **DSR** runs on an overlap-deflated effective sample with an honest `n_trials` contract (warns + flags when unadjusted); **`validate_ohlcv`** drops structurally-invalid bars (NaN/non-positive prices, `High < Low`); the proxy **cooldown applies on the yfinance retry path**; the overclaim guardrail is described honestly (best-effort, not a structural bar) with widened patterns; **`DataLakeSource.is_delisted`** no longer depends on `build_universe()` call order; **`PolygonSource.list_delisted`** follows `next_url` pagination (no silent truncation); **`StooqSource`** wall-detection covers Cloudflare-style challenges; **`haircut_sharpe`** rejects sub-unit `annualization`.
- **Docs:** runtime-pipeline + AI/insight-layer architecture diagrams (`docs/ARCHITECTURE.md`); a top-level `CLAUDE.md`; a README proxy section.

### `1.15.0` — Parquet data-lake source (survivor + delisted)
- **Added** `DataLakeSource` (`vpts.data`): reads a Hive-partitioned parquet lake (`{root}/{TICKER}/year=YYYY/data.parquet`) into the `DataSource` interface — the layout of a real `data_lake/eod/{source}/daily/` tree. Enumerates the universe (`available_symbols`) and builds a **point-in-time `Universe` with inferred delist dates** (`build_universe` — a name whose last bar predates the lake's global last date by `active_gap_days` is marked delisted), so a **survivorship-free** backtest runs straight from the lake. Optional `parquet` extra (`pip install vpts[parquet]`). 5 tests on a synthetic parquet lake (incl. delist inference), skip cleanly where pyarrow is absent.
- **Why:** this is the ingester for real survivorship-free data (delisted + micro-cap) — the one thing the free feeds couldn't provide. The harness is finally pointed at the actual binding constraint.

### `1.14.1` — Fix: LLM client failures degrade instead of crashing
- **Fixed** `AnthropicClient.complete`: the `anthropic.Anthropic()` construction was outside the try/except, so an auth/credential-resolution failure raised a raw `TypeError` that the `InsightGenerator` template fallback did not catch — the layer would crash instead of degrading. Now every backend failure (missing package, unresolved auth, API error) surfaces as `InsightLLMError` and the deterministic template takes over. Surfaced by actually attempting a live call (which cannot authenticate in this environment — the live Claude path remains unverified; only the mock/template paths are tested).

### `1.14.0` — Unsupervised regime detection / pattern discovery (`vpts.ml.regime`)
- **Added** the one ML gap the project was missing: unsupervised **pattern discovery** (`RegimeClusterer` — transparent k-means on standardized features, chosen over GMM/HMM for explainability), **walk-forward regime assignment** (`walk_forward_regimes` — fit on prior bars only, refit periodically, labels aligned across refits; no look-ahead, unit-tested via future-scramble invariance), and an honest **permutation-tested evaluator** (`regime_forward_stats` — best-minus-worst regime forward-return spread vs a label-shuffle null).
- **`examples/regime_discovery.py`** — walk-forward regimes on the real stocknet survivors. **Finding: 3/20 names clear p<0.05 vs ~1 expected by chance (P(≥3)≈0.075) → ≈ chance; the discovered regimes do NOT predict forward returns out of sample.** As predicted and consistent with the rest of the arc: a real new capability, no robust edge — judged by the harness, negative reported as negative.

### `1.13.0` — Base-rate-calibrated (loser-heavy) survivorship injection
- **`SurvivorshipInjector`** gains `delisted_fraction` (target share of the *augmented* universe that is delisted — can exceed the survivors, making the population **loser-heavy** to match the empirical reality that most stocks underperform/delist over their lifetime, Bessembinder 2018) and `terminal_frac` (calibrated death severity). `synthetic_delisted_ohlcv` gains `terminal_frac` (deterministic drift to ≈ `frac×start`, a slow decline — delistings take months, not a week).
- **`examples/survivorship_baserate.py`** — sweeps the loser:winner ratio on the *real* free stocknet survivors, **calibrated to the empirical record** (`--preset bessembinder`, the default): `terminal_frac=0.08` (Bessembinder 2018's −91.95% median delisted lifetime return) and a base-rate ladder 0 / 10% / 35% (delisted share) / 57% (lifetime under-T-bill rate); delisting-return severity cross-checked against Shumway 1997/1999. **Three-lens finding** at the 57% lifetime-loser rate: the linear **IC barely moves** (+0.082→+0.055, stays significant — *blind* to the effect); the **directional conviction edge INVERTS** (top-bucket fwd return +1.49%→−0.84%, whole curve goes negative — a survivorship mirage); the **market-neutral selectivity is resilient** (+1.05%→+1.69%, doesn't flip). Reproduces `RESEARCH.md`'s direction-vs-selectivity nuance as a function of the base rate, on real survivors (dead names synthetic — the free-data wall, verified across Polygon/FMP).

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
