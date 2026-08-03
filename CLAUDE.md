# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`vpts` ("Quiet-Volume") is two things in one repo: a **rule-based Volume-Profile trading product** (Act I, Phases 1–6) and an **adversarial validation/research stack** (Acts II–III) that puts every signal on trial. The honest headline of the study (`RESEARCH.md`) is that **no input produced a survivorship-robust, tradeable edge** — the binding constraint is data, not the model. The durable asset is the *harness* that tells a mirage from an edge. Keep that disconfirming stance.

## Commands

```bash
# Install (core is intentionally lightweight: numpy/pandas/scipy/yfinance only)
pip install -r requirements.txt                 # full stack incl. dashboard
pip install -e ".[dashboard,llm,parquet,dev]"   # editable + optional extras

# Tests — the whole suite is offline & deterministic (no network)
python -m pytest -q                              # run everything
pytest tests/test_harness.py                     # one file
pytest tests/test_harness.py::test_verdict_overfit_gate   # one test
python tests/test_harness.py                     # each test file also has a __main__ runner

# Run the product / research demos (one file per phase & experiment in examples/)
python examples/phase4_demo.py AAPL 1y 1d reversion   # the product (needs internet)
python examples/honest_harness_demo.py                # the harness (offline, <20-line core)
streamlit run streamlit_app.py                        # Phase 5 dashboard
```

Python ≥ 3.10. No linter is configured in-repo; follow PEP 8 and keep/refresh docstrings when modifying modules.

## Architecture (the big picture)

Read `docs/ARCHITECTURE.md` for the module-by-module map and `RESEARCH.md` for the findings. The runtime flow has two tracks over one shared data layer:

- **Act I — product pipeline (rule-based):** `data → profile → regime → scoring → signals → backtest / dashboard`. Each stage is a self-contained module returning an immutable result.
- **Acts II–III — validation/AI stack:** `features · structure · ml` build a no-look-ahead `FactorDataset` → `validation` (purged CPCV) → OOS-IC + **block-permutation** p-value → `stats` (DSR · PBO) + `SurvivorshipInjector` stress sweep → `harness.honest_backtest()` → `HonestReport` + one-line verdict → `insight` (LLM *narrates* a verdict that was computed in code).
- **Act V — data analysis (`vpts.analysis`):** two halves over the same rule. `timeseries` is a *descriptive* diagnostic (distribution · tails · memory via Ljung-Box/variance-ratio/Hurst/ADF · ARCH-LM · drawdown) that claims nothing; `fundamentals` is **point-in-time** statement data behind an injectable-transport `FundamentalsSource`, and `dataset` emits the usual `FactorDataset`/`CrossSectionalPanel` so fundamentals face the same harness. Two non-obvious rules: a `FundamentalSnapshot` **refuses** `available_at < period_end` (period-end joins leak the reporting lag), and `build_fundamental_dataset` samples **one row per filing** — daily sampling of a feature that only changes on filings under-disperses the permutation null and reported `IC +0.16, p=0.005` on pure noise.
- **Data layer (`vpts.data`):** one `DataSource` contract with an honest `provides_delisted` capability flag; `SourceRegistry` is an ordered, capability-aware fallback; `validate_ohlcv` normalises and drops structurally-invalid bars. Sources: `YFinanceSource` (survivors), `StooqSource`/`PolygonSource`/`DataLakeSource` (delisted-capable), `SyntheticSource`. A rotating `ProxyPool` avoids per-IP rate limits.

`structure` depends on `ml`, never the reverse — there are no import cycles. `insight` depends on `harness`, never the reverse (`evidence_from_report`/`explain_report` bridge the two).

## Repo-specific invariants (the non-obvious rules)

- **Dependency-light core.** Core code imports only `numpy`/`pandas`/`scipy` (+ `yfinance` for live data). `plotly`, `streamlit`, `anthropic`, `pyarrow`, `xgboost` are **optional extras, imported lazily inside the function that needs them** — never at module top level — so the package stays importable offline. Match this when adding code.
- **Immutable results.** Every public computation returns a frozen dataclass with `summary()` (and usually `as_dict()`); inputs are never mutated.
- **No look-ahead, enforced.** Features at bar *t* use only data ≤ *t*; labels are strictly future. Dataset builders carry explicit unit tests asserting this — keep them.
- **Network is injectable.** Data sources/fetchers accept an injectable transport (`http_get` / `transport` / `opener_factory`) so the parsing, fallback and rate-limit logic are unit-tested with **no network**. Any new source must follow this seam.
- **One evaluation contract.** A new feature family = build a `FactorDataset` (features → forward return) or `MetaDataset` (→ triple-barrier win) with no look-ahead, then run `cpcv_factor_eval` / `cpcv_meta_eval` + the matching `permutation_test_*`. Significance is always a **label-shuffle permutation p-value**; use the **block-permutation null** when labels overlap (`stride < horizon`) — a per-row shuffle over-rejects.
- **The verdict is code, not the model.** In `vpts.insight`, `assess()` computes the verdict (no_edge / overfit / survivorship_fragile / weak_unvalidated / validated); the LLM only explains it and is scanned for overclaims, with a deterministic template fallback. Never let the LLM decide whether an edge exists.
- **Survivorship is the confound.** Before trusting any positive result, run it through the survivorship-injection sweep (`examples/structural_survivorship.py`, or `frames=`/`feature_builder=` in `honest_backtest`). That is where every promising signal here has died.
- **Honesty discipline.** Disconfirm by default: report new numbers *beside* the old, and do **not** edit a conclusion sentence in `RESEARCH.md` until the new numbers are in. Every evaluator keeps a *signal* test (finds a planted edge) **and** a *null-clearing* test (reports nothing on random input).

## Secrets & config

Credentials are read from the environment, never committed: `ANTHROPIC_API_KEY` (insight LLM), `POLYGON_API_KEY` (Polygon source), `VPTS_PROXIES` / `VPTS_PROXY_FILE` (proxy pool). `proxies.txt` is git-ignored — copy `proxies.example.txt` to create one. With nothing configured, every layer degrades gracefully (offline template, direct fetch).

## Workflow

Standard change cycle for a PR: implement → `python -m pytest -q` (must be green) → review the diff → commit (one logical change per commit) → push to the feature branch → open/merge the PR. When you add or remove tests, update the **test-count references** that drift: the README badge, the README quickstart/layout lines, and the two `RESEARCH.md` counts. Keep `README.md` and `docs/ARCHITECTURE.md` in sync whenever behaviour changes.
