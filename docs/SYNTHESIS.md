# Synthesis — the common spine of three documents

This repository's `docs/` now holds three formal artefacts that look unrelated but are one argument seen from three angles:

- **[`MARKET_EQUILIBRIUM_MODEL.md`](MARKET_EQUILIBRIUM_MODEL.md)** — a general-equilibrium asset-pricing + microstructure model (Levels 1–4), built on one framework (the FTAP: no-arbitrage ⇔ a positive stochastic discount factor) and one guardrail (*existence is not identification*).
- **[`TIMESERIES_MATH.md`](TIMESERIES_MATH.md)** — the mathematics of analysing financial time series, from stationarity to the purged-CV / deflated-Sharpe evaluation gate, grounded in what this codebase computes.
- **[`AI_TIMESERIES_FORECASTING_SWOT.md`](AI_TIMESERIES_FORECASTING_SWOT.md)** — a SWOT of AI/ML architectures for *directional* stock forecasting, concluding that honest out-of-sample directional accuracy is ~50–55% across every model family.

The one-sentence thesis that unifies them:

> **Markets price risk, not odds; models estimate odds, not edge; and the gap between them — the unidentified pricing kernel — is exactly what makes both asset pricing hard and machine-learning backtests lie.**

## The seven cross-cutting insights

**1. The two halves are one theorem from two sides.** The asset-pricing model's deepest result — *the kernel exists but is not identified from prices* (recovery fails because the SDF's permanent component is unidentifiable, §IV.L6) — is the **same wall** the ML models hit as the ~50–55% directional ceiling. To turn a price into a forecast you need the kernel; the kernel is not recoverable; so a model "predicting returns" cannot cleanly separate the market's *belief* from the *risk premium*. The non-identification of the physical measure **is** the coin-flip wall, expressed in probability rather than price.

**2. "Price ≠ probability" and "accuracy ≠ edge" are the same caution.** A 51%-accurate model posting a large pre-cost Sharpe (e.g. the benchmark CatBoost in Rahimikia et al. 2025) is the empirical shadow of the theoretical point that the risk-neutral density is a *valuation*, not a *forecast*. Hit-rate is nearly orthogonal to P&L; what pays is sizing/ranking over the risk-weighted distribution, not getting the sign right more often.

**3. The binding constraint is identification and data — never model capacity.** Across the SWOT, the *between-family* accuracy spread (~50–55%) is smaller than the *leaked-vs-clean* evaluation spread within any one family. In the asset-pricing model, the unknowable part is a specific object (the permanent kernel component). Both say: more model is the wrong lever; better data, honest evaluation, and naming what is structurally unknowable are the right ones. (`RESEARCH.md` reaching the identical verdict empirically — "the binding constraint is the data, not the model" — is this insight discovered the hard way.)

**4. Bias–variance, made economic.** On a near-random-walk, low-SNR target, flexible models spend capacity fitting noise — which is *why* transformers and deep RL lose to trees and regularized baselines here ([`TIMESERIES_MATH.md`](TIMESERIES_MATH.md) §9b, eq. 9.7). The asset-pricing doc's "GBM is a measure-zero corner" is the continuous-time twin: the world is jumps + stochastic volatility + heavy tails, so a model assuming smooth structure overfits precisely where there is nothing real to fit.

**5. The endogenous-kernel rule and the no-look-ahead rule are the same discipline.** Both forbid smuggling in the answer. Don't exogenize the discount rate — you'd assume the risk premium you are trying to explain. Don't let future data touch training — you'd assume the prediction you are trying to claim. One sin, two domains; purged + embargoed cross-validation is the econometric image of the endogenous-kernel requirement.

**6. Where edge plausibly lives is consistent across both analyses.** *Relative* (cross-sectional ranking) over *absolute* (directional timing); risk premia harvested **knowingly** rather than mistaken for alpha; structural / microstructure features over generic price-pattern learning. And the honest ceiling is low *and decaying* — reflexivity / signal decay (McLean–Pontiff) means the act of finding and trading a signal erodes it.

**7. The verification gate is not bureaucracy — the math guarantees the naive number will lie.** Backtest overfitting yields *negative* out-of-sample returns, not merely zero (Bailey–Borwein–López de Prado–Zhu). So the gate — purged + embargoed CV, the **up-rate** (not 50%) baseline, transaction costs, the deflated Sharpe / PBO / t>3 corrections, point-in-time + delisting-inclusive data, and reporting P&L *and* AUC rather than raw accuracy — is the only thing standing between you and a confidently-wrong result.

## From theory to code

Insights 6 and 7 are not just advice — they are implemented in this repository as a runnable, tested recipe:

- **`vpts.ml.gbrt`** — `GradientBoostedTrees`, a dependency-free (numpy) gradient-boosted regression-tree learner, the survey's best-evidence family for noisy tabular financial data (insight 3/6), plus `gbrt_cross_sectional_eval`, which scores it under the existing **purged + embargoed CPCV** (`vpts.validation`) and reports directional accuracy against the **unconditional up-rate**, never 50% (insight 7).
- **`vpts.ml.significance`** — `deflated_sharpe_ratio` (Bailey–López de Prado), `probabilistic_sharpe_ratio`, `expected_max_sharpe`, and `directional_skill` (accuracy minus the majority-direction baseline): the significance gate that corrects a Sharpe for the number of trials and makes a null result correctly fail.
- **`examples/gbrt_significance_demo.py`** — the full recipe end-to-end on synthetic data: GBRT → CPCV → directional-vs-up-rate → deflated-Sharpe gate. Run it with and without a planted signal (`--null`) to watch the harness find signal when present and collapse to "no edge" when not — *including* the deflated Sharpe, which is computed on the out-of-sample IC stream so the null does not falsely pass.

The demo is also a live illustration of insight 5: an earlier version computed the significance gate on an *in-sample* refit and the null spuriously "passed" — the leakage the documents warn about, leaking into the demo itself. Running the gate on the out-of-sample stream is what makes the discipline real rather than decorative.
