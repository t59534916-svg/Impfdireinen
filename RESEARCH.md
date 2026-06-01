# Does `vpts` have a real edge? — an honest validation log

This document records a deliberately adversarial search for **out-of-sample, survivorship-free
predictive edge** in the Volume-Profile system (`vpts`). It is written to be read by a skeptic.
The value delivered is *validated* findings — mostly negatives, one qualified positive — plus a
reusable harness that judges any future idea honestly.

> **Bottom line.** Across eight experiments — a walk-forward backtest and seven fitted models, all
> evaluated with purged combinatorial cross-validation and label-shuffle permutation tests — six
> inputs produced **no** robust out-of-sample edge (including a cross-sectional near-miss that
> vanished when properly powered). The **structural microstructure features** (synthetic delta,
> profile shape, value-area-compression z-score, cost-basis migration) are the **exception**: a
> small but real OOS signal (IC ≈ +0.035) that — uniquely — **survives both** widening to 88 names
> (p = 0.005) **and** low rates of synthetic-delisted injection, degrading *gracefully* rather than
> collapsing like everything before it. It is **modest, gross-of-cost, and survivorship-*sensitive***
> (it loses significance above a ~15–20% delisting rate), so it is a credible research signal, **not
> a validated tradeable edge.**

---

## The question

`vpts` generates directional biases from hand-set confluence weights over Volume-Profile, regime
and volume-pattern factors. A single backtest of the breakout style on 2012–2017 large-caps showed
**+14.5%**. The question this log answers is not "is that number positive?" but:

> Is there any **learnable, out-of-sample, survivorship-free** signal in these factors — or is the
> apparent performance drift, compounding, and survivorship?

## Methodology (the harness)

Every claim below clears the same bars, implemented in `vpts.validation` and `vpts.ml` and covered
by 121 unit tests:

- **No look-ahead.** Features at bar *t* use only data ≤ *t*; labels are strictly future. The
  dataset/panel builders are unit-tested for this.
- **Purged + embargoed CPCV** (`CombinatorialPurgedCV`, López de Prado). The timeline is split into
  groups; every combination of test groups is held out; train rows whose label window overlaps a
  test block are **purged**, and a post-block **embargo** breaks serial-correlation leakage. Scores
  are distributions over recombined OOS paths, not a single split.
- **Permutation significance.** The decisive test everywhere is a label shuffle that destroys the
  feature→outcome link while preserving structure (per-row for time-series, **within-date** for the
  cross-section). The p-value is the fraction of shuffles that match or beat the real statistic. An
  effect that cannot clear its own shuffled null is reported as no edge.
- **Honest scope, stated every time.** All data below is **survivorship-biased** (see Data). These
  are validity checks on OOS information content, **not** tradeable results.

## Data

Free, no-API-key, network-restriction-friendly: split/dividend-adjusted daily OHLCV for **88 US
large-caps, 2012–2017**, committed to the public [`stocknet-dataset`](https://github.com/yumoxu/stocknet-dataset)
(`vpts.data` back-adjusts via Adj Close / Close). **Every name is a 2017 survivor** — the dominant,
unavoidable confound throughout. There is no delisted/point-in-time data in this source.

---

## The eight experiments

| # | Experiment | OOS statistic | Significance | Verdict |
|---|------------|---------------|--------------|---------|
| 1 | Rule-based backtest, CPCV (8 names, 80 paths, net 5 bps) | **−0.68%/path**, median −1.20%, 36% paths profitable | — | apparent +14.5% was drift/compounding; **no edge** |
| 2 | Learned ridge factor weights (CPCV) | OOS IC **+0.028** | did not beat the hand-set baseline | **no learnable improvement** |
| 3 | Triple-barrier **meta-labeling** | survivors AUC **0.576** (p=0.005) → with delisted injected **0.493** | p **0.801** | **survivorship artifact** |
| 4 | **Enriched** per-name features (momentum/vol/microstructure) | pooled IC **+0.010** (baseline +0.028) | p **0.348** | richer inputs don't help; **no edge** |
| 5 | **Cross-sectional rank**, 20 names | combined OOS IC **+0.021** | p **0.100** | suggestive, **not significant** |
| 6 | **Cross-sectional rank, 88 names (well-powered)** | combined OOS IC **−0.009** | p **0.856** | near-miss **washed out**; **no edge** |
| 7 | **Structural microstructure** (synthetic delta, shape, VACR-z, decay) | OOS IC **+0.103** (8 names) → **+0.035** (88 names, 1,308 folds) | p **0.005** (both) | **real signal — survives widening** |
| 8 | **Structural + survivorship injection** | pooled IC +0.041 → +0.013 (5 dead, 20%) → +0.001 (9 dead, 31%) | p 0.005 → **0.085** → 0.473 | **survivorship-*sensitive*; graceful decay, not a cliff** |

### 1 — The single backtest doesn't survive purged CV
The breakout style's +14.5% (85% of names profitable, single full-period backtest) collapses under
CPCV to **−0.68% per OOS path**, median −1.20%, only 36% of paths profitable. The apparent edge was
bull-market drift and compounding — exactly what rigorous validation is meant to expose.

### 2 — Learning the factor weights doesn't help
A ridge model fit on the four confluence factors (train-only standardization, OOS-scored per CPCV
fold) reaches pooled **OOS IC ≈ +0.028** and does not beat the hand-weighted `bias_score` baseline.
No improvement from learning the weights.

### 3 — Meta-labeling is significant *only because of survivorship*
Predicting whether a primary signal *works* (triple-barrier, volatility-scaled, first-touch) and
filtering on it looked real on survivors: pooled **AUC 0.576, p=0.005**, cost-surviving and
threshold-stable. But injecting synthetic **delisted** names (a vol-elevated decline to pennies)
collapses the pooled permutation test to **AUC 0.493, p=0.801**. A per-name AUC t-test stayed >0.5
only because each decliner got its own model; the realistic single cross-sectional model has no
edge. **Survivorship was the explanation.**

### 4 — Genuinely new per-name features don't rescue it
Adding momentum (20/60/12-1), volatility (σ, ATR/price), volume-trend and distance-to-POC — 11
features through the same harness — yields pooled **IC +0.010**, *below* the 4-factor baseline
(+0.028), at **p=0.348**. Ridge shrank every weight to ≈0. Richer inputs carry no OOS signal here.

### 5 → 6 — Cross-sectional rank: a near-miss that proper power kills
Ranking names against each other each rebalance day (1-month reversal, 12-1 momentum, 60-day vol,
volume-trend) is the standard equity-alpha construction the per-name models never tried. On **20
names** it was the best result of the arc — combined OOS rank IC **+0.021, p=0.100** — but with only
~20 names per date the per-date IC is dominated by noise (σ 0.28). Per-date IC noise scales ~1/√N,
so the decisive test is width: re-run on the **full 88-name** universe (16,873 rows, σ 0.20). The
faint positive **washes out to −0.009, p=0.856** — and the strongest single factor (60-day vol,
+0.045 on 20 names) decays to +0.013. The near-miss was a thin-cross-section artifact, not signal.

### 7 → 8 — Structural microstructure: the one signal that survives stress
Transforming the static profile into quantifiable features — **synthetic delta** (Close-Location-Value
× volume, an OHLC order-flow estimate), volume-weighted **skew/kurtosis** and P/b/B/D **shape**,
**value-area-compression z-score**, **POC-migration slope**, **cost-basis migration** (decayed vs
lifetime POC), ledges and poor highs — 13 features through the same harness. This is the first input
to clear the bars:

- **8 survivors:** pooled OOS IC **+0.103**, p **0.005** (the single delta@POC feature alone is
  −0.043; the *combination* predicts).
- **Stress 1 — widening to 88 names:** IC shrinks to **+0.035** but, with 1,308 folds (null σ 0.006),
  is still ≈5.8σ out, **p 0.005**. Unlike the cross-sectional near-miss it **did not wash out** —
  proof it is not a small-sample artifact. Per-name dispersion is sensible: BABA scores **−0.424**
  (a genuine decliner — the dip-features correctly *anti*-predict).
- **Stress 2 — survivorship injection:** adding synthetic decline-to-pennies names degrades the
  signal *gracefully* — +0.041 (0 dead, p 0.005) → +0.029 (1, p 0.015) → +0.013 (5 dead ≈20%,
  **p 0.085, lost**) → +0.001 (9 dead ≈31%, p 0.473). This is **categorically unlike meta-labeling**,
  which collapsed from p 0.005 straight to p 0.80. The structural signal survives *low, realistic*
  large-cap delisting rates (≲10%) but **not** heavy survivorship (≳15–20%).

So a meaningful fraction of the +0.035 IC is genuine and a fraction is survivorship-inflated. The
most survivorship-*robust* components are plausibly the regime/volatility features (`vacr_z`) rather
than the dip-buying ones (`cost_basis_migration`, `poc_loc`); decomposing that is the natural next
experiment. The signal is **small and gross-of-cost** — a credible research lead, not a tradeable edge.

---

## Honest conclusion

On 88 survivorship-biased US large-caps (2012–2017, daily), six of eight studied inputs — the
hand-set rules, learned factor weights, meta-labeling, enriched per-name features, and cross-sectional
ranks — show **no** robust out-of-sample edge (the meta-labeling "edge" was survivorship; the
cross-sectional near-miss was low power). The **structural microstructure features** are the genuine
exception: a **small, real OOS signal** (IC ≈ +0.035) that survives universe-widening and low rates
of delisted injection, degrading gracefully rather than collapsing. It is **modest, gross-of-cost and
survivorship-sensitive** — strong enough to call a credible research lead, not strong enough to call a
validated tradeable edge.

**What would actually change this** (in rough order of expected value):

0. **Decompose the structural signal** — which features survive survivorship injection? If `vacr_z`
   (a regime/breakout feature, not survivorship-prone) carries it, that is a small genuine edge worth
   pursuing; if only the dip-buying features do, it is mostly survivorship.
1. **Survivorship-free / point-in-time data**, including delisted names — the dominant confound,
   untestable in this source. This is the real wall, not model complexity.
2. **A wider, deeper cross-section** (hundreds–thousands of names). The 88-name washout suggests
   breadth *within survivors* isn't enough; genuine breadth + delisted names is the test.
3. **Different data regimes** — intraday microstructure, or non-equity assets where Volume-Profile
   structure may carry more information.

Model sophistication mostly is **not** the answer — four feature/model variations returned ≈0 — but
the *right kind* of feature (structural microstructure, not momentum/vol/rank) did surface the arc's
one real signal. The lesson: feature *content* mattered where feature *complexity* did not.

## What is durable here

The findings — six negatives and one qualified positive — are the result; the **harness** is the
asset. Any new idea plugs in and is judged honestly:

- `vpts.validation` — purged + embargoed Combinatorial Purged CV.
- `vpts.ml` — no-look-ahead dataset/panel builders, ridge/logistic models, CPCV evaluators, and
  label-shuffle permutation tests for per-name, meta-labeling, and cross-sectional settings.
- `vpts.structure` — synthetic delta, profile-shape moments, footprints and time-decay, emitted as a
  `FactorDataset` straight into the harness; plus a survivorship-injection stress test.
- 132 unit tests, including signal-detection *and* null-clearing checks for every evaluator.

## Reproduce

```bash
python examples/github_data_scan.py --plot .          # 1: backtest sweep / regime split
python examples/cpcv_demo.py                          # 1: CPCV on the backtester
python examples/factor_model_demo.py                  # 2: learned factor weights, OOS
python examples/meta_labeling_demo.py                 # 3: triple-barrier meta-labeling
python examples/meta_stress_test.py                   # 3: + survivorship injection
python examples/enriched_factor_demo.py --perms 200   # 4: enriched features + permutation
python examples/cross_sectional_demo.py --perms 200   # 5: cross-sectional rank (20 names)
# 6: well-powered cross-section — pass the full 88-name universe via --tickers
python examples/structural_demo.py --perms 200        # 7: structural microstructure features
python examples/structural_survivorship.py            # 8: structural + survivorship injection
```

## Limitations

Survivorship bias throughout; a single 2012–2017 in-sample period; daily bars only; gross-of-cost
except where noted (meta-labeling tested net of 10 bps); a thin universe by cross-sectional
standards. None of the above is a forward guarantee or financial advice — it is a research log.
