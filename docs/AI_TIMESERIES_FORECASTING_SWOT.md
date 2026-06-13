# SWOT Analysis of AI Architectures & ML Model Designs for Directional Stock Forecasting (Non-HFT)

*Deep-research report. Lens: quant-practitioner, standalone. Evaluation anchor: **directional / classification accuracy** (up/down hit-rate, AUC, precision/recall/F1) — not point-forecast RMSE and not P&L, except to contextualize. Horizons: daily / weekly / swing (minutes-to-months), explicitly not microsecond HFT.*

*Sourcing caveat: web-page fetching was blocked during research, so claims rest on search-engine extractions of the primary papers plus independent re-verification of the load-bearing figures. Confidence is flagged throughout; recency-sensitive preprints whose exact figures could not be re-confirmed are marked accordingly.*

---

## 0. Bottom line up front

1. **The realistic out-of-sample directional-accuracy band for genuine next-period equity direction is ~50–55%**, across *every* model family. ~50% is the no-skill / weak-form-efficiency null; a real edge is a couple of points above the **unconditional up-rate** (~53–55% for indices, because markets rise more often than they fall) — *not* above 50%. Anything reliably >55% is rare; >60% is a strong leakage signal; >70% is almost always an artifact (random-shuffle CV, look-ahead denoising, survivorship, or P&L-conflation).

2. **Model sophistication is not the binding constraint — data quality, evaluation discipline, and the low signal-to-noise ratio are.** On the directional task, gradient-boosted trees ≈ LSTMs ≈ transformers ≈ foundation models ≈ a well-regularized logistic baseline, all clustered near coin-flip. The differences between families are smaller than the differences between honest and dishonest evaluation of the *same* family.

3. **Three category errors dominate the field and inflate reported numbers**, and the report returns to them repeatedly: (i) **point-forecast accuracy ≠ directional accuracy** (a model can top the RMSE leaderboard with zero sign skill); (ii) **directional accuracy ≠ tradeable edge** (base rates, class imbalance, transaction costs, and magnitude-vs-sign mismatch break the link); (iii) **in-sample fit ≠ out-of-sample skill** (multiple-testing and leakage manufacture both).

4. **Where AI genuinely helps is cross-sectional *ranking*, not binary timing** — tiny per-name predictability (~0.4% monthly R²) aggregated into a diversified long-short book — and even that is fragile to costs, microcaps, and limits-to-arbitrage.

---

## 1. The evaluation frame (why most reported accuracy is untrustworthy)

**The ~50% wall and the right baseline.** Out-of-sample directional accuracy for equity direction clusters around 50%, with weak-but-real skill topping out in the low-to-mid 50s. Crucially, raw accuracy must be benchmarked against the **unconditional up-rate (~53–55%)**, not 50%: a constant "always up" predictor already scores ~55% on US returns, and boosted trees in honest studies improve on that by only ~2 points [Gu–Kelly–Xiu 2020, *RFS*; multi-market comparative study, *Int. J. Data Science & Analytics* 2025]. **Confidence: high.**

**Directional accuracy ≠ tradeable edge.** Accuracy and profitability are not monotonically linked once magnitudes and asymmetry enter: ~40% accuracy can be profitable and ~60% can be needed to "guarantee" profit, depending on payoff structure [Gui 2024, arXiv:2407.09831]. The naive random-walk/last-value forecast is "notoriously difficult to surpass," and a directional overlay only beats it once accuracy exceeds ~0.55 [Zhang 2024, arXiv:2406.14469]. **Confidence: medium-high.**

**Point-forecast accuracy ≠ directional accuracy.** Most deep-learning time-series benchmarks (ETT, electricity, traffic, weather, exchange-rate) are *non-financial* and scored by **RMSE/MAE**, which rewards persistence-like forecasts that carry zero sign information on a near-random-walk series. A model can rank #1 on forecast error and still deliver ~50% directional accuracy [convergent finding across the DL literature; a controlled comparison of 9 architectures reports directional accuracy ≈ 50% even for the RMSE-leaderboard winners (ModernTCN best forecast-error rank, yet ~coin-flip on direction) — Saidd 2026, arXiv:2603.16886, "918 Experiments," verified]. **Confidence: high.**

**Why >60% is a red flag.** High-accuracy papers almost always carry one of: data shuffled before the train/test split (sequential leakage); k-fold CV on time series without **purging + embargo**; look-ahead features (full-sample standardization/denoising, restated fundamentals, survivorship-filtered universes); P&L reported as if it were classification skill. The canonical example: an xLSTM reporting ~73% F1 whose accuracy "was vital[ly]" dependent on wavelet denoising applied over the full series *including test data* — a textbook look-ahead leak [López Gil et al. 2024, arXiv:2408.12408]. The reproducibility literature documents this as systemic: a taxonomy of 8 leakage types across 329 papers in 17 fields, with many complex-model "wins" vanishing once leakage is fixed [Kapoor & Narayanan 2023, *Patterns*]. **Confidence: high.**

**Statistical significance, corrected for the search.** Given pervasive data mining, the conventional t>2 hurdle is wrong; a newly "discovered" factor needs **t>3.0**, and ~half of published factors are likely false [Harvey–Liu–Zhu 2016, *RFS* 29(1):5–68 — verified]. Backtest overfitting is "easily achievable" with few trials, and under memory effects produces **negative** expected OOS returns, not merely zero [Bailey–Borwein–López de Prado–Zhu 2014, *Notices of the AMS* 61(5) — verified]. Remedies: Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV), purged/combinatorial-purged CV [Bailey–López de Prado 2014; López de Prado 2018]. **Confidence: high.**

**The epistemic ceiling.** Any test of predictability is jointly a test of an assumed asset-pricing model — the **joint-hypothesis problem** [Fama 1991] — so "beats the market in backtest" is never clean evidence of edge. And with enough trials (strategies, seeds, papers), long winning runs occur by luck. **Confidence: high.**

---

## 2. Per-family SWOT

### 2A. Classical / tabular ML — gradient-boosted trees (XGBoost/LightGBM/CatBoost), random forests; regularized linear & SVM

The credible workhorses for tabular, noisy, medium-N financial cross-sections. Tree ensembles are state-of-the-art on exactly this data regime (≈10K-row, noisy, irregular targets), beating deep learning on tabular benchmarks even ignoring their speed [Grinsztajn et al. 2022, NeurIPS]. In the cleanest peer-reviewed asset-pricing study, trees and NNs dominate linear models — but the edge is a **~0.33–0.40% monthly per-stock OOS R²**, realized economically through cross-sectional *ranking* (long-short decile Sharpe ~1.35 value-weighted, ~2.45 equal-weighted, **pre-cost**), not high binary hit-rates [Gu–Kelly–Xiu 2020, *RFS* — verified]. The dominant signals (momentum, liquidity, volatility) are the same across all families.

| | Tree ensembles | Regularized linear / SVM |
|---|---|---|
| **Strengths** | Best on medium-N noisy tabular data; capture nonlinear feature interactions; robust to many uninformative features (financial sets are mostly noise); fast; usable feature importances | Transparent, low-variance, hard to overfit; logistic gives calibrated up/down probabilities & clean AUC; LASSO does interpretable selection; the honest baseline complex models must beat |
| **Weaknesses** | Overfit noisy targets; ~51–55% real directional ceiling once leakage/purging enforced; no native temporal handling (must engineer lags); unstable importances on correlated features; poor extrapolation across regimes | Miss the nonlinear interactions where the (small) gains live; linear boundary poor for irregular targets; SVM scales badly, sensitive to kernel/C and scaling; awkward probabilities for AUC |
| **Opportunities** | Cross-sectional ranking where a tiny per-name edge aggregates into Sharpe; regime-conditional models; ensembling; honest CPCV | As the reproducibility anchor — if a tree/NN can't beat regularized logistic under purged CV, the "edge" is leakage; double-selection LASSO for honest factor selection |
| **Threats** | Leakage/survivorship inflation; alpha decay as signals go public [Krauss et al. 2017 show even real stat-arb edges decay sharply post-2001]; costs erasing thin edges; non-stationarity breaking the i.i.d. assumption | Same leakage/non-stationarity/cost threats; over-claiming when authors omit the linear baseline (itself a red flag) |

**Verdict:** The strongest *evidence-backed* family for non-HFT equity — but "best" means a fraction-of-a-percent R² that aggregates via ranking, decays over time, and is cost-fragile. Per-name daily up/down is near coin-flip. **Confidence: high.**

### 2B. Deep-learning sequence models — RNN/LSTM/GRU, TCN, Transformers (Informer/Autoformer/FEDformer/PatchTST), N-BEATS/N-HiTS, TFT

The literature is overwhelmingly built and benchmarked on *non-financial, non-directional* tasks scored by RMSE/MAE. Two findings dominate. First, **simple linear models match or beat transformers** on the standard long-horizon benchmarks — DLinear outperforms FEDformer by 40%+ on exchange-rate, ~30% on traffic/electricity/weather [Zeng et al. 2023, AAAI Oral, "Are Transformers Effective…?"] — and the *only* financial series in that suite (FX levels) is scored by MSE, where a near-random-walk makes "predict last value" hard to beat and directionally uninformative. Second, when DL architectures are evaluated *directly on direction*, hit-rates cluster at ~50% even for the models that top the RMSE leaderboard.

| | RNN/LSTM/GRU | TCN | Transformers (incl. PatchTST) | N-BEATS/N-HiTS/TFT |
|---|---|---|---|---|
| **Strengths** | Most-tested on financial direction; handles path-dependence; modest pre-cost edge documented [Fischer–Krauss 2018: 0.46%/day, Sharpe 5.8 pre-cost — verified] | Best RMSE rank in some controlled financial comparisons; parallelizable; stable long receptive fields | SOTA on non-financial RMSE benchmarks (PatchTST +21% MSE); patching/channel-independence genuine advances | Interpretable basis-expansion; TFT gives native quantile/probabilistic output, variable selection, handles known-future covariates — attractive for finance |
| **Weaknesses** | Hit-rate barely >50%; overfits low-SNR data; edge cost-fragile & decays post-2010; often *matched by random forests* | Top RMSE rank still ≈50% directional — strong point-forecast ≠ sign skill; no large dedicated directional validation | DLinear shows linear layers match/beat them; ~50% directional on real finance; biggest data-hunger-vs-SNR mismatch of all families | All validated on point/quantile loss on non-directional data; ~50% directional in financial tests; "finance success" stories are third-party adaptations with frequent red flags |
| **Opportunities** | Leak-free denoising, noise-injection regularization, hybrids with trees/sentiment | Efficient backbone for a classification head with a direction-specific (not MSE) loss | Multimodal price+text; PatchTST as a feature extractor feeding a calibrated classifier | TFT quantile output → probability-of-up calibration; covariate handling suits fundamentals/calendar features |
| **Threats** | Look-ahead-biased denoising inflates results; regime shifts; multiple-testing false discovery | Same low-SNR ceiling; RMSE-leaderboard chasing misleads on the real objective | "Effectiveness" contested even on benchmarks; attention overfits spurious cross-series structure | MSE/pinball objective misaligned with sign accuracy; interpretability can manufacture false confidence |

**Verdict:** Data-hungry models on a low-SNR, near-random-walk target — a structural mismatch. No DL family reliably beats well-built tree ensembles on directional finance; the headline DL advances are point-forecast gains on non-financial data. **Confidence: high — the ~50% directional finding is multiply corroborated, including the controlled 9-architecture "918 Experiments" study (arXiv:2603.16886, verified) in which the RMSE-rank winner is still ~coin-flip on direction.**

### 2C. Time-series foundation models (2023–25) — TimeGPT, Chronos, TimesFM, Moirai, Lag-Llama, MOMENT

Genuinely impressive *zero-shot point/probabilistic accuracy* on general (non-financial) benchmarks (Chronos, TimesFM, Moirai are competitive zero-shot vs supervised models on MASE/WQL). But trained/evaluated overwhelmingly on non-financial corpora (web traffic, electricity, M4), and the decisive finance-specific test is negative: the first comprehensive study of TSFMs in global markets finds off-the-shelf models perform **poorly zero-shot and fine-tuned**, with directional accuracy "all just above 51%" across the four windows and negative backtested returns off-the-shelf; only models **pretrained from scratch on financial data** (and strong benchmarks like CatBoost) improved [Rahimikia–Ni–Wang 2025, arXiv:2511.18578 — paper, ~51% directional figure, and core finding all verified]. *Note the "accuracy ≠ edge" corroboration: the benchmark CatBoost still posted ~46% annualized / Sharpe ~6.8 pre-cost at the same ~51% directional accuracy — sizing/ranking, not hit-rate, drives the (pre-cost) economics.* Independent work shows specialized FMs struggle to beat supervised baselines [Xu et al., ICLR 2025], and benchmark leakage/contamination inflates published zero-shot numbers (the GIFT-Eval suite had to construct an explicit "non-leaking" pretraining set) [Aksu et al. 2024, arXiv:2410.10393].

- **Strengths:** Real zero/few-shot capability; no per-series training; good uncertainty quantification; strong on *structured/periodic/trending* signals — useful for **auxiliary** financial series (volatility, volume seasonality, macro nowcasting) rather than raw return direction; mostly open weights (auditable).
- **Weaknesses:** No demonstrated directional edge on equity returns (~51%, negative P&L off-the-shelf); trained on non-financial corpora; squared/quantile objectives bias toward persistence (= zero information on a random walk); modest absolute gains over naive baselines even on home turf.
- **Opportunities:** Domain-specific *from-scratch* pretraining (FinCast, FinText-TSFM) is where gains appear; fine-tuning + exogenous covariates; multivariate/cross-series conditioning; better evaluation (directional/AUC/net-of-cost, not RMSE-on-price).
- **Threats:** Benchmark contamination inflates published numbers; low-SNR + non-stationarity may impose a hard ceiling no pretraining transfer overcomes; closed models (TimeGPT) resist verification; single-name "85% accuracy" claims [e.g. arXiv:2412.09394] are unreproducible artifacts.

**Verdict:** Impressive general-purpose forecasters; **no evidence of a directional stock edge off-the-shelf.** The honest signal is that domain-specific financial pretraining — not zero-shot transfer — is the only path that has helped. **Confidence: high for the qualitative conclusion.**

### 2D. Reinforcement learning — DQN, PPO/A2C/DDPG/SAC/TD3, FinRL

A distinct paradigm: RL frames trading as a policy (actions = buy/sell/hold or weights) optimizing a return-based reward, folding direction + sizing + costs into one objective. But a discrete-action agent over {long, flat, short} *is* a directional classifier whose label is chosen to maximize reward instead of cross-entropy — so RL doesn't escape the directional problem, it adds credit-assignment, exploration, and sample-efficiency burdens on top. The evidence base is weaker than the supervised literature's: deep-RL is notoriously seed/hyperparameter/implementation-fragile [Henderson et al. 2018, AAAI, "Deep RL That Matters"], and finance-RL papers almost never report seed variance, DSR, PBO, or significance tests.

- **Strengths:** Directly optimizes the economic objective (PnL/Sharpe net of costs/turnover) rather than a proxy classification loss; naturally handles sequential, path-dependent costs and inventory; continuous-action methods express portfolio weights directly.
- **Weaknesses:** Severe sample inefficiency vs scarce, autocorrelated financial data; seed/hyperparameter fragility makes single-run results uninterpretable; weak evaluation culture (single historical path, rarely DSR/PBO/purged walk-forward); DQN-family "breaks" under regime shift, PPO degrades under abrupt shifts; reward shaping invites reward hacking (Sharpe-as-reward optimizes the eval metric).
- **Opportunities:** Transplant quant-stats rigor (CPCV + DSR + PBO + multi-seed) as a publication standard; FinRL Contests pushing reproducible benchmarks; hybrid supervised-signal-into-RL-allocation pipelines; robust/risk-aware/distributional RL targeting non-stationarity.
- **Threats:** Non-stationarity is *structural* — it violates the MDP foundation and caps generalization regardless of data/model improvements; survivorship/look-ahead inflate backtests; the RL research loop (many seeds × architectures × rewards) mechanically manufactures inflated Sharpe; head-to-head evidence that a **supervised classifier beat recurrent-RL/DQN/A2C** on crypto undercuts the paradigm's core justification [*Expert Systems with Applications* 2022, S0957417422006339].

**Caution on the famous positive result:** the FinRL ensemble's reported Sharpe ~1.53 (DJIA, 2020-07→2022-03) sits on a single historical path straddling the COVID rebound (a long-favorable regime), with no seed-variance, DSR, or significance test — the archetype of "strong backtest, untested out-of-distribution" [Yang et al. 2020, ICAIF].

**Verdict:** A coherent and arguably more honest *framing* of trading; an *evidence base* dominated by single-path, single-seed, no-significance backtests on favorable windows. Treat reported RL Sharpe as a selection-inflated upper bound until DSR/PBO/multi-seed reporting is standard. **Confidence: high for the methodological critique.**

---

## 3. Cross-model synthesis

| Family | Realistic OOS directional accuracy | Data efficiency on scarce financial data | Overfitting risk (low-SNR) | Non-stationarity robustness | Interpretability | Practitioner verdict |
|---|---|---|---|---|---|---|
| **Tree ensembles** | ~51–55% (per-name near coin-flip; edge via ranking) | High (best on medium-N tabular) | Moderate (controllable via depth/regularization/CPCV) | Low (i.i.d. assumption) | Medium-high (feature importances) | **Default choice.** Best evidence-backed; use for cross-sectional ranking, not binary timing |
| **Regularized linear / logistic** | ~50–54% | Highest (low variance) | Lowest | Low–moderate | Highest | **The mandatory baseline.** If a complex model can't beat it under purged CV, the edge is leakage |
| **RNN/LSTM/TCN** | ~50–54% | Low–moderate (data-hungry) | High | Low | Low | Use only if path-dependence is essential and data is ample; often matched by RF |
| **Transformers (PatchTST etc.)** | ~50% | Lowest (most data-hungry) | Highest | Low | Low | Point-forecast tool; weakest fit for low-SNR direction; consider as a feature extractor only |
| **N-BEATS/N-HiTS/TFT** | ~50% | Low–moderate | High | Low | Medium (TFT) | TFT's covariate/quantile handling useful for probability-of-up calibration; not a directional edge by itself |
| **Foundation models (zero-shot)** | ~51%, negative P&L off-the-shelf | N/A (pretrained) but no transfer to returns | N/A zero-shot; high if fine-tuned on little data | Low | Low | Auxiliary series (vol/macro) only; needs from-scratch financial pretraining to help on returns |
| **Reinforcement learning** | Implicit; evidence weak/inflated | Lowest (very sample-hungry) | Very high | Very low (breaks MDP) | Low | Compelling framing, weak evidence; supervised classifiers have beaten it head-to-head |

**The unifying picture:** the *between-family* spread in honest directional accuracy (~50–55%) is smaller than the *within-family* spread between leaked and clean evaluation. Inductive biases matter less than they should because the target is near-random-walk and low-SNR: flexible models (transformers, deep RL) spend their capacity fitting noise (high variance), while simpler, regularized models (trees, logistic) lose little by being less flexible — the bias-variance tradeoff resolves toward simplicity here. This is why "use a bigger model" is usually the wrong answer on this problem.

---

## 4. The methodological red-flags checklist (apply to any reported result, including your own)

A directional-prediction number is probably untrustworthy if **any** of these hold:

1. **Headline directional accuracy >60%** for genuine next-period *direction* (not volatility sign, not a dominant-class target). >70–90% ≈ near-certain leakage.
2. **Data shuffled before the split**, or **k-fold CV** on time series without **purging + embargo**.
3. **Look-ahead / point-in-time violations**: full-sample standardization, full-sample denoising/feature-selection, restated fundamentals, survivorship-filtered universe.
4. **Wrong baseline**: accuracy compared to 50% rather than the unconditional up-rate (~53–55%) or a naive/random-walk forecast.
5. **No transaction costs, turnover, slippage, or capacity** in the P&L. ML edges are turnover-heavy and concentrated in microcaps/illiquid names — costs routinely erase them [Avramov–Cheng–Metzker 2023, *Mgmt Sci* — verified].
6. **No multiple-testing-corrected significance**: no trial count, no DSR/PBO, no t>3 hurdle.
7. **In-sample-only or single fixed split**; no walk-forward / true OOS.
8. **Accuracy/F1 under class imbalance** with no AUC/MCC and no realized-return P&L (majority-class voting masquerading as skill).
9. **Survivorship bias**: universe excludes delisted/dead tickers; results cherry-picked across many strategies/assets/seeds.
10. **"Beats the market" treated as proof of edge**, ignoring the joint-hypothesis problem and favorable single-regime windows.

---

## 5. Practical model selection by data regime and horizon

- **Cross-sectional, many names, monthly–weekly (the strongest-evidence setting):** gradient-boosted trees or NNs for *ranking*, evaluated by rank-IC and decile long-short under purged/combinatorial-purged CV, net of realistic costs. This is where peer-reviewed predictability actually lives.
- **Single-series daily/swing direction:** start with regularized logistic / gradient-boosted trees on lagged, leakage-safe features; treat ~52–55% as the ceiling; require the model to beat both the up-rate and a logistic baseline under walk-forward-with-embargo before believing it.
- **When sequence/path-dependence is genuinely informative and data is ample:** LSTM/TCN — but A/B against trees; they often tie.
- **Probabilistic, multi-horizon, with covariates:** TFT for calibrated probability-of-up and uncertainty, not as a standalone edge.
- **Auxiliary series (volatility, volume, macro nowcasting):** foundation models (Chronos/TimesFM/Moirai) zero-shot are reasonable here — *not* for raw return direction.
- **Position/allocation optimization with explicit costs:** RL is a defensible framing, but only with multi-seed reporting, DSR/PBO, and out-of-regime testing; otherwise prefer "supervised signal → rules-based sizing."
- **Universal gates (all of the above):** purged + embargoed CV; an honest baseline (up-rate + logistic); transaction costs; multiple-testing correction (DSR/PBO/t>3); a delisting-inclusive, point-in-time universe; report P&L and AUC, not accuracy.

---

## 6. Confidence & limitations of this report

- Highest-confidence, independently verified or well-established: the ~50–55% directional band and the up-rate baseline; the t>3 hurdle [Harvey–Liu–Zhu 2016]; backtest-overfitting/negative-OOS, DSR/PBO [Bailey–López de Prado]; GKX ~0.4% monthly R² and Sharpe; the microcap/cost erosion [Avramov–Cheng–Metzker 2023]; trees>DL on tabular [Grinsztajn 2022]; DLinear vs transformers [Zeng 2023]; foundation-model finance-negative result [Rahimikia et al. 2025]; the supervised>RL head-to-head; the leakage/reproducibility literature.
- Verified on a second pass (2026-06-13): the Rahimikia et al. "just above 51%" directional figure; the existence and architecture-rank figures of the "918 Experiments" study (arXiv:2603.16886, Saidd 2026); the Rahimikia and Avramov–Cheng–Metzker papers and their qualitative findings.
- Still flagged / reported-not-reverified (precise decimals only — qualitative claims hold): the specific off-the-shelf-TSFM negative-return percentages in Rahimikia et al.; the precise Fischer–Krauss directional %; the exact erosion percentages (62/68/80%) in Avramov–Cheng–Metzker. Note arXiv:2603.16886 covers 4h/24h horizons on crypto/forex/equity indices (a recent preprint), so treat it as corroboration of the ~50% directional finding rather than a daily-equity-specific anchor.
- Method limitation: page-fetching was blocked during research, so figures derive from search extractions plus targeted re-verification of the load-bearing claims; precise decimals should be confirmed against the source PDFs before being quoted verbatim. The *direction* of every conclusion is robust to these specifics and is corroborated across independent sources.

---

## 7. Sources

**Realistic accuracy, base rates, accuracy ≠ edge**
- Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS 33(5):2223–2273 — https://www.nber.org/papers/w25398
- Avramov, Cheng & Metzker (2023), *Machine Learning vs. Economic Restrictions*, Management Science 69(5):2587–2619 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3450322
- Gui (2024), *Machine learning in weekly movement prediction* — https://arxiv.org/abs/2407.09831
- Zhang (2024), *Movement-Prediction-Adjusted Naive Forecast…* — https://arxiv.org/abs/2406.14469
- *Machine learning, stock market forecasting, and market efficiency: a comparative study*, Int. J. Data Science & Analytics (2025) — https://link.springer.com/article/10.1007/s41060-025-00854-4

**Methodology, multiple testing, overfitting, leakage**
- Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns*, RFS 29(1):5–68 — https://www.nber.org/papers/w20592
- Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism*, Notices of the AMS 61(5):458 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
- Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting* — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, JPM — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- López de Prado (2018), *The 10 Reasons Most Machine Learning Funds Fail*, JPM 44(6) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104816
- Kapoor & Narayanan (2023), *Leakage and the reproducibility crisis in ML-based science*, Patterns — https://arxiv.org/abs/2207.07048
- Arnott, Harvey & Markowitz (2019), *A Backtesting Protocol in the Era of Machine Learning*, J. Financial Data Science — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3275654
- Fama (1991), *Efficient Capital Markets: II*, J. Finance — joint-hypothesis problem

**Classical / tabular ML**
- Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS D&B — https://arxiv.org/abs/2207.08815
- Krauss, Do & Huck (2017), *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500*, EJOR 259(2):689–702 — https://www.sciencedirect.com/science/article/abs/pii/S0377221716308657

**Deep learning sequence models**
- Zeng et al. (2023), *Are Transformers Effective for Time Series Forecasting?* (DLinear), AAAI — https://arxiv.org/abs/2205.13504
- Saidd (2026), *A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting: Evidence from 918 Experiments* — https://arxiv.org/abs/2603.16886
- Nie et al. (2023), *A Time Series is Worth 64 Words* (PatchTST), ICLR — https://arxiv.org/abs/2211.14730
- Challu et al. (2023), *N-HiTS*, AAAI — https://arxiv.org/abs/2201.12886
- Lim et al. (2021), *Temporal Fusion Transformers*, Int. J. Forecasting — https://www.sciencedirect.com/science/article/pii/S0169207021000637
- Fischer & Krauss (2018), *Deep Learning with LSTM for Financial Market Predictions*, EJOR 270:654–669 — https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652
- Ghosh, Neufeld & Sahoo (2022), *Forecasting directional movements… LSTM and random forests*, Finance Research Letters — https://arxiv.org/abs/2004.10178
- López Gil et al. (2024), *An Evaluation of DL Models for Stock Market Trend Prediction* (xLSTM-TS; look-ahead denoising red flag) — https://arxiv.org/abs/2408.12408

**Time-series foundation models**
- Rahimikia, Ni & Wang (2025), *Re(Visiting) Time Series Foundation Models in Finance* — https://arxiv.org/abs/2511.18578
- Ansari et al. (2024), *Chronos* — https://arxiv.org/abs/2403.07815
- Das et al. (2024), *TimesFM*, ICML — https://arxiv.org/abs/2310.10688
- Woo et al. (2024), *Moirai*, ICML — https://arxiv.org/abs/2402.02592
- Garza et al. (2023), *TimeGPT-1* — https://arxiv.org/abs/2310.03589
- Rasul et al. (2023), *Lag-Llama* — https://arxiv.org/abs/2310.08278
- Aksu et al. (2024), *GIFT-Eval* — https://arxiv.org/abs/2410.10393
- Xu, Gupta et al. (2025), *Specialized Foundation Models Struggle to Beat Supervised Baselines*, ICLR — https://arxiv.org/abs/2411.02796

**Reinforcement learning**
- Henderson et al. (2018), *Deep Reinforcement Learning That Matters*, AAAI — https://arxiv.org/abs/1709.06560
- Gort et al. (2022), *Deep RL for Cryptocurrency Trading: Addressing Backtest Overfitting* — https://arxiv.org/abs/2209.05559
- Wang et al. (2025), *FinRL Contests* — https://arxiv.org/abs/2504.02281
- Yang, Liu, Zhong & Walid (2020), *Deep RL for Automated Stock Trading: An Ensemble Strategy*, ICAIF — https://openfin.engineering.columbia.edu/sites/default/files/content/publications/ensemble.pdf
- *Outperforming algorithmic trading RL systems: A supervised approach to the cryptocurrency market*, Expert Systems with Applications (2022) — https://www.sciencedirect.com/science/article/abs/pii/S0957417422006339
- Zhang, Zohren & Roberts (2019), *Deep Reinforcement Learning for Trading* — https://arxiv.org/abs/1911.10107
