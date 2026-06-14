# The Mathematics of Analysing Financial Time Series

**Status of this document.** A rigorous, self-contained reference for the mathematics used to analyse time series — built around *this* repository's actual pipeline (volume-profile feature construction, labelling, and the purged-CV + permutation validation harness in `vpts/`), and extended to the full canon so the harness sits in its proper statistical context. It is the methodological companion to [`MARKET_EQUILIBRIUM_MODEL.md`](MARKET_EQUILIBRIUM_MODEL.md): that document asks *what a price means*; this one asks *what can be learned from a sequence of them*. The two share one guardrail, stated at the end of §0 and cashed out in §11.

Every numbered equation is tagged with its logical role:

- **[DEF]** definition / identity (true by construction)
- **[MODEL]** a generative model or assumption (with empirical status: *holds / contested / known false*)
- **[EST]** an estimator or fitting procedure (a function of the sample)
- **[STAT]** a test statistic / inferential object (with its null)
- **[VALID]** a validation, resampling, or multiple-testing-control procedure
- **[REPO]** implemented in this codebase — with a pointer to the file, so the math maps 1:1 to the code

The cardinal discipline, inherited from the companion document: **never present an estimator as if it were the truth it estimates, and never present an in-sample fit as out-of-sample evidence.** A backtest is an estimate with a standard error and a multiple-testing burden; the whole back half of this document is about taking that seriously.

---

## §0 — Conventions, notation, and the standing caveat

### Notation

| Symbol | Meaning |
|---|---|
| $(\Omega,\mathcal F,(\mathcal F_t),\mathbb P)$ | Probability space with a filtration (information available up to $t$) |
| $\{X_t\}_{t\in\mathbb Z}$ | A (real-valued) discrete-time stochastic process / time series |
| $P_t,\ H_t,\ L_t,\ C_t,\ V_t$ | Bar OHLCV: close (also $P_t$), high, low, close, volume at bar $t$ |
| $r_t$ | Return at $t$: simple $R_t=P_t/P_{t-1}-1$ or log $r_t=\ln(P_t/P_{t-1})$ — the doc states which |
| $\mu,\ \sigma^2,\ \gamma(k),\ \rho(k)$ | Mean, variance, autocovariance at lag $k$, autocorrelation at lag $k$ |
| $L$ | Lag (backshift) operator: $L X_t = X_{t-1}$, $L^k X_t = X_{t-k}$ |
| $\varepsilon_t,\ \{\varepsilon_t\}\sim WN(0,\sigma^2)$ | Innovation / white-noise shock |
| $\phi(\cdot),\ \theta(\cdot)$ | AR and MA lag polynomials |
| $f(\lambda)$ | Spectral density at frequency $\lambda$ |
| $\sigma_t^2,\ v_t$ | Conditional variance (GARCH) / spot variance |
| $\hat\theta,\ \theta_0$ | An estimator and the true parameter it targets |
| $\mathbb 1\{\cdot\}$ | Indicator function |
| $\Phi,\ \phi_{\mathcal N}$ | Standard-normal CDF, PDF |
| $\text{IC}$ | Information coefficient (cross-sectional rank correlation of signal vs forward return) |
| $\Pi(\cdot)$ | Volume-at-price distribution (the volume profile) |
| **POC, VAH, VAL** | Point of control (mode of $\Pi$), value-area high/low |
| $\text{ATR}_n$ | Average true range over $n$ bars |
| $\text{CLV}_t$ | Close location value $\in[-1,1]$ |
| $N,\ k,\ \varphi[N,k]$ | CPCV: number of groups, test-groups per split, number of backtest paths |
| $\text{SR},\ \text{DSR},\ \text{PBO}$ | Sharpe ratio, deflated Sharpe ratio, probability of backtest overfitting |

Two glyph notes: $C_t$ is the close, distinct from $C(\cdot)$ a count or combination; $\rho$ is autocorrelation here, distinct from the cross-asset correlation of the companion document.

### The standing caveat (read once, applies everywhere)

Almost every estimator below assumes some form of **stationarity** — that the law generating the data does not move. Financial series violate this: means, variances, and dependence structures drift and break (regime change, microstructure evolution, policy shifts). So every number this document teaches you to compute is *conditional on a stationarity assumption that is known false in general*. The discipline is not to pretend otherwise but to (i) state the assumption, (ii) test it where possible (§5), and (iii) validate out-of-sample under resampling that respects dependence (§10).

**The shared guardrail.** The companion document's lesson — *a price is a risk-weighted valuation, not a probability* — has a time-series twin: **a statistically predictive signal is not the same as a tradeable edge, and an in-sample edge is not the same as an out-of-sample one.** The gap is made of three things this document quantifies: dependence (which inflates naïve significance), multiplicity (trying many things), and non-stationarity (the future is drawn from a different law). This repository's own research log (`RESEARCH.md`) is a case study: an apparent +14.5% backtest that collapses to −0.68% per path under purged CV, and a real out-of-sample correlation that turns out to be survivorship, not signal.

---

## §1 — Foundations

**Time series as a stochastic process.** A time series is one realisation of a process $\{X_t\}$ on $(\Omega,\mathcal F,\mathbb P)$; we see a single path and must infer the law. This is the original sin of the field: cross-sectional statistics average over independent draws, but here we have *one* draw observed through time, and inference is only possible if time-averaging can substitute for ensemble-averaging — i.e. under **ergodicity**.

$$\text{(1.1)}\qquad \frac{1}{T}\sum_{t=1}^{T} g(X_t)\ \xrightarrow{\text{a.s.}}\ \mathbb E[g(X_t)] \quad\text{as } T\to\infty.$$
**[MODEL]** Ergodic theorem (Birkhoff). *Status: assumed, untestable from one path.* Without it, sample means estimate nothing.

**Stationarity.** Strict: the joint law of $(X_{t_1},\dots,X_{t_n})$ is invariant to time shifts. The workable weakening is **covariance (weak) stationarity**:

$$\text{(1.2)}\qquad \mathbb E[X_t]=\mu\ \forall t,\qquad \operatorname{Cov}(X_t,X_{t+k})=\gamma(k)\ \text{depends on } k \text{ only}.$$
**[MODEL]** *Status: contested for returns (roughly holds short-horizon), known false for prices (unit-root non-stationary — §2).*

**Autocovariance / autocorrelation function (ACF).**

$$\text{(1.3)}\qquad \gamma(k)=\operatorname{Cov}(X_t,X_{t+k}),\qquad \rho(k)=\frac{\gamma(k)}{\gamma(0)},\qquad \hat\gamma(k)=\frac1T\sum_{t=1}^{T-k}(X_t-\bar X)(X_{t+k}-\bar X).$$
**[DEF]/[EST]** The sample ACF $\hat\rho(k)$ is the workhorse diagnostic. Under white noise, $\hat\rho(k)\approx\mathcal N(0,1/T)$, giving the $\pm1.96/\sqrt T$ bands. The **partial** ACF (PACF) $\alpha(k)$ is the correlation of $X_t,X_{t+k}$ with intermediate lags projected out — it identifies AR order.

**White noise and the Wold decomposition.** $\{\varepsilon_t\}\sim WN(0,\sigma^2)$ iff mean 0, variance $\sigma^2$, $\gamma(k)=0$ for $k\neq0$. The structural foundation of linear time-series analysis:

$$\text{(1.4)}\qquad X_t=\sum_{j=0}^{\infty}\psi_j\,\varepsilon_{t-j}+\eta_t,\qquad \psi_0=1,\ \sum\psi_j^2<\infty,$$
**[DEF]** Wold (1938): *every* covariance-stationary process is an infinite MA of its own innovations plus a deterministic part $\eta_t$. This is why ARMA models (§2) are not arbitrary — they are parsimonious rational approximations to the Wold representation.

**Returns algebra (the first modelling choice).**

$$\text{(1.5)}\qquad r_t=\ln\frac{P_t}{P_{t-1}}=\ln(1+R_t),\qquad \sum_{t}r_t=\ln\frac{P_T}{P_0}\ \text{(log returns add)},\qquad \prod_t(1+R_t)=\frac{P_T}{P_0}\ \text{(simple returns compound)}.$$
**[DEF]** Log returns are time-additive (convenient for horizon aggregation and Gaussian modelling); simple returns are asset-additive across a portfolio. Confusing them is a units error. *Note: the RESEARCH.md backtest illusion is partly (1.5) — compounding a positive drift over a bull market manufactures a large terminal number from no per-bar edge.*

**The stylized facts (what any model of returns must respect).** Across markets and frequencies, return series exhibit a robust, model-independent set of empirical regularities (Cont 2001, *Quantitative Finance* 1, 223–236):

$$\text{(1.6)}\qquad \rho_r(k)\approx 0\ (k\ge1)\quad\text{but}\quad \rho_{|r|}(k),\ \rho_{r^2}(k)>0\ \text{slowly decaying};\qquad \mathbb P(|r|>x)\sim x^{-\alpha},\ \alpha\in(2,5).$$
**[MODEL]** *Status: holds.* Returns are nearly **uncorrelated** (so the mean is unforecastable — §2) yet **not independent**: absolute/squared returns are strongly autocorrelated (**volatility clustering** — §3). Distributions are **heavy-tailed** (power-law tail index $\alpha\approx3$, finite variance but infinite/large higher moments), **negatively skewed** with a **leverage effect** (down-moves raise future volatility more), and display **aggregational Gaussianity** (the tail thins as the horizon lengthens). Any model that produces i.i.d. Gaussian returns is, by (1.6), known false — and every later section is in some sense a response to one of these facts.

**The random-walk / martingale hypothesis.** The weakest efficient-markets statement is that prices are a martingale, $\mathbb E[P_{t+1}\mid\mathcal F_t]=P_t$ — i.e. returns are unforecastable in mean. Stronger forms (i.i.d. increments, RW1; independent, RW2; uncorrelated, RW3) are nested and separately testable. The canonical test is the **variance ratio** (Lo–MacKinlay 1988, *RFS* 1, 41–66): under a random walk, variance scales linearly with horizon, so

$$\text{(1.7)}\qquad \text{VR}(q)=\frac{\operatorname{Var}(r_t^{(q)})}{q\,\operatorname{Var}(r_t)}=1+2\sum_{k=1}^{q-1}\Big(1-\tfrac{k}{q}\Big)\rho(k)\ \xrightarrow{H_0}\ 1,$$
**[STAT]** where $r_t^{(q)}$ is the $q$-period return. $\text{VR}(q)>1$ indicates positive autocorrelation/trending; $<1$ mean reversion. Lo–MacKinlay reject RW for weekly US index returns — note the tension with §10's finding that this rejection is rarely *tradeable* after costs (the joint-hypothesis problem, §11). The companion document's $\mathbb Q$-martingale (IV.4) is the *risk-neutral* analogue; here the question is whether the **physical** price is a martingale, which it very nearly — but not exactly — is.

---

## §2 — Linear models: ARMA, spectra, unit roots, cointegration, VAR

**ARMA(p,q).** With lag polynomials $\phi(L)=1-\phi_1 L-\dots-\phi_p L^p$ and $\theta(L)=1+\theta_1 L+\dots+\theta_q L^q$:

$$\text{(2.1)}\qquad \phi(L)\,X_t=\theta(L)\,\varepsilon_t.$$
**[MODEL]** Stationary iff the roots of $\phi(z)=0$ lie outside the unit circle; invertible iff $\theta(z)$'s do. *Status: a parsimonious linear approximation; returns are weakly autocorrelated at best, so ARMA captures little of the mean but frames everything else.*

**Estimation and identification.** Yule–Walker equations (method of moments on the ACF) for pure AR; conditional/exact Gaussian MLE for ARMA. Order chosen by the **Box–Jenkins** loop (identify via ACF/PACF → estimate → diagnose residual whiteness via Ljung–Box) and information criteria:

$$\text{(2.2)}\qquad \text{AIC}=-2\ln\hat L+2k,\qquad \text{BIC}=-2\ln\hat L+k\ln T,\qquad Q_{LB}=T(T+2)\sum_{k=1}^{m}\frac{\hat\rho(k)^2}{T-k}.$$
**[EST]/[STAT]** AIC targets predictive loss (does not penalise enough to be consistent); BIC is consistent for the true order if it is finite; $Q_{LB}\sim\chi^2_{m-p-q}$ under correct specification (Ljung–Box 1978, *Biometrika*).

**Spectral analysis.** The frequency-domain dual of the ACF:

$$\text{(2.3)}\qquad f(\lambda)=\frac{1}{2\pi}\sum_{k=-\infty}^{\infty}\gamma(k)e^{-ik\lambda},\qquad \gamma(k)=\int_{-\pi}^{\pi}f(\lambda)e^{ik\lambda}\,d\lambda,\qquad I_T(\lambda)=\frac{1}{2\pi T}\Big|\sum_{t=1}^{T}X_t e^{-it\lambda}\Big|^2.$$
**[DEF]/[EST]** $f$ is the spectral density (Fourier transform of the ACF — Wiener–Khinchin); the **periodogram** $I_T$ is its raw, inconsistent estimator (variance does not vanish; must be smoothed). Power-law $f(\lambda)\sim\lambda^{-2d}$ near 0 signals long memory (§3).

**Unit roots and integration.** A series is $I(1)$ if its first difference is stationary. The **augmented Dickey–Fuller** regression tests $H_0:$ unit root ($\beta=0$):

$$\text{(2.4)}\qquad \Delta X_t=\alpha+\delta t+\beta X_{t-1}+\sum_{i=1}^{p}\zeta_i\,\Delta X_{t-i}+\varepsilon_t,\qquad \text{ADF}=\frac{\hat\beta}{\operatorname{se}(\hat\beta)}\ \text{(non-standard, Dickey–Fuller dist.)}.$$
**[STAT]** Dickey–Fuller (1979, *JASA*); Phillips–Perron (1988, *Biometrika*) is the HAC-robust variant. **KPSS** (Kwiatkowski–Phillips–Schmidt–Shin 1992, *J. Econometrics*) flips the null to *stationarity* — running both is the honest practice, since each has low power against the other's alternative. *Prices are typically $I(1)$, returns $I(0)$ — which is why one models returns.*

**Cointegration and error correction.** Two $I(1)$ series can share a stationary linear combination ($I(0)$): they are tied by a long-run equilibrium even as each wanders.

$$\text{(2.5)}\qquad y_t-\beta x_t=u_t\sim I(0)\ \Rightarrow\ \Delta y_t=\alpha\,(y_{t-1}-\beta x_{t-1})+\dots+\varepsilon_t,\quad \alpha<0.$$
**[MODEL]/[EST]** Engle–Granger (1987, *Econometrica*) two-step; Johansen (1991, *Econometrica*) the system MLE / VECM with rank tests for the number of cointegrating vectors. This is the statistical engine of pairs/relative-value trading — and note the link to the companion document's Δ2: a cointegrating residual is a *convergence trade*, and §III's limits-to-arbitrage say it can diverge before it converges.

**Vector autoregression, Granger causality, impulse responses.**

$$\text{(2.6)}\qquad \mathbf X_t=\mathbf c+\sum_{i=1}^{p}\mathbf A_i\mathbf X_{t-i}+\boldsymbol\varepsilon_t,\qquad \mathbf X_t=\boldsymbol\mu+\sum_{j\ge0}\boldsymbol\Psi_j\boldsymbol\varepsilon_{t-j}\ \text{(MA}\!\to\!\text{IRF)}.$$
**[MODEL]** Sims (1980, *Econometrica*). $x$ *Granger-causes* $y$ if past $x$ improves the forecast of $y$ given past $y$ (Granger 1969, *Econometrica*) — a statement about predictability, **not** structural causation. Impulse-response and forecast-error-variance decomposition require an identification scheme (Cholesky ordering, etc.) to map reduced-form $\boldsymbol\varepsilon$ to structural shocks; the choice is an assumption, not data.

---

## §3 — Second moments: volatility, jumps, long memory

The mean is nearly unpredictable; the *variance* is highly predictable (volatility clusters). This is where most of the structure in financial series actually lives.

**ARCH / GARCH.** Engle (1982, *Econometrica*) and Bollerslev (1986, *J. Econometrics*):

$$\text{(3.1)}\qquad r_t=\mu+\varepsilon_t,\quad \varepsilon_t=\sigma_t z_t,\ z_t\sim(0,1),\qquad \sigma_t^2=\omega+\sum_{i=1}^{q}\alpha_i\varepsilon_{t-i}^2+\sum_{j=1}^{p}\beta_j\sigma_{t-j}^2.$$
**[MODEL]** GARCH(1,1) ($\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2$) is the workhorse; covariance-stationary iff $\alpha+\beta<1$, with unconditional variance $\omega/(1-\alpha-\beta)$ and persistence $\alpha+\beta$ (often $\approx0.99$ daily). *Status: captures clustering — holds; assumes a symmetric response — known false.* Extensions encode the **leverage effect** (down-moves raise vol more): EGARCH (Nelson 1991, *Econometrica*, log-variance with sign asymmetry) and GJR-GARCH (Glosten–Jagannathan–Runkle 1993, *JF*, a threshold term $\gamma\varepsilon_{t-1}^2\mathbb 1\{\varepsilon_{t-1}<0\}$). Estimated by **QMLE** (Gaussian quasi-likelihood is consistent even if $z_t$ is non-Gaussian — §9). This is the discrete-time image of the companion document's stochastic-variance state $v_t$ (III.2).

**Realised variance and jumps.** With intraday returns $r_{t,i}$, $i=1..M$:

$$\text{(3.2)}\qquad \text{RV}_t=\sum_{i=1}^{M}r_{t,i}^2\ \xrightarrow{p}\ \int_{t-1}^{t}v_s\,ds+\sum_{\text{jumps}}J^2,\qquad \text{BV}_t=\frac{\pi}{2}\sum_{i=2}^{M}|r_{t,i-1}||r_{t,i}|\ \xrightarrow{p}\ \int_{t-1}^{t}v_s\,ds.$$
**[EST]** Realised variance converges to total quadratic variation; **bipower variation** (Barndorff-Nielsen–Shephard 2004, *J. Financial Econometrics*) is jump-robust, so $\text{RV}-\text{BV}$ estimates the **jump** contribution — the empirical handle on the companion document's jump measure $\ell(dz)$ (III.1). *Daily OHLC bars (this repo's data) cannot form RV; the repo proxies volatility with ATR — see §6.*

**Long memory.** When the ACF decays hyperbolically ($\rho(k)\sim k^{2d-1}$, $0<d<\tfrac12$) rather than geometrically, shocks persist far longer than ARMA allows.

$$\text{(3.3)}\qquad (1-L)^d X_t=\varepsilon_t\ \text{(ARFIMA)},\qquad \mathbb E\Big[\tfrac{R(n)}{S(n)}\Big]\sim n^{H},\qquad H=d+\tfrac12.$$
**[MODEL]/[EST]** Fractional integration (Granger–Joyeux; Hosking 1981, *Biometrika*); the **Hurst exponent** $H$ estimated by rescaled-range (R/S) analysis (Hurst 1951; Mandelbrot–Wallis) or detrended fluctuation analysis (DFA, Peng et al. 1994). $H=\tfrac12$ is a random walk, $H>\tfrac12$ trending/persistent, $H<\tfrac12$ mean-reverting. *Status: contested — R/S is biased in small samples and confounded by short-memory and non-stationarity; treat $H$ estimates as suggestive, not decisive.*

**Forecasting volatility: the HAR model.** The dominant practical volatility forecaster sidesteps fractional integration by regressing realised variance on its own averages over a *cascade* of horizons (daily, weekly, monthly), approximating long memory with three lags (Corsi 2009, *J. Financial Econometrics* 7, 174–196):

$$\text{(3.4)}\qquad \text{RV}_{t+1}=\beta_0+\beta_d\,\text{RV}_t^{(d)}+\beta_w\,\text{RV}_t^{(w)}+\beta_m\,\text{RV}_t^{(m)}+\varepsilon_{t+1},$$
**[MODEL]/[EST]** where $\text{RV}^{(w)},\text{RV}^{(m)}$ are 5- and 22-day averages. The **HAR-RV** is a plain OLS regression yet forecasts as well as far more complex long-memory models — a recurring lesson (echoed in this repo's experiments 4 and 9) that *parsimony beats sophistication* out-of-sample. *On daily OHLC bars one substitutes a range estimator (Parkinson/Garman–Klass) or ATR (§6) for RV.*

---

## §4 — State space, filtering, and regimes

**State-space form** unifies a huge class of models (ARMA, time-varying parameters, stochastic vol, unobserved trends):

$$\text{(4.1)}\qquad \underbrace{\mathbf x_t=\mathbf F\mathbf x_{t-1}+\mathbf w_t}_{\text{state (transition)}},\qquad \underbrace{y_t=\mathbf H\mathbf x_t+v_t}_{\text{observation}},\qquad \mathbf w_t\sim\mathcal N(0,\mathbf Q),\ v_t\sim\mathcal N(0,R).$$
**[MODEL]** The hidden state $\mathbf x_t$ is observed only through noisy $y_t$.

**Kalman filter.** The optimal (MMSE, and under Gaussianity the exact Bayesian) recursive estimator of $\mathbf x_t$ given $y_{1:t}$ — it is the dynamic version of the projection theorem (Lemma P in the companion document). Predict then update:

$$\text{(4.2)}\qquad
\begin{aligned}
&\hat{\mathbf x}_{t|t-1}=\mathbf F\hat{\mathbf x}_{t-1},\quad \mathbf P_{t|t-1}=\mathbf F\mathbf P_{t-1}\mathbf F'+\mathbf Q,\\
&\mathbf K_t=\mathbf P_{t|t-1}\mathbf H'(\mathbf H\mathbf P_{t|t-1}\mathbf H'+R)^{-1},\\
&\hat{\mathbf x}_t=\hat{\mathbf x}_{t|t-1}+\mathbf K_t(y_t-\mathbf H\hat{\mathbf x}_{t|t-1}),\quad \mathbf P_t=(\mathbf I-\mathbf K_t\mathbf H)\mathbf P_{t|t-1}.
\end{aligned}$$
**[EST]** Kalman (1960). The gain $\mathbf K_t$ is the precision-weighting of new information against the prior — exactly the normal–normal update (companion Lemma N) made recursive. The innovations $y_t-\mathbf H\hat{\mathbf x}_{t|t-1}$ are a white-noise sequence whose Gaussian likelihood gives the prediction-error decomposition for MLE; the RTS smoother runs it backward for $\mathbf x_t\mid y_{1:T}$. Nonlinear/non-Gaussian extensions: extended/unscented Kalman, and **particle filters** (sequential Monte Carlo) for general state spaces.

**Regime-switching / hidden Markov models.** A discrete latent state $S_t\in\{1,\dots,m\}$ following a Markov chain with transition matrix $\mathbf P=[p_{ij}]$ governs the parameters:

$$\text{(4.3)}\qquad y_t\mid S_t=j\ \sim\ \mathcal N(\mu_j,\sigma_j^2),\qquad \mathbb P(S_t=j\mid S_{t-1}=i)=p_{ij}.$$
**[MODEL]** Hamilton (1989, *Econometrica*) for switching regressions; the latent path is inferred by the forward–backward (Baum–Welch, an EM algorithm) for the likelihood and **Viterbi** for the most-likely state sequence. This is the formal version of "the market is in a different regime" — and a sober reminder (companion §Boundary) that a *known* switching law is still one stationary meta-model. **[REPO]** `vpts/regime/quiet.py` implements a simpler, non-probabilistic regime read: a "quiet-phase" score from volatility/volume compression rather than a fitted HMM — a hand-built detector occupying the same conceptual slot.

---

## §5 — Non-stationarity and nonlinearity: breaks, trends, filters, regimes-in-the-mean

Because the standing caveat is real, detecting and handling parameter change — and nonlinearity — is not optional.

**Structural breaks.** Chow (1960) tests a break at a *known* date; **CUSUM** (Brown–Durbin–Evans 1975, *JRSS-B*) monitors cumulative recursive residuals for parameter drift; **Bai–Perron** (1998, *Econometrica*; 2003, *J. Applied Econometrics*) estimates *multiple* breaks at *unknown* dates by minimising total SSR with a dynamic program. **[STAT]** A break detected is the stationarity assumption failing in-sample — the honest response is to model the change, not to extend the window through it.

**Trend/cycle decomposition.** The Hodrick–Prescott filter (1997, *JMCB*) splits $y_t=\tau_t+c_t$ by penalised least squares:

$$\text{(5.1)}\qquad \min_{\{\tau_t\}}\ \sum_t (y_t-\tau_t)^2+\lambda\sum_t\big[(\tau_{t+1}-\tau_t)-(\tau_t-\tau_{t-1})\big]^2.$$
**[EST]** $\lambda$ trades fit against smoothness (1600 for quarterly data, by convention). *Known issues: spurious end-point cycles and the look-ahead of a two-sided filter — using HP-filtered features in a backtest is a classic leakage bug (§10).* Band-pass alternatives: Baxter–King; multiresolution alternatives: the **wavelet** transform, which localises variance jointly in time and frequency (better than the global Fourier spectrum for non-stationary series). Change-point detection (CUSUM, Bayesian online change-point, PELT) is the online cousin.

**Nonlinear models in the mean.** Linear ARMA cannot capture state-dependent dynamics (trending in one regime, mean-reverting in another). The threshold/regime family makes the parameters depend on an observable or its own lag:

$$\text{(5.2)}\qquad X_t=\begin{cases}\phi^{(1)}(L)X_t+\varepsilon_t,& q_{t-d}\le c\\[2pt]\phi^{(2)}(L)X_t+\varepsilon_t,& q_{t-d}> c\end{cases}\quad\text{(SETAR)},\qquad X_t=\phi^{(1)}X_{t-1}+\big(\phi^{(2)}-\phi^{(1)}\big)X_{t-1}\,G(q_{t-d};\gamma,c)+\varepsilon_t\ \text{(STAR)}.$$
**[MODEL]** Self-exciting threshold AR (Tong 1990, *Non-Linear Time Series*) switches sharply at threshold $c$; smooth-transition AR (Teräsvirta 1994, *JASA*) interpolates via a logistic/exponential $G\in[0,1]$. These are the *in-the-mean* cousins of the Hamilton regime-switching model (§4, which switches on a *latent* state) and of this repo's regime gating (§6). *Status: flexible but easy to overfit — the bias–variance warning of §9 applies sharply.*

**Testing for nonlinearity / i.i.d.** The **BDS test** (Brock–Dechert–Scheinkman–LeBaron 1996, *Econometric Reviews* 15) uses the correlation integral to test the null that a series is i.i.d. against unspecified (possibly nonlinear/chaotic) dependence — typically applied to model *residuals* to check whether linear filtering left structure behind. **[STAT]** Rejection says "there is more here," not "it is tradeable."

---

## §5b — Living with non-stationarity: what to do when the law moves

The standing caveat (§0) is not a disclaimer to be filed and forgotten — it is the central practical problem, and it has a real, if partial, constructive answer. This section develops it at length, separating what *can* be handled from what *cannot*. The honest position is neither "assume stationarity" nor "give up," but a graded discipline.

**The reframing that makes the problem tractable.** You do not need *the market* to be stationary; you need the *specific relationship you exploit* to be stable over your trading horizon — a different, answerable question. And a relationship's stability is inherited from its **source**:

$$\text{(5.3)}\qquad \underbrace{\text{statistical pattern}}_{\text{least stationary}} \;\prec\; \text{risk premium} \;\prec\; \underbrace{\text{structural constraint}}_{\text{most stationary}}.$$
**[PRINCIPLE]** A bare statistical regularity ("$X$ predicted $Y$ in-sample") has no reason to persist and decays fastest; a risk premium persists as long as the risk and the aversion to it persist; a structural/institutional constraint — an index fund that must buy regardless of price, a hedger who must roll, a regulation that forces a flow — persists as long as the institution does. **The same structural counterparty that makes an edge *survive* (§EDGE_METHODOLOGY.5) is what makes it *stationary*.** So the first response to non-stationarity is not statistical at all: anchor the edge in durable structure, not in a pattern.

**The adaptive-estimation tradeoff (handling slow drift).** When the true parameter drifts slowly, re-estimate on a trailing window — but the window length is itself an optimization. With drift rate $\delta$ (parameter change per period) and per-observation noise $\sigma^2$, a rolling window of length $w$ has

$$\text{(5.4)}\qquad \mathrm{MSE}(w)\;\approx\;\underbrace{\frac{\sigma^2}{w}}_{\text{estimation variance}}\;+\;\underbrace{c\,\delta^2 w^2}_{\text{staleness bias}},\qquad\Longrightarrow\qquad \text{(5.5)}\quad w^\star \;\propto\; \Big(\frac{\sigma^2}{\delta^2}\Big)^{1/3}.$$
**[EST]** The window optimisation is the formal statement of "re-fit fast enough to track the drift, slow enough to average out the noise" — the bias–variance tradeoff of §9b, now along the **time** axis: a faster-adapting model tracks the regime but is noisier. The exponentially-weighted estimator $\hat\sigma^2_t=\lambda\hat\sigma^2_{t-1}+(1-\lambda)r_t^2$ is the smooth version, with effective window $1/(1-\lambda)$ (RiskMetrics). The **time-varying-parameter** model — let $\theta_t=\theta_{t-1}+\eta_t$ follow a random walk and run the Kalman filter (§4) — is the optimal linear adaptive estimator under that drift model. **[MODEL/EST]**

**Regime-switching, and why it does not escape the problem.** Hamilton's model (§4) treats non-stationarity as switching among a *finite set of stationary regimes* with a fixed transition law. Where regimes recur (calm/turbulent volatility states), it is the right tool. But a *known* switching law is only a larger *stationary meta-model* — it relocates the stationarity assumption one level up, to the transition matrix. It handles *recurrent* change; it cannot handle a *genuinely novel* regime, because the new regime is not in the state space. **[MODEL]**

**The robust / no-regret paradigm (giving up the law entirely).** A different philosophy abandons the attempt to estimate the data-generating law at all:

- **Distributionally robust optimisation.** Instead of minimising expected loss under a single estimated $\mathbb P$, minimise the *worst case* over an ambiguity set $\mathcal U$ of laws around the empirical distribution:
$$\text{(5.6)}\qquad \min_{a}\;\sup_{\mathbb Q\in\mathcal U}\;\mathbb E_{\mathbb Q}\big[\ell(a)\big].$$
**[MODEL]** (Hansen–Sargent robust control; Wasserstein-DRO.) You trade a little average performance for stability across a neighbourhood of laws — explicitly buying insurance against having the wrong $\mathbb P$.

- **Online / no-regret learning.** Drop the i.i.d. assumption entirely: treat the data as an *arbitrary, even adversarial* sequence. Online gradient descent attains **regret** — cumulative loss minus that of the best single decision in hindsight — bounded by
$$\text{(5.7)}\qquad \mathrm{Regret}_T \;\le\; O(\sqrt T)\quad\Longrightarrow\quad \frac{\mathrm{Regret}_T}{T}\to 0,$$
**[STAT]** for *any* sequence of bounded convex losses, with **no stationarity assumed anywhere** (Zinkevich 2003, *ICML*). *The catch, made precise:* the comparator is the best *fixed* decision in hindsight, which is itself poor if the world changed a lot. Against a *moving* comparator (dynamic regret), the bound holds only if the comparator's total variation is **budgeted** — you can track change that is *limited in total*, not arbitrary change (Besbes–Gur–Zeevi 2015, *Operations Research* 63(5), "variation budget"). This is the exact mathematical boundary of adaptation: **bounded non-stationarity is learnable; unbounded non-stationarity is not.**

**A taxonomy of non-stationarity by tractability.** Putting it together:

| Kind | Example | Handled by | Tractable? |
|---|---|---|---|
| Slowly-varying parameters | drifting volatility, slow beta change | adaptive estimation (5.4–5.5), Kalman TVP | **yes** |
| Recurrent regimes | calm/turbulent vol, risk-on/off | regime-switching (§4), robust opt. (5.6) | **mostly** |
| Bounded-variation change | gradual structural evolution | no-regret w/ variation budget (5.7) | **partly** |
| Genuine structural break / Knightian novelty | a regime absent from the *entire* sample | **nothing** | **no** |
| Reflexive self-decay | your own trading erodes the signal | shorten horizon, expect decay | **no (self-induced)** |

**[PRINCIPLE / boundary]**

**The hard limit, made precise.** The last two rows are the edge of the map, and no method removes them:
- **The Lucas critique** (Lucas 1976). If agents' decision rules depend on the policy/structural regime, then parameters estimated under one regime are *not invariant* to a change of regime — they are not "structural," and forecasts under a new regime are unfounded. "Estimate on history, deploy in the future" silently assumes the future regime equals the past. A break whose *possibility was not in the support* of your model is not risk you can hedge; it is Knightian uncertainty (§11).
- **Reflexivity** (§11). Exploiting a relationship changes it — a non-stationarity you *cause*, which no external change-point detector catches, because you *are* the regime change.

**The operational upshot.** Non-stationarity does not counsel despair; it dictates a discipline: (i) prefer *structural* edges over *statistical* ones — they are more stationary by construction (5.3); (ii) re-estimate adaptively, sized to the drift timescale (5.5), and monitor with change-point detection; (iii) where you cannot trust a single law, use robust/online methods (5.6–5.7) and accept lower average return for stability; (iv) size for the regime change that *will* come — cap leverage, and stress-test not only against the regimes in your sample but against a plausible one outside it; (v) read every backtest as conditional on "no regime change unlike the sample," and never stake the firm on that condition holding.

---

## §6 — The signal math this repository computes

This section documents `vpts`'s feature construction as time-series mathematics. All of it operates **without look-ahead** (features at bar $t$ use data $\le t$), which is itself the most important modelling constraint (§10).

**Average true range (volatility scale).** **[REPO]** `vpts/regime/indicators.py`, `profile/calculator.py`:

$$\text{(6.1)}\qquad \text{TR}_t=\max\big(H_t-L_t,\ |H_t-C_{t-1}|,\ |L_t-C_{t-1}|\big),\qquad \text{ATR}_n=\frac1n\sum_{i=t-n+1}^{t}\text{TR}_i.$$
**[EST]** A robust, OHLC-only volatility proxy (Wilder 1978). It scales bin widths, level tolerances, and — crucially — the barrier distances in labelling (§7), so that targets are defined in volatility units, not price units.

**The volume profile as a price-conditional volume density.** **[REPO]** `vpts/profile/calculator.py`. Distribute each bar's volume across price bins in proportion to the overlap of $[L_t,H_t]$ with each bin (the volume-conserving "uniform" estimator), giving an empirical *volume-at-price* distribution $\Pi$:

$$\text{(6.2)}\qquad \Pi(b)=\sum_t V_t\cdot\frac{\big|[L_t,H_t]\cap \text{bin}_b\big|}{H_t-L_t},\qquad \text{POC}=\arg\max_b \Pi(b).$$
**[DEF]/[EST]** This is a kernel-smoothed histogram (a KDE with a uniform/box kernel of bar-dependent width) of traded volume over price. The **POC** is its mode; HVN/LVN are its peaks/valleys (found by `scipy.signal.find_peaks` on a Gaussian-smoothed $\Pi$ — equivalently a mode/antimode hunt). The **value area** is built by the Market-Profile expansion rule — start at the POC and greedily annex the heavier of the two adjacent pairs until 70% of volume is enclosed:

$$\text{(6.3)}\qquad \text{VA}=[\text{VAL},\text{VAH}]\ \text{minimal-ish interval s.t. }\sum_{b\in\text{VA}}\Pi(b)\ge 0.70\sum_b\Pi(b).$$
**[DEF]** Statistically this is a (greedy, not exact) **70% highest-density region** of $\Pi$ — the volume analogue of a credible interval around the modal price. ATR-adaptive binning (quarter-ATR bins) makes resolution regime-dependent: finer in quiet markets, coarser in volatile ones.

**Close location value & synthetic order-flow delta.** **[REPO]** `vpts/structure/analytics.py`. With no tick data, infer aggressor side from where a bar closes in its range:

$$\text{(6.4)}\qquad \text{CLV}_t=\frac{(C_t-L_t)-(H_t-C_t)}{H_t-L_t}\in[-1,1],\qquad \delta_t=V_t\cdot\text{CLV}_t,\qquad \delta^{\text{net}}=\frac{\sum_t\delta_t}{\sum_t V_t}.$$
**[EST]** $\delta_t$ is a signed-volume **synthetic cumulative volume delta** — an OHLC estimator of the buy/sell imbalance that the companion document's microstructure block (order flow $y$, Kyle/GM) models with real signed trades. The *POC-delta fraction* (signed volume at the modal price) is the repo's "buy-the-dip / passive accumulation" tell. It is an *estimate* of order flow, with all the bias that an OHLC proxy carries — flagged as such.

**Volume-weighted shape moments.** **[REPO]** Treating $\Pi$ as a distribution with $p_b=\Pi(b)/\sum\Pi$:

$$\text{(6.5)}\qquad m=\sum_b p_b x_b,\quad s^2=\sum_b p_b(x_b-m)^2,\quad \text{skew}=\sum_b p_b\Big(\tfrac{x_b-m}{s}\Big)^3,\quad \text{kurt}=\sum_b p_b\Big(\tfrac{x_b-m}{s}\Big)^4.$$
**[DEF]** Standardised 3rd/4th moments of the volume-at-price distribution — the same moment machinery the companion document reads off the *risk-neutral* density (BKM, IV.L3.vi), here applied to the realised volume density to classify profile shape (P / b / D / B).

**Time-decayed gravity (cost-basis migration).** **[REPO]** An exponentially-weighted profile with half-life $h$ reveals whether recent trade is migrating to a new fair value:

$$\text{(6.6)}\qquad \Pi^{\text{decay}}(b)=\sum_t V_t\, 2^{-\text{age}(t)/h}\cdot(\text{overlap}_b),\qquad \text{decayed POC}=\arg\max_b\Pi^{\text{decay}}(b).$$
**[EST]** The $2^{-\text{age}/h}$ kernel is an EMA; the gap between the decayed and lifetime POC is a slow drift estimator (cf. the EMA as a one-sided low-pass filter — a causal cousin of the HP trend in §5, but look-ahead-free).

**Rolling standardisation (the z-score trigger).** **[REPO]** The value-area-compression ratio $\text{VACR}=(\text{VAH}-\text{VAL})/P$ is turned into a stationary signal by a **rolling z-score**:

$$\text{(6.7)}\qquad z_t=\frac{\text{VACR}_t-\hat\mu_{t}^{(w)}}{\hat\sigma_{t}^{(w)}},\qquad \hat\mu_t^{(w)},\hat\sigma_t^{(w)}\ \text{computed on } [t-w+1,\,t]\ \text{only}.$$
**[EST]** Rolling standardisation is the practical concession to non-stationarity: it removes a slowly-varying local mean/scale so a level becomes comparable across regimes. The trailing window is what keeps it causal.

**The confluence score (linear fusion).** **[REPO]** `vpts/scoring/scorer.py`. The hand-set strategy is a weighted linear combination of component strengths $g_c\in[0,1]$ with signs $d_c\in\{-1,0,1\}$:

$$\text{(6.8)}\qquad \text{quality}=100\cdot\frac{\sum_c w_c g_c}{\sum_c w_c}\in[0,100],\qquad \text{bias}=100\cdot\frac{\sum_c w_c g_c d_c}{\sum_c w_c}\in[-100,100].$$
**[DEF]** A fixed-weight linear scorecard — the "primitive/assumption" layer of the strategy. Experiment 2 in `RESEARCH.md` *learns* these $w_c$ by ridge regression (§8) and finds no out-of-sample improvement over the hand weights: the linear functional form, not the weights, is the binding object.

---

## §7 — Labelling: defining the target as a first-passage problem

A predictor needs something to predict. The repo uses **triple-barrier labelling** (López de Prado, *AFML* 2018, ch. 3) — a first-passage formulation that is far more honest than fixed-horizon returns because it encodes a path-dependent, risk-defined exit. **[REPO]** `vpts/ml/labeling.py`.

For an event at $t$ taken in side $s\in\{+1,-1\}$, with volatility-scaled barriers (profit $u=\text{pt}\cdot\sigma_t$, stop $\ell=\text{sl}\cdot\sigma_t$, vertical $T$ bars):

$$\text{(7.1)}\qquad \tau=\inf\Big\{j>0:\ s\,r_{t,t+j}\ge u\ \ \text{or}\ \ s\,r_{t,t+j}\le-\ell\ \ \text{or}\ \ j=T\Big\},\qquad y_t=\operatorname{sign}\big(s\,r_{t,t+\tau}\big).$$
**[DEF]** The label is determined by which barrier the (volatility-normalised) path hits first — a **first-passage / barrier-hitting time** of the price process, exactly the mathematics the companion document uses for option exercise and the Kyle revelation horizon. Scaling barriers by $\sigma_t$ (here ATR/close) makes the label stationary across volatility regimes — a target with constant difficulty. The breakeven win-rate is set by the reward:risk $u/\ell$ (2:1 → 33%).

**Meta-labelling.** A *primary* model picks the side $s=\operatorname{sign}(\text{bias})$; the triple barrier yields a binary **meta-label** $\mathbb 1\{\text{the bet won}\}$; a *secondary* model learns $\mathbb P(\text{win}\mid \text{features})$ — i.e. it learns *whether to act*, not *which way*. **[VALID-adjacent]** This cleanly separates **direction** from **selectivity**, and RESEARCH.md exploits exactly that split: the directional call is survivorship-driven and inverts, but the *selectivity* lift is the most survivorship-resilient signal in the whole study (it degrades rather than flips). **MFE/MAE** labelling is the continuous cousin — label by whether maximum favourable excursion beat maximum adverse excursion, a volatility-scaled triple barrier in disguise.

---

## §8 — Dependence, the cross-section, and the information coefficient

**Cross-correlation and lead–lag.** $\rho_{xy}(k)=\operatorname{Corr}(x_t,y_{t+k})$; a peak at $k>0$ says $x$ leads $y$ — again predictability, not causation (§2's Granger caveat).

**The information coefficient.** The central performance statistic for a cross-sectional signal: the rank correlation between today's signal and the forward return, across names, each rebalance date. **[REPO]** `vpts/ml`, `vpts/scoring`:

$$\text{(8.1)}\qquad \text{IC}_d=\rho_{\text{Spearman}}\big(\text{signal}_{i,d},\ r^{\text{fwd}}_{i,d}\big)_{i=1..N_d},\qquad t_{\text{IC}}=\overline{\text{IC}}\,\frac{\sqrt{D}}{\sigma_{\text{IC}}},\qquad \operatorname{se}(\text{IC}_d)\approx\frac{1}{\sqrt{N_d-1}}.$$
**[STAT]** Spearman (rank) IC is robust to the non-normal, outlier-heavy nature of returns. Two facts the repo leans on hard: (i) per-date IC noise scales like $1/\sqrt{N_d}$, so a thin cross-section ($N\approx20$) produces a huge $\sigma_{\text{IC}}\approx0.28$ and spurious near-misses — *width* (more names) is the decisive test, and the 20-name "+0.021, p=0.10" washes to "−0.009, p=0.86" at 88 names (experiments 5→6); (ii) the single-feature t-stat $t_{\text{IC}}$ understates the multiple-testing burden of having tried many features (§10).

**Factor models and dimension reduction.** Cross-sectional regressions $r_i=\boldsymbol\beta_i'\mathbf f+\epsilon_i$; **PCA** (eigendecomposition of the return covariance) extracts statistical factors; the companion document's $\mathbb Q$/$\mathbb P$ point applies — a "factor premium" is compensation for risk, not free money. **Copulas** (Sklar's theorem: any joint law factors into marginals plus a copula $C$) model tail dependence beyond linear $\rho$; **DCC-GARCH** (Engle 2002, *JBES*) makes the correlation matrix itself time-varying — the multivariate image of §3.

**Nonlinear dependence (information-theoretic).** Correlation and rank-IC miss nonlinear/asymmetric dependence (the stylized fact (1.6) that returns are uncorrelated but *not* independent). Information theory measures the full dependence:

$$\text{(8.2)}\qquad I(X;Y)=\sum_{x,y}p(x,y)\ln\frac{p(x,y)}{p(x)p(y)}\ \ge 0,\qquad T_{Y\to X}=\sum p(x_{t+1},x_t^{(\cdot)},y_t^{(\cdot)})\ln\frac{p(x_{t+1}\mid x_t^{(\cdot)},y_t^{(\cdot)})}{p(x_{t+1}\mid x_t^{(\cdot)})}.$$
**[STAT]** **Mutual information** $I(X;Y)$ is zero iff $X\perp Y$ (it catches any dependence, not just linear); **transfer entropy** $T_{Y\to X}$ (Schreiber 2000, *Phys. Rev. Lett.* 85) is its directed, lagged version — a model-free, nonlinear generalisation of Granger causality (§2). *Caveat: both need a lot of data to estimate densities and are easy to over-read in finite, noisy samples — the multiple-testing and overfitting cautions of §§9–10 apply.*

---

## §9 — Estimation and inference machinery

**Maximum likelihood / QMLE.** Most models above are fit by maximising the (Gaussian prediction-error) likelihood; under misspecified innovations the Gaussian **quasi-**MLE remains consistent and asymptotically normal with a sandwich covariance (Bollerslev–Wooldridge), which is why GARCH estimates survive fat-tailed $z_t$. **[EST]**

**GMM.** When a model implies moment conditions $\mathbb E[g(X_t,\theta_0)]=0$ (e.g. the Euler equation $\mathbb E[mR-1]=0$ of the companion document), estimate by minimising $\bar g(\theta)'\mathbf W\bar g(\theta)$ (Hansen 1982, *Econometrica*). **[EST]** The optimal $\mathbf W$ is the inverse long-run variance — which requires:

**HAC standard errors.** Serial correlation and heteroskedasticity make naïve standard errors wrong (usually too small). The Newey–West (1987, *Econometrica*) estimator corrects them:

$$\text{(9.1)}\qquad \hat S=\hat\Gamma_0+\sum_{k=1}^{m}\Big(1-\frac{k}{m+1}\Big)\big(\hat\Gamma_k+\hat\Gamma_k'\big).$$
**[EST]** The Bartlett weights guarantee a positive-definite estimate; $m$ is the bandwidth. *This is the single most common fix for the "my t-stat is inflated by autocorrelation" problem — the time-series analogue of clustering.*

**The bootstrap (resampling under dependence).** I.i.d. bootstrap is invalid for dependent data — it destroys the autocorrelation. Two repairs:

$$\text{(9.2)}\qquad \textbf{block bootstrap: }\text{resample contiguous blocks of length }b;\qquad \textbf{stationary bootstrap: }\text{geometric block length }L\sim\text{Geom}(p).$$
**[VALID]** The stationary bootstrap (Politis–Romano 1994, *JASA*) randomises block length so the resampled series is itself stationary — the standard tool for confidence intervals on Sharpe ratios, IC, and other path statistics. Block length $b\sim T^{1/3}$ (or $1/p$) must grow with the dependence horizon.

**Permutation / randomisation tests.** **[REPO]** `vpts/ml` — the decisive significance test in the whole research log. Destroy the feature→label link by shuffling labels (per-row for a single series; **within-date** for a cross-section, to preserve cross-sectional structure), recompute the statistic $B$ times, and read off:

$$\text{(9.3)}\qquad p=\frac{1+\#\{b:\ \hat\theta^{(b)}_{\text{shuffled}}\ \ge\ \hat\theta_{\text{observed}}\}}{1+B}.$$
**[VALID]** A model-free null that asks "could structure-free noise have produced a statistic this large?" The $+1$s make it a valid finite-sample test. *An effect that cannot clear its own shuffled null is reported as no edge* — this is what kills meta-labelling (p 0.005 on survivors → 0.80 with delisted names injected).

### §9b — Regularised regression, classification, and learning machines

The fitted models in this repo (and modern empirical finance generally) are penalised regressions and tree ensembles. Their math, and the one decomposition that explains when they fail.

**Ridge regression.** **[REPO]** `vpts/ml/factor_model.py` learns the confluence weights by exactly this closed form (train-only standardisation, target centred):

$$\text{(9.4)}\qquad \hat{\mathbf w}=\arg\min_{\mathbf w}\ \|\mathbf y-\mathbf X\mathbf w\|^2+\alpha\|\mathbf w\|_2^2=(\mathbf X'\mathbf X+\alpha\mathbf I)^{-1}\mathbf X'\mathbf y.$$
**[EST]** Hoerl–Kennard (1970, *Technometrics* 12). The $\ell_2$ penalty $\alpha$ shrinks weights toward 0, trading a little bias for a large variance reduction — essential when features are collinear (confluence factors are) and the signal is faint. *In experiment 2, ridge shrank every learned weight to $\approx0$ and did not beat the hand-set baseline: the data did not support the extra degrees of freedom.*

**LASSO and elastic net.** Swap the penalty: $\ell_1$ ($\alpha\|\mathbf w\|_1$, Tibshirani 1996, *JRSS-B* 58, 267–288) drives some weights *exactly* to zero (selection); the elastic net (Zou–Hastie 2005, *JRSS-B* 67) mixes $\ell_1+\ell_2$ to keep selection while handling correlated groups. **[EST]** These are the standard tools for the high-dimensional factor zoo — and, with the Harvey–Liu–Zhu hurdle (§10), the honest way to fight selection bias.

**Logistic regression (the meta-model).** **[REPO]** `vpts/ml/meta_model.py` minimises the L2-regularised log-loss by gradient descent to learn $\mathbb P(\text{win}\mid\text{features})$:

$$\text{(9.5)}\qquad p_i=\sigma(\mathbf w'\mathbf x_i+b)=\frac{1}{1+e^{-(\mathbf w'\mathbf x_i+b)}},\qquad \hat{\mathbf w}=\arg\min_{\mathbf w}\ -\sum_i\big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\big]+\tfrac{\lambda}{2}\|\mathbf w\|_2^2.$$
**[EST]** The binary cross-entropy is convex; the secondary model *filters* primary signals (meta-labelling, §7), so it is scored by classification metrics (§10), not by direction.

**Gradient-boosted trees (XGBoost).** **[REPO]** A forward stagewise additive ensemble $F_M(\mathbf x)=\sum_{m=1}^{M} \nu\, f_m(\mathbf x)$ of regression trees, each fit to the (regularised) functional gradient of the loss (Friedman 2001, *Annals of Statistics* 29; XGBoost: Chen–Guestrin 2016, *KDD*, adds a second-order objective and explicit tree penalties):

$$\text{(9.6)}\qquad f_m=\arg\min_{f}\ \sum_i \ell\big(y_i,\,F_{m-1}(\mathbf x_i)+f(\mathbf x_i)\big)+\Omega(f),\qquad \Omega(f)=\gamma T+\tfrac12\lambda\|\mathbf w\|^2.$$
**[EST]** Hugely flexible — and hugely prone to overfit on low-signal financial data. In RESEARCH.md experiment 9, XGBoost memorised the training set (in-sample AUC **0.943**) yet scored **0.496 out-of-sample — below 0.5, worse than the linear logistic (0.529)**. The gaudy in-sample number is precisely the trap §10 exists to catch.

**The bias–variance decomposition (why this happens).** For squared-error loss, expected test error at $\mathbf x$ factors as

$$\text{(9.7)}\qquad \mathbb E\big[(y-\hat f(\mathbf x))^2\big]=\underbrace{\big(\mathbb E[\hat f(\mathbf x)]-f(\mathbf x)\big)^2}_{\text{bias}^2}+\underbrace{\operatorname{Var}(\hat f(\mathbf x))}_{\text{variance}}+\underbrace{\sigma_\varepsilon^2}_{\text{irreducible}}.$$
**[DEF]** Hastie–Tibshirani–Friedman (2009, *ESL*). Flexible models (XGBoost) cut bias but inflate variance; in a regime where the irreducible noise $\sigma_\varepsilon^2$ dominates the signal — exactly daily return prediction (§11) — the variance term swamps any bias gain, so the *more flexible* model generalises *worse*. Regularisation ($\alpha,\lambda,\gamma$, tree depth, $\nu$) is the dial that buys variance reduction with bias; cross-validation (§10) is how it is set honestly. This single equation is why "use a bigger model" is usually the wrong answer here, and why this repo's linear book beat its gradient-boosted one.

---

## §10 — Forecast evaluation and the overfitting problem (the core of the harness)

This is where the document earns its keep, and where this repository's discipline lives. The enemy is not bad models; it is **good-looking in-sample numbers that do not survive honest out-of-sample evaluation under dependence and multiplicity.**

**Forecast loss and comparison.** For forecasts with errors $e_t$, the **Diebold–Mariano** (1995, *JBES*) test compares two models' loss differentials $d_t=L(e_{1t})-L(e_{2t})$ with a HAC variance:

$$\text{(10.1)}\qquad \text{DM}=\frac{\bar d}{\sqrt{\widehat{\operatorname{LRV}}(\bar d)/T}}\ \xrightarrow{d}\ \mathcal N(0,1)\ \text{under } H_0:\mathbb E[d_t]=0.$$
**[STAT]** Note the HAC variance (§9) — predictive accuracy comparisons are autocorrelated and must be deflated.

**Why ordinary cross-validation is invalid here, and the fix.** Standard $k$-fold CV assumes i.i.d. rows. Financial labels span a forward horizon, so a train row near a test block **leaks** its label into the test set. The fix has three parts (López de Prado, *AFML* 2018):

1. **Purging** — drop train observations whose label window $[t,t+\text{horizon}]$ overlaps any test block. **[VALID][REPO]** `vpts/validation/cpcv.py`.
2. **Embargo** — additionally drop a few train observations *immediately after* each test block, to break the serial-correlation bleed across the boundary. **[VALID][REPO]**
3. **Combinatorial Purged CV (CPCV)** — instead of one train/test partition, split the timeline into $N$ groups, use *every* combination of $k$ as the test set, and aggregate the recombined out-of-sample segments into a **distribution** of performance, not a point:

$$\text{(10.2)}\qquad \#\text{splits}=\binom{N}{k},\qquad \#\text{backtest paths}\ \varphi[N,k]=\frac{k}{N}\binom{N}{k}=\binom{N-1}{k-1}.$$
**[VALID]** (E.g. $N=6,k=2$: 15 splits, $\varphi=5$ paths.) The output is a *sampling distribution* of OOS return/Sharpe — median, dispersion, fraction of paths profitable — which is what turns "+14.5% in one backtest" into "−0.68% per path, only 36% profitable" (experiment 1). *Honest scope (from the code's own docstring): with a no-parameter strategy CPCV measures robustness/dispersion, not selection-overfitting protection — that protection matters the moment any parameter is fit on the folds.*

**Data-snooping across many strategies.** When you try $M$ configurations and report the best, its in-sample Sharpe is biased up by the maximum-of-$M$ effect. Controls:

- **White's Reality Check** (2000, *Econometrica*) and **Hansen's SPA** (2005, *JBES*): bootstrap the null that the *best* strategy does not beat the benchmark, correcting for the full set searched. **[VALID]**
- **Multiple-testing hurdles**: with hundreds of tested factors, the conventional $|t|>1.96$ is far too lax — Harvey–Liu–Zhu (2016, *RFS* 29, 5–68) argue a new factor needs $|t|\gtrsim3.0$ once Bonferroni/Holm/BHY corrections for the search are applied. **[VALID]**

**Backtest overfitting, quantified.** The frontier tools, all directly relevant to a research log like this one:

$$\text{(10.3)}\qquad \text{PBO}=\mathbb P\big(\text{the in-sample-best config is below-median out-of-sample}\big),$$
**[VALID]** estimated by **combinatorially-symmetric cross-validation** (Bailey–Borwein–López de Prado–Zhu 2017, *J. Computational Finance* — "The Probability of Backtest Overfitting"). And the **Deflated Sharpe Ratio** (Bailey–López de Prado 2014, *J. Portfolio Management* 40(5), 94–107), which discounts a Sharpe for the number of trials $M$, the skew $\gamma_3$ and kurtosis $\gamma_4$ of returns, and the sample length:

$$\text{(10.4)}\qquad \text{DSR}=\Phi\!\left(\frac{(\widehat{\text{SR}}-\text{SR}_0)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{\text{SR}}+\frac{\gamma_4-1}{4}\widehat{\text{SR}}^2}}\right),\qquad \text{SR}_0\approx\sigma_{\text{SR}}\Big[(1-\gamma_E)\Phi^{-1}\!\big(1-\tfrac1M\big)+\gamma_E\Phi^{-1}\!\big(1-\tfrac1{M e}\big)\Big],$$
**[VALID]** where $\text{SR}_0$ is the Sharpe you'd expect from the *luckiest* of $M$ random trials ($\gamma_E$ = Euler–Mascheroni). A strategy is significant only if it beats that inflated benchmark.

**The Sharpe ratio is itself a noisy estimate.** It has a sampling distribution, and autocorrelation corrupts its annualisation (Lo 2002, *FAJ* 58(4), 36–52):

$$\text{(10.5)}\qquad \operatorname{se}(\widehat{\text{SR}})\approx\sqrt{\frac{1+\tfrac12\text{SR}^2}{T}}\ \text{(iid)},\qquad \text{SR}_{\text{annual}}=\frac{q\,\text{SR}}{\sqrt{q+2\sum_{k=1}^{q-1}(q-k)\rho_k}}\ \ \neq\ \sqrt q\,\text{SR}\ \text{unless }\rho_k=0.$$
**[STAT]** Positive autocorrelation (smoothed returns) inflates the naïvely-annualised Sharpe; the $\sqrt q$ rule is valid only for serially-uncorrelated returns. *Reporting a Sharpe without its standard error, or $\sqrt{252}$-annualising an autocorrelated daily series, is the most common quiet overstatement in backtesting.*

**Probabilistic Sharpe ratio and minimum track-record length.** The companion to the DSR with an explicit single-strategy test (Bailey–López de Prado 2012, *J. Risk* 15(2)): the probability that the true Sharpe exceeds a benchmark $\text{SR}^\ast$, and the track length needed to make that claim at confidence $1-\delta$:

$$\text{(10.6)}\qquad \text{PSR}(\text{SR}^\ast)=\Phi\!\left(\frac{(\widehat{\text{SR}}-\text{SR}^\ast)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{\text{SR}}+\frac{\gamma_4-1}{4}\widehat{\text{SR}}^2}}\right),\qquad \text{MinTRL}=1+\Big[1-\gamma_3\widehat{\text{SR}}+\tfrac{\gamma_4-1}{4}\widehat{\text{SR}}^2\Big]\Big(\tfrac{Z_{1-\delta}}{\widehat{\text{SR}}-\text{SR}^\ast}\Big)^2.$$
**[VALID]** Skew/kurtosis enter because non-normal returns make the Sharpe estimate noisier; the related **Minimum Backtest Length** (Bailey–Borwein–López de Prado–Zhu 2014, *Notices of the AMS* 61(5), 458–471) says that with $M$ trials a backtest shorter than $\sim 2\ln M$ years is expected to find a spurious Sharpe of 1 — a length most backtests (this repo's single 2012–2017 window included) do not have.

**Classification metrics (the meta-model's scorecard).** **[REPO]** Meta-labelling (§7, §9b) outputs probabilities scored by:

$$\text{(10.7)}\qquad \text{AUC}=\mathbb P\big(\hat p_{\oplus}>\hat p_{\ominus}\big),\qquad \text{precision}=\frac{TP}{TP+FP},\quad \text{recall}=\frac{TP}{TP+FN},\quad \text{LogLoss}=-\tfrac1n\sum_i\big[y_i\ln\hat p_i+(1-y_i)\ln(1-\hat p_i)\big].$$
**[STAT]** The **ROC-AUC** (Fawcett 2006, *Pattern Recognition Letters* 27) is the probability that a random positive is ranked above a random negative — 0.5 is coin-flipping, and it is *threshold-free* and robust to class imbalance, which is why it is the harness's headline meta-label metric. **Precision** is what a *selective* trader cares about (of the setups I act on, how many win — RESEARCH.md's selectivity lift, §7); the **Brier score** $\tfrac1n\sum(\hat p_i-y_i)^2$ and a reliability (calibration) curve check whether the probabilities are *honest*, not merely well-ranked. *The XGBoost result (in-sample AUC 0.943, OOS 0.496) is read directly off (10.7): perfect in-sample ranking, no out-of-sample ranking at all — (9.7)'s variance term in metric form.*

---

## §11 — Boundary of these methods (what the math cannot fix)

Concrete limits, matching the companion document's closing discipline.

1. **Non-stationarity is the wall.** Every estimator assumes a fixed (or slowly/known-switching) law; structural breaks (§5) are detections of that assumption failing. No amount of in-sample sophistication recovers a relationship that has changed out-of-sample. RESEARCH.md's verdict — "the binding constraint is the data, not the model" — is this limit in practice.

2. **Low signal-to-noise.** Daily return predictability lives at IC $\sim0.01$–$0.05$; with per-date IC noise $\sim1/\sqrt N$, you need enormous breadth $\times$ length to distinguish it from zero. Most apparent edges are inside their own error bars (experiments 5→6).

3. **Multiplicity and reflexivity.** Trying many features/strategies guarantees lucky in-sample winners (§10); and a signal, once found and traded, decays (McLean–Pontiff 2016, *JF* — published predictors lose ~58% post-publication). The signal eats itself. CPCV, PBO, DSR, and the $t>3$ hurdle are damage control, not immunity.

4. **The joint-hypothesis problem (shared with the companion document).** Any test of "is there predictability?" is jointly a test of the predictability *and* of the model/risk-adjustment used to measure it (Fama 1970). You cannot conclude "edge" without a maintained model of normal returns — and a "predictive signal" is frequently just compensation for risk (the $\mathbb P$ vs $\mathbb Q$ wedge of `MARKET_EQUILIBRIUM_MODEL.md`, §IV.L6). A profitable backtest may be harvesting a premium, i.e. *selling insurance*, not finding a mispricing.

5. **Survivorship and look-ahead — the dominant practical confounds.** Conditioning on names that *survived* manufactures edges that **invert** off survivors (RESEARCH.md experiment 9: +0.26%/bet → −1.07%/bet under delisted injection). Two-sided filters (HP, centred z-scores), label leakage, and restated/point-in-time data are the silent look-ahead bugs that purging/embargo (§10) and no-look-ahead feature builders (§6) exist to prevent. These biases are not reduced by better models; only better *data and protocol* remove them.

The through-line: the mathematics of §§1–9 lets you *describe and fit*; the mathematics of §10 lets you *honestly evaluate*; and §11 is the list of things that no estimator can repair and that only data quality, out-of-sample discipline, and humility about non-stationarity can address.

---

## References

Confidence key: ★ = standard result, attribution confident; entries marked **Verified** were checked against the published record on 2026-06-13; ◐ = attribution confident, fine details not independently re-verified.

**Time-series foundations & linear models**
- ★ Cont, R. (2001), "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues," *Quantitative Finance* 1, 223–236. — **Verified**: stylized facts (1.6).
- ★ Lo, A. & A.C. MacKinlay (1988), "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test," *RFS* 1, 41–66. — **Verified**: variance-ratio test (1.7).
- ★ Wold, H. (1938), *A Study in the Analysis of Stationary Time Series*. — Wold decomposition (1.4).
- ★ Box, G. & G. Jenkins (1970), *Time Series Analysis: Forecasting and Control*. — ARIMA methodology (§2).
- ★ Ljung, G. & G. Box (1978), "On a Measure of Lack of Fit in Time Series Models," *Biometrika* 65. — (2.2).
- ★ Dickey, D. & W. Fuller (1979), "Distribution of the Estimators for Autoregressive Time Series with a Unit Root," *JASA* 74. — ADF (2.4).
- ★ Phillips, P. & P. Perron (1988), "Testing for a Unit Root in Time Series Regression," *Biometrika* 75.
- ◐ Kwiatkowski, Phillips, Schmidt & Shin (1992), "Testing the Null Hypothesis of Stationarity…," *J. Econometrics* 54. — KPSS.
- ★ Granger, C. (1969), "Investigating Causal Relations by Econometric Models and Cross-Spectral Methods," *Econometrica* 37.
- ★ Sims, C. (1980), "Macroeconomics and Reality," *Econometrica* 48. — VAR (2.6).
- ★ Engle, R. & C. Granger (1987), "Co-integration and Error Correction," *Econometrica* 55.
- ★ Johansen, S. (1991), "Estimation and Hypothesis Testing of Cointegration Vectors…," *Econometrica* 59.

**Volatility, jumps, long memory**
- ★ Engle, R. (1982), "Autoregressive Conditional Heteroscedasticity…," *Econometrica* 50. — ARCH.
- ★ Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroskedasticity," *J. Econometrics* 31. — GARCH (3.1).
- ★ Nelson, D. (1991), "Conditional Heteroskedasticity in Asset Returns: A New Approach," *Econometrica* 59. — EGARCH.
- ★ Glosten, Jagannathan & Runkle (1993), "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks," *JF* 48. — GJR.
- ◐ Barndorff-Nielsen, O. & N. Shephard (2004), "Power and Bipower Variation with Stochastic Volatility and Jumps," *J. Financial Econometrics* 2. — (3.2).
- ◐ Hosking, J. (1981), "Fractional Differencing," *Biometrika* 68. — ARFIMA.
- ◐ Hurst, H. (1951), "Long-Term Storage Capacity of Reservoirs," *Trans. ASCE* 116. — R/S, Hurst exponent.
- ★ Corsi, F. (2009), "A Simple Approximate Long-Memory Model of Realized Volatility," *J. Financial Econometrics* 7(2), 174–196. — **Verified**: HAR-RV (3.4).

**State space, regimes, breaks, filters, nonlinearity**
- ★ Kalman, R. (1960), "A New Approach to Linear Filtering and Prediction Problems," *J. Basic Engineering* 82. — (4.2).
- ★ Hamilton, J. (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle," *Econometrica* 57. — regime switching (4.3).
- ◐ Brown, Durbin & Evans (1975), "Techniques for Testing the Constancy of Regression Relationships over Time," *JRSS-B* 37. — CUSUM.
- ★ Bai, J. & P. Perron (1998), "Estimating and Testing Linear Models with Multiple Structural Changes," *Econometrica* 66; (2003), *J. Applied Econometrics* 18.
- ◐ Hodrick, R. & E. Prescott (1997), "Postwar U.S. Business Cycles: An Empirical Investigation," *J. Money, Credit and Banking* 29. — HP filter (5.1).
- ◐ Tong, H. (1990), *Non-Linear Time Series: A Dynamical System Approach*, Oxford UP. — SETAR (5.2).
- ◐ Teräsvirta, T. (1994), "Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models," *JASA* 89. — STAR (5.2).
- ◐ Brock, Dechert, Scheinkman & LeBaron (1996), "A Test for Independence Based on the Correlation Dimension," *Econometric Reviews* 15. — BDS test.
- ★ Zinkevich, M. (2003), "Online Convex Programming and Generalized Infinitesimal Gradient Ascent," *ICML*. — **Verified**: online gradient descent attains $O(\sqrt T)$ regret with no stationarity assumption (5.7).
- ★ Besbes, O., Y. Gur & A. Zeevi (2015), "Non-Stationary Stochastic Optimization," *Operations Research* 63(5):1227–1244. — **Verified**: the variation-budget bound on dynamic regret — bounded non-stationarity is learnable, unbounded is not.

**Dependence & machine-learning estimators**
- ◐ Engle, R. (2002), "Dynamic Conditional Correlation," *JBES* 20. — DCC-GARCH (§8).
- ◐ Schreiber, T. (2000), "Measuring Information Transfer," *Physical Review Letters* 85. — transfer entropy (8.2).
- ★ Hoerl, A. & R. Kennard (1970), "Ridge Regression: Biased Estimation for Nonorthogonal Problems," *Technometrics* 12. — ridge (9.4).
- ★ Tibshirani, R. (1996), "Regression Shrinkage and Selection via the Lasso," *JRSS-B* 58, 267–288. — **Verified**: LASSO.
- ◐ Zou, H. & T. Hastie (2005), "Regularization and Variable Selection via the Elastic Net," *JRSS-B* 67. — elastic net.
- ★ Friedman, J. (2001), "Greedy Function Approximation: A Gradient Boosting Machine," *Annals of Statistics* 29. — gradient boosting (9.6).
- ★ Chen, T. & C. Guestrin (2016), "XGBoost: A Scalable Tree Boosting System," *Proc. 22nd ACM SIGKDD*, 785–794. — **Verified** (9.6).
- ★ Hastie, T., R. Tibshirani & J. Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer. — bias–variance (9.7).

**Inference, resampling, evaluation, overfitting**
- ★ Hansen, L.P. (1982), "Large Sample Properties of Generalized Method of Moments Estimators," *Econometrica* 50.
- ★ Newey, W. & K. West (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix," *Econometrica* 55. — HAC (9.1).
- ★ Politis, D. & J. Romano (1994), "The Stationary Bootstrap," *JASA* 89. — (9.2).
- ★ Diebold, F. & R. Mariano (1995), "Comparing Predictive Accuracy," *JBES* 13. — (10.1).
- ★ White, H. (2000), "A Reality Check for Data Snooping," *Econometrica* 68.
- ★ Hansen, P.R. (2005), "A Test for Superior Predictive Ability," *JBES* 23. — SPA.
- ★ Lo, A. (2002), "The Statistics of Sharpe Ratios," *Financial Analysts Journal* 58(4), 36–52. — **Verified**: SR distribution & autocorrelation annualisation (10.5).
- ★ Bailey, D. & M. López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality," *J. Portfolio Management* 40(5), 94–107. — **Verified** (10.4).
- ★ Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest Overfitting," *J. Computational Finance* 20(4). — **Verified**: PBO/CSCV (10.3).
- ★ Bailey, D. & M. López de Prado (2012), "The Sharpe Ratio Efficient Frontier," *J. Risk* 15(2). — **Verified**: probabilistic Sharpe ratio & minimum track record length (10.6).
- ★ Bailey, Borwein, López de Prado & Zhu (2014), "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance," *Notices of the AMS* 61(5), 458–471. — **Verified**: minimum backtest length (10.6).
- ◐ Fawcett, T. (2006), "An Introduction to ROC Analysis," *Pattern Recognition Letters* 27. — AUC/ROC (10.7).
- ★ Harvey, C., Y. Liu & H. Zhu (2016), "… and the Cross-Section of Expected Returns," *RFS* 29(1), 5–68. — **Verified**: multiple-testing $t>3$ hurdle.
- ★ López de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley. — Triple-barrier (§7), meta-labelling, purged/embargoed CPCV (§10); implemented in `vpts/`.
- ★ McLean, R.D. & J. Pontiff (2016), "Does Academic Research Destroy Stock Return Predictability?" *JF* 71, 5–32. — signal decay (§11).
- ★ Fama, E. (1970), "Efficient Capital Markets," *JF* 25. — joint-hypothesis problem (§11).
- ◐ Wilder, J.W. (1978), *New Concepts in Technical Trading Systems*. — ATR / true range (6.1).

*Companion:* [`MARKET_EQUILIBRIUM_MODEL.md`](MARKET_EQUILIBRIUM_MODEL.md) — the asset-pricing model whose SDF, $\mathbb P$/$\mathbb Q$ distinction, and joint-hypothesis guardrail this document's §11 inherits.
