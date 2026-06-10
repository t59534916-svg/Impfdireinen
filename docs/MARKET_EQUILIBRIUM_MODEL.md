# An Internally Consistent Equational Model of a Financial Market and Its Participants

**Status of this document.** A formal construction in the tradition of general-equilibrium asset pricing and market microstructure. Every equation is tagged with its logical type:

- **[P]** primitive / assumption (with empirical status: *holds / contested / known false*)
- **[FOC]** optimality / first-order condition
- **[EQ]** market-clearing / equilibrium condition
- **[ID]** definition / identity
- **[EC]** econometric specification (an estimating equation, *not* an equilibrium object)

The construction runs the full difficulty ladder: **Part I** (Level 1: one period, finite states, two agent types, competitive equilibrium, closed form), **Part II** (Level 2: dynamic, recursive preferences, noisy rational-expectations equilibrium, market makers and microstructure), **Part III** (Level 3: continuous time, stochastic volatility and jumps, heterogeneous constrained intermediaries, inelastic demand, explicit limits to arbitrage). Each part is self-contained and runs the layers L0–L6. Two closing sections state what the formalization cannot capture and verify its internal consistency, including the tensions that remain.

The pricing kernel is endogenous throughout: it is derived from agents' first-order conditions and market clearing, never posited. Where a block uses a normalization (e.g., a unit gross rate inside a trading window), that is flagged and reconciled in the consistency check.

---

## Notation

Symbols are global across Parts I–III unless the table says otherwise. Where one letter serves two roles, the table says so explicitly and the roles never collide within a part.

| Symbol | Meaning |
|---|---|
| **Spaces, time, probability** | |
| $(\Omega,\mathcal F,\mathbb F,\mathbb P)$ | State space, σ-algebra, filtration $\mathbb F=(\mathcal F_t)$, physical measure |
| $\omega$, $S$ | Generic state; number of states (Part I, finite) |
| $\pi(\omega)$ | Common prior probability of state $\omega$ (Part I) |
| $t$ | Time: $t\in\{0,1\}$ (Part I); $t\in\{0,1,2,\dots\}$ (Part II); $t\in[0,\infty)$ (Part III) |
| $\mathbb E[\cdot]$, $\mathbb E_t[\cdot]$ | Expectation under $\mathbb P$; conditional on $\mathcal F_t$ |
| **Assets, prices, payoffs** | |
| $j\in\{0,1,\dots,J\}$ | Assets; $j=0$ is the one-period riskless bond |
| $d_j(\omega)$, $\mathbf D$ | Time-1 payoff of asset $j$; $S\times J$ payoff matrix (Part I) |
| $D_t$ | Aggregate dividend / consumption-claim flow (Parts II–III) |
| $p_j$, $P_t$ | Asset prices; $q(\omega)$ = Arrow–Debreu state price |
| $R_j$, $R_f$, $r_t$ | Gross return on $j$; gross riskless rate; instantaneous riskless rate. **Endogenous everywhere.** |
| $\bar\theta_j$, $\theta^k$ | Net supply of asset $j$; portfolio of agent $k$ |
| $f$ | Liquidation value of the *microstructure* asset (Parts II–III); $\mu_f,\Sigma_0$ its prior mean and variance, $\Sigma_1$ posterior variance |
| **Agents, preferences** | |
| $k\in\mathcal K$ | Agent types; each is a tuple $(\mathcal I_k, U_k, \mathcal C_k, \mathcal A_k)$: information, objective, constraint set, action space |
| $c_t^k$, $C_t$ | Individual and aggregate consumption; $e^k$ endowments; $W_t^k$ wealth of agent $k$ |
| $u_k(\cdot)$ | Felicity function |
| $\gamma_k$ | Absolute risk aversion (CARA contexts: Parts I–II) or relative risk aversion (CRRA/EZ contexts: Parts II–III); the part states which |
| $\Gamma$ | Aggregate (harmonic) risk aversion: $\Gamma \equiv \big(\sum_k 1/\gamma_k\big)^{-1}$ |
| $\beta$, $\varrho$ | Subjective discount factor (discrete time); subjective discount *rate* (continuous time) |
| $\psi$, $\theta_{EZ}$ | Elasticity of intertemporal substitution; $\theta_{EZ}\equiv(1-\gamma)/(1-1/\psi)$ |
| $V_t$, $\mu_t^{ce}$ | Continuation value (recursive utility); certainty equivalent $\mu_t^{ce}\equiv(\mathbb E_t[V_{t+1}^{1-\gamma}])^{1/(1-\gamma)}$ |
| $m$, $m_{t+1}$, $m_t$ | Stochastic discount factor (state-, one-period-, and level-form) |
| **Information and microstructure** | |
| $s$ | Private signal about $f$ |
| $\tau_x$ | Precision of random variable $x$: $\tau_x \equiv 1/\mathrm{Var}(x)$ (so $\tau_f$, $\tau_\varepsilon$, $\tau_u$, $\tau_{\hat s}$). *Note: $\tau$ is never used for risk tolerance; tolerance is written $1/\gamma_k$.* |
| $\varepsilon$ | Signal noise, $s = f+\varepsilon$ |
| $u$ | Noise-trader net demand, $u\sim N(0,\sigma_u^2)$ |
| $\varphi$ | Mass of informed investors (Grossman–Stiglitz block); $c_I$ = cost of becoming informed |
| $\hat s$, $\varpi$ | Price-revealed composite signal $\hat s = s + \varpi u$; its noise loading $\varpi$ |
| $x$, $X(\cdot)$, $b_I$ | Insider order, insider strategy, insider trading intensity (Kyle block) |
| $y$ | Aggregate order flow $y = x+u$ |
| $\lambda$ | Kyle's lambda (price impact, dollars per share per share) |
| $a$, $b$, $\delta_t$, $\iota$, $\mathcal H_t$ | Ask, bid, public posterior $\delta_t=\mathbb P(f=f^H\mid \mathcal H_t)$, probability a trader is informed, trade history (Glosten–Milgrom block) |
| **Continuous time (Part III)** | |
| $B_t^D$, $B_t^v$, $B^u_t$ | Standard Brownian motions (dividend, variance, noise order flow); correlation $d\langle B^D,B^v\rangle_t=\rho\,dt$ |
| $N(dt,dz)$, $\tilde N$, $\ell(dz)$ | Poisson random measure, its compensated version, jump intensity measure; $z$ = jump size in log dividends |
| $g$, $v_t$, $\kappa$, $\bar v$, $\sigma_v$ | Dividend drift; spot variance; variance mean-reversion speed, level, vol-of-vol |
| $\eta^D_t,\eta^v_t$, $Y(z)$ | Market prices of diffusive risk; SDF jump kernel |
| $\Phi(v)$, $H(n;v)$, $A_\Phi(n),B_\Phi(n)$ | Price–dividend ratio; value of the $n$-maturity dividend strip; its Riccati coefficients |
| **Institutions and frictions (Part III)** | |
| $W^H, W^I, W^A$ | Wealth of households, intermediaries, arbitrageurs. *(Superscripted $W$ is always wealth; $B$ is always Brownian.)* |
| $\vartheta$ | Intermediary outside-equity multiple (He–Krishnamurthy constraint) |
| $h_j$, $\xi_t$ | Margin/haircut on asset $j$; shadow price of the margin constraint |
| $\chi$ | Proportional transaction cost rate |
| $A_i$, $w_i(j)$, $\varkappa_j$, $b_{p,i}$ | Assets under management of institution $i$; its portfolio weight on $j$; characteristics of $j$; its price-elasticity coefficient (Koijen–Yogo block) |
| $\zeta$, $\mathcal M$ | Aggregate price elasticity of demand; flow-to-price multiplier $\mathcal M = 1/\zeta$ |

Two glyph warnings: $\varphi$ (informed mass) vs. $\Phi$ (price–dividend function) are distinct; $\Sigma_0,\Sigma_1$ (Kyle variances) vs. the summation symbol $\sum$ are distinct.

---

# Part I — Level 1: Static exchange economy, two agent types, closed form

## I.L0 — Environment

Time is discrete with two dates, $t\in\{0,1\}$. Uncertainty: finite $\Omega=\{\omega_1,\dots,\omega_S\}$, $\mathcal F = 2^\Omega$, $\mathcal F_0=\{\emptyset,\Omega\}$, $\mathcal F_1=\mathcal F$, common prior $\pi(\omega)>0$.

$$\text{(I.1)}\qquad \big(\Omega, 2^\Omega, (\mathcal F_0,\mathcal F_1), \mathbb P\big),\quad \mathbb P(\{\omega\})=\pi(\omega)>0 .$$
**[P]** Common prior, objective and known. *Status: contested* (rational-expectations common prior; see the Boundary section).

There is one perishable consumption good per date (the numéraire at each date). Assets: $j=0$ is riskless with $d_0(\omega)=1$ for all $\omega$, in zero net supply $\bar\theta_0=0$; assets $j=1,\dots,J$ are risky with payoffs $d_j(\omega)\ge 0$ and net supplies $\bar\theta_j> 0$. Payoff matrix $\mathbf D = [d_j(\omega)]\in\mathbb R^{S\times J}$.

$$\text{(I.2)}\qquad \operatorname{rank}(\mathbf D) = S \quad\text{(complete markets)}.$$
**[P]** *Status: known false empirically* (markets are incomplete; used here to obtain Arrow prices and closed forms).

Agents $k\in\{A,B\}$ have time-0 endowment $e_0^k>0$ of the good and initial asset holdings $\bar\theta^k$ with $\sum_k \bar\theta^k = \bar\theta$. Time-1 endowments are zero — all date-1 consumption is financed through assets:

$$\text{(I.3)}\qquad C_0 \equiv e_0^A + e_0^B, \qquad C_1(\omega) \equiv \sum_{j=1}^J \bar\theta_j d_j(\omega).$$
**[ID]** Aggregate resources. (Setting $e_1^k=0$ is **[P]**, innocuous under (I.2) since any spanned endowment can be repackaged as asset holdings.)

Given (I.2), no arbitrage is equivalent to the existence of strictly positive Arrow prices $q(\omega)$ (state-$\omega$ consumption claims), and every asset price is their linear functional:

$$\text{(I.4)}\qquad p_j = \sum_{\omega} q(\omega)\, d_j(\omega), \qquad p_0 = \sum_\omega q(\omega) \equiv \frac{1}{R_f}.$$
**[ID]** Law of one price under spanning; $R_f$ is *defined* here but its level is **endogenous** — it comes out of (I.9) below, not assumed. (No-arbitrage ⇔ positive state prices in finite state spaces: Ross 1976-style argument; the general theorem is in Harrison–Kreps 1979, *JET*.)

## I.L1 — Information

Both agents observe everything: $\mathcal I_A = \mathcal I_B = \mathbb F$. Signals are degenerate; the prior is common (I.1); updating is trivial. Asymmetric information is deferred to Part II by design — Level 1 isolates risk sharing from information.

**[P]** Symmetric information. *Status: known false* (insider trading and information asymmetries are documented; Part II repairs this).

## I.L2 — Participants as decision problems

Both types are utility-maximizing investors; market makers, noise traders, strategic informed traders, arbitrageurs, and passive demand enter in Parts II–III. For $k\in\{A,B\}$, the tuple is

$$\mathcal I_k = \mathbb F,\qquad
U_k = -\tfrac{1}{\gamma_k}e^{-\gamma_k c_0^k} - \beta \sum_\omega \pi(\omega)\,\tfrac{1}{\gamma_k}e^{-\gamma_k c_1^k(\omega)},\qquad
\mathcal A_k = \{(c_0^k, c_1^k(\cdot))\in\mathbb R^{1+S}\},$$
$$\text{(I.5)}\qquad \mathcal C_k:\quad c_0^k + \sum_\omega q(\omega)\, c_1^k(\omega) \;\le\; \mathcal W^k \equiv e_0^k + \sum_j \bar\theta^k_j\, p_j .$$
**[P]** CARA felicity $u_k(c)=-\gamma_k^{-1}e^{-\gamma_k c}$, $\gamma_A\neq\gamma_B$ allowed; Arrow–Debreu budget (valid given (I.2)). *Status of CARA: known false* (zero wealth effects in risky demand; admits unboundedly negative consumption — this very pathology buys the closed form and removes corner solutions).

Each agent solves $\max_{(c_0,c_1)\in\mathcal C_k} U_k$. The Lagrangian first-order conditions, after eliminating the multiplier by dividing the date-1 condition by the date-0 condition, are

$$\text{(I.6)}\qquad q(\omega) \;=\; \beta\,\pi(\omega)\,\frac{u_k'(c_1^k(\omega))}{u_k'(c_0^k)} \;=\; \beta\,\pi(\omega)\, e^{-\gamma_k\left(c_1^k(\omega)-c_0^k\right)},\qquad k\in\{A,B\},\ \forall\omega .$$
**[FOC]** Necessary and sufficient by strict concavity of $U_k$ and convexity of $\mathcal C_k$; interior by CARA (utility defined on all of $\mathbb R$). Budgets bind by strict monotonicity.

## I.L3 — Price formation

Walrasian clearing, the appropriate mechanism for symmetric-information price-taking agents (no order flow exists to learn from — microstructure mechanisms are vacuous here and appear in Part II):

$$\text{(I.7)}\qquad \sum_k c_0^k = C_0, \qquad \sum_k c_1^k(\omega) = C_1(\omega)\ \ \forall\omega
\qquad\Big(\Longleftrightarrow\ \sum_k\theta^k_j = \bar\theta_j\ \forall j,\ \text{given (I.2) and binding budgets}\Big).$$
**[EQ]** Goods-market clearing at both dates; asset clearing is equivalent by spanning and Walras's law (verified in the Self-Consistency section).

## I.L4 — Equilibrium concept, existence, uniqueness, closed form

**Definition (competitive equilibrium).** A price system $(q(\omega))_{\omega\in\Omega}\gg 0$ and allocations $(c_0^k, c_1^k(\cdot))_{k}$ such that (i) each allocation solves agent $k$'s problem (I.5)–(I.6) at those prices, and (ii) markets clear (I.7).

**Derivation of the closed form.** Solve (I.6) for the consumption spread of each agent:

$$\text{(I.8)}\qquad c_1^k(\omega)-c_0^k = -\frac{1}{\gamma_k}\,\ln\!\frac{q(\omega)}{\beta\,\pi(\omega)} .$$
**[FOC]** (Rearrangement of (I.6); the step is a logarithm.)

Sum (I.8) over $k\in\{A,B\}$ and impose (I.7):

$$C_1(\omega)-C_0 = -\Big(\tfrac{1}{\gamma_A}+\tfrac{1}{\gamma_B}\Big) \ln\!\frac{q(\omega)}{\beta\pi(\omega)}
\quad\Longleftrightarrow\quad
\boxed{\;q(\omega) = \beta\,\pi(\omega)\, e^{-\Gamma\,\left(C_1(\omega)-C_0\right)}\;},\qquad \Gamma \equiv \Big(\tfrac{1}{\gamma_A}+\tfrac{1}{\gamma_B}\Big)^{-1}.$$
$$\text{(I.9)}$$
**[EQ]** Equilibrium state prices in closed form. The market prices risk with the *harmonic mean* of risk aversions — risk tolerances add. This is Wilson's syndicate-aggregation result (Wilson 1968, *Econometrica*): the two-agent economy admits a representative CARA agent with tolerance $1/\Gamma = 1/\gamma_A + 1/\gamma_B$, and here aggregation is *derived*, not assumed.

Existence and uniqueness: (I.9) exhibits the equilibrium price vector explicitly; it is strictly positive since $\pi>0$, and it is the *unique* one consistent with (I.6)–(I.7), since the chain (I.8)→(I.9) is a chain of equivalences. Allocations follow uniquely:

$$\text{(I.10)}\qquad c_1^k(\omega) = c_0^k + \frac{\Gamma}{\gamma_k}\big(C_1(\omega)-C_0\big),$$
**[FOC + EQ]** (substitute (I.9) back into (I.8)). Each agent absorbs the share $\Gamma/\gamma_k = (1/\gamma_k)/(1/\gamma_A+1/\gamma_B)$ of aggregate risk — **linear risk sharing in proportion to risk tolerance** (Wilson 1968; Pareto efficiency follows from the First Welfare Theorem under (I.2), Arrow–Debreu 1954, *Econometrica*; Debreu 1959).

The date-0 levels come from the (binding) budgets. With $Q\equiv\sum_\omega q(\omega)$ and $K \equiv \sum_\omega q(\omega)\,(C_1(\omega)-C_0)$:

$$\text{(I.11)}\qquad c_0^k = \frac{\mathcal W^k - \frac{\Gamma}{\gamma_k} K}{1+Q}.$$
**[ID]** (Insert (I.10) into (I.5) with equality.)

Portfolio implementation: given (I.2), invert spanning. Equation (I.10) says $c_1^k$ is an affine function of $C_1(\omega)=\sum_j\bar\theta_j d_j(\omega)$, so

$$\text{(I.12)}\qquad \theta^k_{j} = \frac{\Gamma}{\gamma_k}\,\bar\theta_j \ \ (j\ge 1),\qquad \theta_0^k = c_0^k - \frac{\Gamma}{\gamma_k} C_0 \ \text{(bond position making (I.10) levels)} .$$
**[ID]** Two-fund separation: every agent holds the *market portfolio* scaled by relative risk tolerance, plus the bond (mutual-fund separation; cf. Cass–Stiglitz 1970, *JET*, and Rubinstein 1974, *JFE*, for aggregation in securities markets).

## I.L5 — Frictions

None at this level: trading is frictionless **[P]**, *status: known false*. Every friction is introduced in Part III as an explicit delta to a Part I/II condition, so the frictionless conditions here are the baseline being perturbed. (One preview: a short-sale constraint $\theta_j^k \ge 0$ would turn (I.6) into an inequality — see (III.20).)

## I.L6 — Pricing kernel and cross-sectional restriction

Define the SDF as the state-price density:

$$\text{(I.13)}\qquad m(\omega) \equiv \frac{q(\omega)}{\pi(\omega)} \;\overset{\text{(I.9)}}{=}\; \beta\, e^{-\Gamma\,(C_1(\omega)-C_0)} \;>\;0 .$$
**[ID + EQ]** The kernel is **endogenous**: it is the marginal-rate-of-substitution of either agent (by (I.6) they agree — complete-markets kernel uniqueness), expressed via clearing in terms of *aggregate* consumption. Positivity is automatic (exponential).

Euler equations and the endogenous riskless rate, derived by substituting (I.13) into (I.4):

$$\text{(I.14)}\qquad \mathbb E[m\,R_j] = 1 \ \ \forall j, \qquad R_f = \frac{1}{\mathbb E[m]} = \frac{1}{\beta\,\mathbb E\big[e^{-\Gamma(C_1-C_0)}\big]} .$$
**[FOC + EQ]** ($R_j \equiv d_j/p_j$.) The discount rate is an output: patient ($\beta\uparrow$), abundant-future ($C_1\downarrow$ risk), or more risk-tolerant ($\Gamma\downarrow$) economies generate it differently.

Cross-sectional restriction (covariance decomposition of (I.14), an algebraic identity given (I.14)):

$$\text{(I.15)}\qquad p_j = \frac{\mathbb E[d_j]}{R_f} + \mathrm{Cov}(m, d_j), \qquad
\mathbb E[R_j] - R_f = -R_f\,\mathrm{Cov}(m, R_j).$$
**[ID]** Assets covarying negatively with $m$ — i.e., paying off when aggregate consumption is high, by (I.13) — carry positive premia. This is the finite-state consumption CAPM (Lucas 1978, *Econometrica*, static version; Breeden 1979, *JFE*, for the continuous-time analogue). Fundamental theorem connection: existence of the strictly positive $m$ in (I.13) is equivalent to absence of arbitrage here (finite states: Harrison–Kreps 1979, *JET*).

---

# Part II — Level 2: Dynamics, recursive preferences, noisy REE, market makers

Part II has two interacting blocks, and the architecture is stated honestly up front. **Block C** (consumption core) prices aggregate risk dynamically and produces the endogenous kernel $m_{t+1}$. **Block M** (microstructure) determines how information gets into the price of an individual security *within* a trading window, through three mechanisms: a noisy rational-expectations market (Grossman–Stiglitz), a strategic batch auction (Kyle), and a sequential dealer market (Glosten–Milgrom). Block M uses the normalization $R_f=1$ inside the window **[P]** (*status: an approximation, accurate for windows of hours or days; reconciled with Block C's kernel in the Self-Consistency section, where the resulting wedge is quantified*).

## II.L0 — Environment

Time $t\in\{0,1,2,\dots\}$. A single consumption good. $(\Omega,\mathcal F,\mathbb F,\mathbb P)$ with $\mathbb F$ generated by the dividend process and the Block-M randomness.

Assets in Block C: a claim to the aggregate dividend $\{D_t\}$ (equity, net supply $1$), and a one-period riskless bond in zero net supply (price $1/R_{f,t}$, **endogenous**). Aggregate dividend growth:

$$\text{(II.1)}\qquad \Delta c_{t+1} \equiv \ln(D_{t+1}/D_t) \ \text{ i.i.d. } N(\mu_c,\sigma_c^2).$$
**[P]** Lognormal i.i.d. growth. *Status: contested-to-false* (growth has small persistent components and heteroskedasticity; the i.i.d. case is used to obtain closed forms, and the consequence of relaxing it is stated at (II.10)).

Asset in Block M: one security with terminal liquidation value $f$, plus the same bond. In the Grossman–Stiglitz (GS) and Kyle blocks, $f\sim N(\mu_f, \Sigma_0)$, $\Sigma_0 = 1/\tau_f$ **[P]** (*normality: known false for prices of limited-liability assets — it puts positive mass on $f<0$; flagged again in the consistency check*). In the Glosten–Milgrom (GM) block, $f\in\{f^L,f^H\}$, prior $\delta_0 = \mathbb P(f=f^H)$ **[P]**.

## II.L1 — Information

Block C: symmetric information, $\mathcal F_t = \sigma(D_s, s\le t)$ for all agents **[P]** (*contested*).

Block M: the signal structure is private. Exact updating rules, stated once and used throughout:

**Lemma N (normal–normal posterior).** If $X\sim N(\mu_X, \tau_X^{-1})$ and, given $X$, the signal $S = X + \epsilon$ with $\epsilon\sim N(0,\tau_\epsilon^{-1})$ independent, then

$$\text{(II.2)}\qquad X\mid S \;\sim\; N\!\left(\frac{\tau_X\,\mu_X + \tau_\epsilon\, S}{\tau_X+\tau_\epsilon},\ \frac{1}{\tau_X+\tau_\epsilon}\right).$$
**[ID]** Bayes' rule for Gaussian families: posterior precision is the sum of precisions; posterior mean is the precision-weighted average (DeGroot 1970, *Optimal Statistical Decisions*).

**Lemma P (projection theorem).** For jointly normal $(X,Z)$: $\ \mathbb E[X\mid Z] = \mu_X + \mathrm{Cov}(X,Z)\mathrm{Var}(Z)^{-1}(Z-\mu_Z)$, $\ \mathrm{Var}(X\mid Z) = \mathrm{Var}(X) - \mathrm{Cov}(X,Z)\mathrm{Var}(Z)^{-1}\mathrm{Cov}(Z,X)$. **[ID]**

Information sets: informed GS investors have $\mathcal I_I = \sigma(s, p)$ with $s = f+\varepsilon$, $\varepsilon\sim N(0,\tau_\varepsilon^{-1})$ independent of $f$; uninformed have $\mathcal I_U = \sigma(p)$; the Kyle insider knows $f$; the Kyle market maker sees only total order flow $y$; the GM dealer sees the trade sequence $\mathcal H_t$. The common prior over all primitives is (II.1) and the distributions above **[P]** (*common prior: contested*).

## II.L2 — Participants as decision problems

**(a) Long-horizon households (Block C), Epstein–Zin.** Identical preferences across households (aggregation is *proved* below, not assumed).

$$\mathcal I = \mathbb F,\qquad \mathcal A = \{(C_t, \alpha_t)\}_{t\ge0} \ \text{(consumption, portfolio weights)},\qquad
\mathcal C: \ W_{t+1} = R_{w,t+1}\,(W_t - C_t),\ W_t \ge \text{natural limit},$$
$$\text{(II.3)}\qquad U:\quad V_t = \Big[(1-\beta)\,C_t^{1-1/\psi} + \beta\,\big(\mu^{ce}_t\big)^{1-1/\psi}\Big]^{\frac{1}{1-1/\psi}},
\qquad \mu^{ce}_t \equiv \big(\mathbb E_t[V_{t+1}^{1-\gamma}]\big)^{\frac{1}{1-\gamma}} .$$
**[P]** Epstein–Zin–Weil recursive utility (Epstein–Zin 1989, *Econometrica*; Weil 1989, *JME*): risk aversion $\gamma$ and EIS $\psi$ disentangled. *Status: contested* (axiomatically coherent; empirically debated). $R_{w,t+1} = \sum_j \alpha_{j,t}R_{j,t+1}$ is the return on total wealth.

*Derivation of the SDF.* The recursion (II.3) is homogeneous of degree one in $(C_t, \mu_t^{ce})$, and the constraint set is linear in wealth, so $V_t$ is homogeneous of degree one in $W_t$. The consumption FOC equates the marginal utility of date-$t$ consumption with the marginal value of wealth (envelope theorem), and the portfolio FOC prices every traded return against the same marginal rates. Carrying out these two steps (Epstein–Zin 1989, Thm. 3.1, for the full argument) yields the one-period kernel in continuation-value form:

$$\text{(II.4)}\qquad m_{t+1} \;=\; \beta\left(\frac{C_{t+1}}{C_t}\right)^{-1/\psi}\left(\frac{V_{t+1}}{\mu^{ce}_t}\right)^{1/\psi - \gamma},
\qquad \mathbb E_t[m_{t+1}R_{j,t+1}]=1 \ \ \forall j .$$
**[FOC]** Equivalent return-based form: $m_{t+1} = \beta^{\theta_{EZ}}(C_{t+1}/C_t)^{-\theta_{EZ}/\psi}R_{w,t+1}^{\theta_{EZ}-1}$ (Epstein–Zin 1989). Sanity check: $\gamma = 1/\psi$ kills the second factor and (II.4) collapses to the CRRA kernel $\beta(C_{t+1}/C_t)^{-\gamma}$, as it must.

*Aggregation (required before any representative-agent use).* Households have identical homothetic recursive preferences and markets in Block C are complete over aggregate states. A Pareto optimum with identical homothetic utilities gives each household a constant share of aggregate consumption, $c_t^k = a_k C_t$ with $a_k$ constant (Gorman aggregation; Negishi 1960, *Metroeconomica*, planner construction). Substituting $c^k_t = a_k C_t$ into (II.4), the constants $a_k$ cancel in every ratio, so all households — and hence a representative household consuming $C_t$ — share the *same* kernel. Heterogeneity that does **not** aggregate (risk aversion, information, constraints) is deliberately housed in Block M and Part III. **[derived]**

**(b) Competitive informed investors (Block M, GS).** $\mathcal I_I = \sigma(s,p)$; CARA over terminal wealth, coefficient $\gamma_I$; $\mathcal A$: demand $x_I\in\mathbb R$; $\mathcal C$: $W = W_0 + x_I(f - p)$ (with $R_f=1$ in-window). For CARA-normal wealth, $\mathbb E[-e^{-\gamma W}] = -\exp(-\gamma\mathbb E W + \tfrac{\gamma^2}{2}\mathrm{Var}\,W)$, so the program is mean–variance and

$$\text{(II.5)}\qquad x_I = \frac{\mathbb E[f\mid s,p] - p}{\gamma_I\,\mathrm{Var}[f\mid s,p]} = \frac{\mathbb E[f\mid s] - p}{\gamma_I \,\mathrm{Var}[f\mid s]},
\qquad \mathbb E[f\mid s] = \frac{\tau_f\mu_f + \tau_\varepsilon s}{\tau_f+\tau_\varepsilon},\quad \mathrm{Var}[f\mid s]=\frac{1}{\tau_f+\tau_\varepsilon} .$$
**[FOC]**, using Lemma N. The second equality holds because, given $s$, the price (a function of $(s,u)$, shown below) adds only information about $u\perp f$.

**(c) Competitive uninformed investors (Block M, GS).** Identical except $\mathcal I_U = \sigma(p)$, risk aversion $\gamma_U$: $x_U = \big(\mathbb E[f\mid p]-p\big)\big/\big(\gamma_U \mathrm{Var}[f\mid p]\big)$ **[FOC]**, with the conditional moments computed in II.L3 once the price function is known — this circularity *is* the rational-expectations fixed point.

**(d) Strategic informed trader (Block M, Kyle).** Risk-neutral; knows $f$; internalizes price impact; $\mathcal A$: market order $x\in\mathbb R$; conjectured pricing rule $p(y)$:

$$\text{(II.6)}\qquad X(f) \in \arg\max_x\ \mathbb E\big[(f - p(x+u))\,x \,\big|\, f\big].$$
**[P + FOC]** Risk neutrality of the insider: *assumption, contested but standard* (short horizon, diversified outside wealth).

**(e) Market makers.** *Kyle batch auction:* at least two risk-neutral dealers observe total $y=x+u$ and Bertrand-compete to fill it; competition drives expected profit conditional on $y$ to zero:

$$\text{(II.7)}\qquad p(y) = \mathbb E[f \mid y].$$
**[EQ]** Zero-profit / semi-strong-efficiency pricing — an equilibrium condition, not an optimization. *Sequential dealer (GM):* quotes set so each side of the book earns zero expected profit against the mixed pool of traders:

$$\text{(II.8)}\qquad a_t = \mathbb E[f \mid \mathcal H_{t-1},\ \text{buy at } a_t],\qquad b_t = \mathbb E[f\mid \mathcal H_{t-1},\ \text{sell at } b_t].$$
**[EQ]** (Glosten–Milgrom 1985, *JFE*.) Regret-free quotes: the expectation conditions on the *information content of the trade itself*.

**(f) Noise / liquidity traders.** Exogenous demand $u\sim N(0,\sigma_u^2)$ (GS, Kyle); in GM, a fraction $1-\iota$ of arrivals buy or sell with probability $\tfrac12$ each.

$$\text{(II.9)}\qquad u \sim N(0,\sigma_u^2) \ \text{independent of } (f,\varepsilon).$$
**[P]** *Status: known false as a description of optimization* — these agents lose money on average and their objective is unmodeled. The standard defense is that they proxy for hedging or liquidity-driven trades (Black 1986, *JF*, "Noise"); their budget problem is a real loose end, reported in the Self-Consistency section, and without them prices would be fully revealing and information rents impossible (Grossman–Stiglitz 1980, *AER*).

## II.L3 — Price formation: Walrasian REE, batch auction, dealer market — and how they relate

### (i) Noisy rational-expectations equilibrium (Walrasian clearing)

Net supply of the Block-M asset is $\bar x\ge0$; mass $\varphi$ of informed, $1-\varphi$ uninformed. Clearing for every realization $(s,u)$:

$$\text{(II.10)}\qquad \varphi\, x_I(s,p) + (1-\varphi)\, x_U(p) + u = \bar x .$$
**[EQ]**

*What the price reveals — derived, not conjectured.* Substituting (II.5) into (II.10), $\big(\varphi\tau_\varepsilon/\gamma_I\big)s + u$ must be a function of $p$ alone (everything else in (II.10) is). Hence observing $p$ is informationally equivalent to observing the composite signal

$$\text{(II.11)}\qquad \hat s \equiv s + \varpi u,\qquad \varpi \equiv \frac{\gamma_I}{\varphi\,\tau_\varepsilon},
\qquad \hat s \mid f \sim N\!\big(f,\ \tau_\varepsilon^{-1} + \varpi^2\sigma_u^2\big),\qquad
\tau_{\hat s} \equiv \big(\tau_\varepsilon^{-1}+\varpi^2\sigma_u^2\big)^{-1}.$$
**[EQ + ID]** More informed capital ($\varphi\uparrow$), better signals ($\tau_\varepsilon\uparrow$), or bolder informed trading ($\gamma_I\downarrow$) make prices more revealing. By Lemma N,

$$\text{(II.12)}\qquad \mathbb E[f\mid p] = \frac{\tau_f \mu_f + \tau_{\hat s}\,\hat s}{\tau_f + \tau_{\hat s}},
\qquad \mathrm{Var}[f\mid p] = \frac{1}{\tau_f+\tau_{\hat s}} .$$
**[ID]** Exact posterior. Solving (II.10) for $p$ using (II.5), (II.12):

$$\text{(II.13)}\qquad p \;=\; \frac{\ \varphi\,\frac{\tau_f\mu_f+\tau_\varepsilon s}{\gamma_I} \;+\; (1-\varphi)\,\frac{\tau_f\mu_f + \tau_{\hat s}\hat s}{\gamma_U} \;+\; u \;-\; \bar x\ }{\ \varphi\,\frac{\tau_f+\tau_\varepsilon}{\gamma_I} + (1-\varphi)\,\frac{\tau_f+\tau_{\hat s}}{\gamma_U}\ } \;=\; A_p + B_p\,\hat s,$$
**[EQ]** with $A_p,B_p$ constants ($B_p>0$), because $\varphi\tau_\varepsilon s/\gamma_I + u = (\varphi\tau_\varepsilon/\gamma_I)\,\hat s$ — the price is a strictly increasing function of $\hat s$ alone, which *verifies* the informational equivalence used at (II.11). The fixed point closes: beliefs in (II.12) are computed from the true law of (II.13). The $-\bar x$ term is the risk-premium discount: with positive supply the price sits below the risk-neutral posterior mean even at neutral signals. (Grossman–Stiglitz 1980, *AER*; Hellwig 1980, *JET*.)

*Endogenous information acquisition.* Let becoming informed cost $c_I$ in numéraire. For a CARA-$\gamma_I$ agent with information set $\mathcal G \supseteq \sigma(p)$, substituting the optimal demand into the objective gives conditional expected utility $-e^{-\gamma_I W_0}\exp\big(-(\mathbb E[f|\mathcal G]-p)^2/(2\mathrm{Var}[f|\mathcal G])\big)$. Taking unconditional expectations with the Gaussian quadratic-form formula, and using the total-variance identity $\mathrm{Var}(f-p) = \mathbb E[\mathrm{Var}(f-p\mid\mathcal G)] + \mathrm{Var}(\mathbb E[f-p\mid\mathcal G])$ — the same for both types since both condition on $p$ — all terms cancel in the ratio except:

$$\text{(II.14)}\qquad \frac{\mathbb EU_{\text{informed}}}{\mathbb EU_{\text{uninformed}}} = e^{\gamma_I c_I}\sqrt{\frac{\mathrm{Var}[f\mid s]}{\mathrm{Var}[f\mid p]}}
\qquad\Longrightarrow\qquad
e^{2\gamma_I c_I} \;=\; \frac{\mathrm{Var}[f\mid p]}{\mathrm{Var}[f\mid s]} = \frac{\tau_f+\tau_\varepsilon}{\tau_f + \tau_{\hat s}(\varphi^*)}\quad\text{at an interior } \varphi^*.$$
**[EQ]** Indifference condition determining $\varphi^*$ (Grossman–Stiglitz 1980). **The Grossman–Stiglitz impossibility follows**: if the price were fully revealing ($\tau_{\hat s}=\tau_\varepsilon$, RHS $=1$) and $c_I>0$, the LHS exceeds 1 — contradiction. Informationally efficient prices and costly information are jointly inconsistent; equilibrium informativeness is interior. **[derived]**

### (ii) Strategic batch auction (Kyle)

Conjecture linear rules $p(y) = \mu_f + \lambda y$ and $X(f) = b_I (f-\mu_f)$; both are *verified*. The insider problem (II.6) given the linear rule: $\max_x x(f-\mu_f-\lambda x)$, strictly concave iff $\lambda>0$, giving

$$\text{(II.15)}\qquad X(f) = \frac{f-\mu_f}{2\lambda} \quad\Rightarrow\quad b_I = \frac{1}{2\lambda}.$$
**[FOC]** The dealer condition (II.7) with Lemma P applied to $(f, y)$, $y = b_I(f-\mu_f)+u$:

$$\text{(II.16)}\qquad \lambda = \frac{\mathrm{Cov}(f,y)}{\mathrm{Var}(y)} = \frac{b_I\Sigma_0}{b_I^2\Sigma_0+\sigma_u^2} .$$
**[EQ]** Solving (II.15)–(II.16) simultaneously:

$$\text{(II.17)}\qquad \boxed{\ \lambda = \frac{1}{2}\sqrt{\frac{\Sigma_0}{\sigma_u^2}},\qquad b_I = \sigma_u\,\Sigma_0^{-1/2},\qquad
\Sigma_1 \equiv \mathrm{Var}(f\mid y) = \frac{\Sigma_0}{2},\qquad
\mathbb E[\pi_{\text{insider}}] = \frac{\sigma_u\sqrt{\Sigma_0}}{2}\ }$$
**[EQ]** (Kyle 1985, *Econometrica*.) Price impact rises with the value of private information ($\Sigma_0$) and falls with noise depth ($\sigma_u$); exactly half the private information enters the price; the insider's expected gain equals, dollar-for-dollar, the noise traders' expected loss, $\mathbb E[(f-p)u] = -\lambda\sigma_u^2 = -\sigma_u\sqrt{\Sigma_0}/2$, with dealers breaking even — the zero-sum audit is run in the Self-Consistency section. The SOC requires $\lambda>0$: satisfied. Uniqueness holds within the linear class; uniqueness over all measurable equilibria is delicate and not claimed here.

### (iii) Sequential dealer market (Glosten–Milgrom)

$f\in\{f^L,f^H\}$, public belief $\delta_t$. Each arrival is informed with probability $\iota$ (buys iff $f = f^H$, sells iff $f=f^L$), noise otherwise (buy/sell with prob. $\tfrac12$). Exact Bayes updates:

$$\text{(II.18)}\qquad \delta_t^{+} \equiv \mathbb P(f^H\mid \mathcal H_{t-1}, \text{buy}) = \frac{\delta_t (1+\iota)}{1+\iota(2\delta_t - 1)},\qquad
\delta_t^{-} \equiv \mathbb P(f^H \mid \mathcal H_{t-1}, \text{sell}) = \frac{\delta_t(1-\iota)}{1+\iota(1-2\delta_t)} .$$
**[ID]** (Bayes' rule with $\mathbb P(\text{buy}\mid f^H) = \iota + \tfrac{1-\iota}{2}$, $\mathbb P(\text{buy}\mid f^L) = \tfrac{1-\iota}{2}$.) Quotes from (II.8):

$$\text{(II.19)}\qquad a_t = f^L + \delta_t^{+}\,(f^H - f^L),\qquad b_t = f^L + \delta_t^{-}\,(f^H-f^L),
\qquad a_t - b_t \Big|_{\delta_t = 1/2} = \iota\,(f^H-f^L) .$$
**[EQ]** The spread is *pure adverse selection* — it is positive with zero order-handling cost iff $\iota>0$, and proportional to both the share of informed trading and the stakes. Because each transaction price is a conditional expectation under the public filtration, the law of iterated expectations makes transaction prices a **martingale** with respect to public information **[derived]** (Glosten–Milgrom 1985, *JFE*).

### (iv) Relating the mechanisms

**Proposition II.1 (dealer pricing is the risk-neutral limit of Walrasian REE).** In (II.13), let the uninformed sector become risk-neutral, $\gamma_U \to 0$. The uninformed terms dominate the numerator and denominator, and

$$\text{(II.20)}\qquad p \;\longrightarrow\; \frac{\tau_f\mu_f + \tau_{\hat s}\,\hat s}{\tau_f + \tau_{\hat s}} \;=\; \mathbb E[f\mid p] .$$
**[derived]** The Walrasian price converges to the conditional expectation of value given the price's information — formally the same object as the Kyle dealer rule (II.7), with the composite signal $\hat s$ playing the role of order flow $y$. The two mechanisms differ in (a) *who* absorbs noise (risk-averse investors at a price concession vs. risk-neutral dealers at zero expected profit) and (b) *strategic* behavior (the Kyle insider shades $b_I$ to manage impact; GS investors are price takers). Kyle 1989 (*REStud*) builds the bridge model — strategic traders submitting demand schedules — that nests both. The supply term $-\bar x$ in (II.13), absent in (II.20), is the risk premium that risk-neutral dealer pricing cannot carry; this matters for the cross-block consistency check.

## II.L4 — Equilibrium concepts

**Definition (noisy REE, Block M-GS).** A price function $P:(s,u)\mapsto p$ and demands $(x_I, x_U)$ such that (i) $x_I$ and $x_U$ are optimal given beliefs computed by Bayes' rule from the *true* joint law of $(f,s,u,P(s,u))$; (ii) (II.10) holds for a.e. $(s,u)$; (iii) endogenous $\varphi^*$ satisfies (II.14). Existence in the linear class: explicit construction (II.13) — for any $\varphi\in(0,1]$ a linear REE exists and is unique *within the linear class* (Hellwig 1980; Grossman–Stiglitz 1980). Global uniqueness is not guaranteed in general; for a class of related economies existence/uniqueness beyond the Gaussian case is established in Breon-Drish (2015, *REStud*) — *citation from memory; year and scope should be verified before quoting*.

**Definition (Kyle equilibrium).** A pair $(X, p(\cdot))$ such that $X$ solves (II.6) given $p(\cdot)$, and $p(\cdot)$ satisfies (II.7) given $X$ — a Bayesian–Nash equilibrium between the insider and a competitive dealer sector. (II.17) is its unique linear representative.

**Definition (GM equilibrium).** Quote processes satisfying (II.8) with beliefs (II.18) — sequential equilibrium of the trading game.

**Block C equilibrium** is Radner: price processes and plans such that households optimize (II.4) and both markets clear, $C_t = D_t$ (goods) and bond in zero net supply **[EQ]** (Radner 1972, *Econometrica*; Lucas 1978 exchange-economy structure).

## II.L5 — Frictions at this level

Two are already endogenous to Level 2, and both are *informational*: (i) the cost of information $c_I$, which by (II.14) makes equilibrium prices necessarily *imperfectly* revealing; (ii) the adverse-selection spread (II.19) and price impact (II.17), which are equilibrium compensation for trading against better-informed counterparties — not physical costs. Physical/contractual frictions (transaction costs, margins, short-sale bans, balance-sheet limits) are introduced as deltas in Part III L5.

## II.L6 — The endogenous kernel and what it prices

Under (II.1) and the proven aggregation, conjecture (and verify) that wealth is proportional to consumption, $V_t = \bar\phi\, C_t$ for a constant $\bar\phi$ (homogeneity of (II.3) plus i.i.d. growth). Then $\mu^{ce}_t = \bar\phi\, C_t \big(\mathbb E[(C_{t+1}/C_t)^{1-\gamma}]\big)^{1/(1-\gamma)} = \bar\phi C_t \cdot \text{const}$, and (II.4) collapses to

$$\text{(II.21)}\qquad m_{t+1} = \tilde\beta\, \Big(\frac{C_{t+1}}{C_t}\Big)^{-\gamma},\qquad
\tilde\beta \equiv \beta\,\Big(\mathbb E\big[(C_{t+1}/C_t)^{1-\gamma}\big]\Big)^{\frac{1/\psi-\gamma}{1-\gamma}} \ \text{(a constant)} .$$
**[derived from (II.4) + (II.1)]** Under i.i.d. growth, Epstein–Zin is observationally equivalent to CRRA for *asset returns* — the EIS moves only the constant $\tilde\beta$ (Kocherlakota 1990, *JF*). Recursive preferences earn their keep only with predictable or long-run consumption risk (Bansal–Yaron 2004, *JF*) — stated as the relevant extension, not derived here.

Endogenous riskless rate and equity premium (lognormal algebra on (II.21) and $\mathbb E_t[m_{t+1}R_{t+1}]=1$):

$$\text{(II.22)}\qquad \ln R_f = -\ln\tilde\beta + \gamma\mu_c - \tfrac{1}{2}\gamma^2\sigma_c^2,
\qquad \ln \mathbb E_t[R_{e,t+1}] - \ln R_f = \gamma\,\sigma_c^2 ,$$
**[EQ]** where $R_e$ is the return on the consumption claim (whose log return is $\text{const} + \Delta c_{t+1}$ under i.i.d. growth, so $-\mathrm{Cov}(\ln m, \ln R_e) = \gamma\sigma_c^2$). *Empirical status, stated plainly:* with $\sigma_c \approx 2\%$ per year, (II.22) delivers a premium of $0.0004\gamma$; matching a ~6% historical premium needs $\gamma\approx150$ — the **equity premium puzzle** (Mehra–Prescott 1985, *JME*), equivalently a violation of the Hansen–Jagannathan (1991, *JPE*) volatility bound at plausible $\gamma$. The model is internally consistent and empirically rejected at this point; Part III's intermediary and disaster channels are the modern repairs.

Cross-sectional restriction, every traded claim including Block M's asset:

$$\text{(II.23)}\qquad \mathbb E_t[R_{j,t+1}] - R_{f,t} = -R_{f,t}\,\mathrm{Cov}_t(m_{t+1}, R_{j,t+1}) \qquad \forall j .$$
**[ID given (II.4)]** Block M prices at $\mathbb E[f\mid\cdot]$, which obeys (II.23) only if $\mathrm{Cov}_t(m_{t+1}, f)\approx 0$ — the diversifiability assumption behind dealer risk neutrality. The wedge when it fails is quantified in the Self-Consistency section.

---

# Part III — Level 3: Continuous time, stochastic volatility, jumps, constrained intermediaries, inelastic demand, limits to arbitrage

## III.L0 — Environment

$t\in[0,\infty)$ on $(\Omega,\mathcal F,\mathbb F,\mathbb P)$, $\mathbb F$ generated by independent Brownian motions $B^D, B^v, B^u$ (with $d\langle B^D, B^v\rangle_t = \rho\,dt$) and a Poisson random measure $N(dt,dz)$ with compensator $\ell(dz)\,dt$, $\int (e^z-1)^2\ell(dz)<\infty$.

Aggregate dividend (also aggregate consumption in equilibrium):

$$\text{(III.1)}\qquad \frac{dD_t}{D_{t^-}} = g\,dt + \sqrt{v_t}\,dB_t^D + \int_{\mathbb R}(e^z - 1)\,N(dt,dz),$$
$$\text{(III.2)}\qquad dv_t = \kappa(\bar v - v_t)\,dt + \sigma_v\sqrt{v_t}\,dB_t^v,\qquad 2\kappa\bar v \ge \sigma_v^2 \ \text{(Feller: } v_t>0\text{)}.$$
**[P]** Jump-diffusion with square-root stochastic variance (Heston 1993, *RFS*; Merton 1976, *JFE*, for the jump component; the affine class is Duffie–Pan–Singleton 2000, *Econometrica*). *Status: constant-volatility models are known false* (volatility clustering: Engle 1982, *Econometrica*) — that is why $v_t$ is a state variable; *known* $\ell(dz)$ is itself *contested-to-false* (see Boundary: peso problems).

Assets: the equity claim to $D$ (net supply 1); an instantaneous riskless bond in zero net supply, rate $r_t$ **endogenous**; a cross-section of claims $j$ with exposures $(\sigma_{jD}, \sigma_{jv}, J_j(z))$ to the three risks; and a *redundant pair* of claims with identical cash flows but different margins (for the limits-to-arbitrage block). Trading is continuous **[P]** (*frictionless continuous trading: known false; the L5 deltas are the corrections*).

## III.L1 — Information

Common information $\mathbb F$ for the macro block **[P]** (*contested*); the microstructure asymmetry of Part II carries over inside trading windows via the continuous-time Kyle model (Back 1992, *RFS*), used in III.L3(ii).

## III.L2 — Participants

**(a) Households $H$** (mass 1): stochastic differential utility (the continuous-time Epstein–Zin), value $V^H$:

$$\text{(III.3)}\qquad V^H_t = \mathbb E_t\!\int_t^\infty \phi(C_s, V^H_s)\,ds,\qquad
\phi(C,V) = \frac{\varrho}{1-1/\psi}\,\frac{C^{1-1/\psi} - \big((1-\gamma)V\big)^{\frac{1-1/\psi}{1-\gamma}}}{\big((1-\gamma)V\big)^{\frac{1-1/\psi}{1-\gamma}-1}} ,$$
**[P]** (Duffie–Epstein 1992, *Econometrica*, stochastic differential utility; reduces to time-additive CRRA with rate $\varrho$ when $\psi = 1/\gamma$.) Constraint: self-financing wealth with consumption outflow; actions: $(C_t, \alpha_t)$. The HJB equation and FOCs:

$$\text{(III.4)}\qquad 0 = \max_{C,\alpha}\ \Big\{\phi(C, V^H) + \mathcal L^{W,v} V^H\Big\},\qquad
\text{[FOC]: } \phi_C = V^H_W,\quad
\alpha:\ \text{(III.13) below}.$$
**[FOC]** ($\mathcal L^{W,v}$ = generator of wealth and state dynamics.) The utility-gradient representation of the kernel for SDU is $m_t = \exp\big(\int_0^t \phi_V(C_s,V_s)\,ds\big)\,\phi_C(C_t, V_t)$ (Duffie–Skiadas 1994, *J. Math. Econ.* — *citation from memory*).

**(b) Intermediaries $I$** (specialists): CRRA-$\gamma_I$ over their consumption stream; uniquely able to hold the full risky menu; households can take equity stakes in intermediaries only up to a multiple of inside capital:

$$\text{(III.5)}\qquad \text{outside equity}_t \;\le\; \vartheta\, W_t^I .$$
**[P]** Skin-in-the-game / agency constraint (He–Krishnamurthy 2013, *AER*). *Status: a reduced form for moral hazard; the constraint's tightness is contested, its existence is well-evidenced (intermediary balance sheets price assets: Adrian–Etula–Muir 2014, JF; He–Kelly–Manela 2017, JFE).* While the intermediary itself trades without binding portfolio constraints, its Euler equation prices the risky menu:

$$\text{(III.6)}\qquad \mathbb E_t[dR_j] - r_t\,dt = \gamma_I\,\mathrm{Cov}_t\!\Big(dR_j,\ \frac{dW^I_t}{W^I_t}\Big) \qquad\text{(for assets the intermediary is marginal in)} .$$
**[FOC]** When (III.5) binds, household capital cannot flow in, the intermediary's leverage $\alpha^I_t$ rises as $W^I_t$ falls, $dW^I/W^I$ loads more on returns, and premia in (III.6) rise — the **amplification region**: $\partial(\text{premium})/\partial W^I < 0$. Global dynamics of such economies (occupation of the constrained region, endogenous volatility, the "volatility paradox" — lower exogenous risk inviting higher leverage and larger endogenous risk) are characterized in Brunnermeier–Sannikov (2014, *AER*).

**(c) Arbitrageurs $A$:** specialize in the redundant pair; mean–variance flow objective; margin constraint:

$$\text{(III.7)}\qquad \max_{\theta}\ \mathbb E_t[dW^A] - \frac{\gamma_A}{2}\mathrm{Var}_t[dW^A]
\quad\text{s.t.}\quad \sum_j h_j\,|\theta_j|\,P_j \;\le\; W^A_t .$$
**[P + objective]** Margins $h_j\in(0,1]$ per dollar of position (Brunnermeier–Pedersen 2009, *RFS*). The Lagrangian FOC with multiplier $\xi_t\ge0$, for a long position in $j$:

$$\text{(III.8)}\qquad \mathbb E_t[dR_j] - r_t\,dt = \gamma_A\,\mathrm{Cov}_t(dR_j, dW^A/W^A) + \xi_t\, h_j\,dt .$$
**[FOC]** The **margin CAPM**: expected returns carry a funding premium proportional to the position's margin use (Gârleanu–Pedersen 2011, *RFS*). Funding tightness $\xi_t > 0$ is an equilibrium object (scarcity of arbitrage capital).

**(d) Passive / price-inelastic institutions** (mass of institutions $i$, AUM $A_i$): portfolio weights given by a characteristics-based demand system,

$$\text{(III.9)}\qquad \frac{w_i(j)}{w_i(0)} = \exp\!\big(b_{p,i}\, \ln P_j + b_{x,i}'\,\varkappa_j + \epsilon_{i,j}\big),\qquad b_{p,i} < 1,$$
**[EC]** — an **econometric specification**, deliberately tagged as such: within this model it is *not* derived from an optimization problem (Koijen–Yogo 2019, *JPE*, derive it from a restricted mean–variance problem with characteristics-spanned beliefs and then estimate it via instruments; here it is grafted on as institutional demand). The condition $b_{p,i}<1$ makes excess demand downward-sloping in price, which underwrites uniqueness of the demand-system fixed point (Koijen–Yogo 2019). The aggregate price elasticity $\zeta$ implied by (III.9) is *finite and empirically small*; the flow-to-price multiplier is

$$\text{(III.10)}\qquad \mathcal M = \frac{1}{\zeta},\qquad \text{empirically } \mathcal M \approx 5 \ \text{at the aggregate level},$$
**[EC]** (Gabaix–Koijen, "In Search of the Origins of Financial Fluctuations: The Inelastic Markets Hypothesis," NBER working paper, c. 2021 — *working-paper status; magnitude quoted from memory; verify before use*). The frictionless blocks of this model imply $\zeta$ orders of magnitude larger; this unresolved tension is reported, not smoothed, in the Self-Consistency section.

**(e) Noise order flow:** cumulative exogenous flow $dq^u_t = \sigma_q\,dB^u_t$ **[P]** (*known false as optimization; same defense and same budget caveat as (II.9)*).

## III.L3 — Price formation: two tiers

**(i) Walrasian tier (low frequency).** For each asset, designated marginal pricers' demands plus institutional demand (III.9) plus noise flow clear against supply:

$$\text{(III.11)}\qquad \theta^H_t + \theta^I_t + \theta^A_t + \sum_i \frac{A_i\,w_i(\cdot)}{P_\cdot} + q^u_t = \bar\theta \qquad \forall t .$$
**[EQ]** Goods market: $C^H_t + C^I_t + C^A_t = D_t$ **[EQ]**.

**(ii) Microstructure tier (within the window).** Transaction prices are set by dealers filtering order flow (continuous-time Kyle: Back 1992, *RFS*): the quote midpoint is $p^e_t = \mathbb E[f\mid \mathcal F^{MM}_t]$ with price impact $\lambda_t$, where $f$ is the *Walrasian value of the asset* — the value (III.11) and the kernel (III.14) assign once current private information becomes public. Observed price decomposes as

$$\text{(III.12)}\qquad p^{obs}_t = p^e_t + \varsigma_t,\qquad p^e_t \ \text{a martingale on } \mathcal F^{MM},\quad \varsigma_t \ \text{stationary (inventory, tick, transient impact)},$$
**[EC]** the efficient-price/noise decomposition used to *measure* microstructure quality (Hasbrouck 1993, *RFS* — econometric identity, not an equilibrium condition). The bridge to the Walrasian tier is the Part II result (II.20) taken to the limit: as private information is revealed ($\Sigma_t\to0$ at the end of the trading window in Kyle/Back), $p^e_t \to f = P^{W}$ — dealer prices converge to the Walrasian price, and microstructure deviations are transient. **The two tiers are consistent by construction at the window boundary; within the window, the microstructure price can deviate from Walrasian value by exactly the unrevealed-information and inventory terms.**

## III.L4 — Equilibrium concept

**Definition (Radner equilibrium with constraints).** Processes $\{P_t, r_t\}$ and plans $\{C^k_t,\theta^k_t\}_{k\in\{H,I,A\}}$ such that (i) each agent's plan solves its HJB problem (III.4)/(III.6)-program/(III.7) given prices and constraints (III.5), (III.7); (ii) institutional demand follows (III.9); (iii) all markets clear (III.11) for all $t$; (iv) all agents share the (correct) law of motion of the aggregate state $(D_t, v_t, W^I_t, W^A_t)$.

Existence: for the frictionless complete-markets exchange-economy core, equilibrium existence in continuous time is classical (Duffie–Huang 1985, *Econometrica* — dynamic implementation of Arrow–Debreu; Karatzas–Lehoczky–Shreve 1990, *Math. OR* — *citation from memory*). With the constraint (III.5) and heterogeneous agents, no general existence/uniqueness theorem applies; equilibria are constructed as solutions to the PDE system in the state $(v_t, \omega^I_t \equiv W^I_t/(W^I_t+W^H_t))$ and verified numerically (He–Krishnamurthy 2013; Brunnermeier–Sannikov 2014). **Multiplicity is real, not hypothetical**: margin-spiral economies admit multiple equilibria for the same fundamentals (Brunnermeier–Pedersen 2009), and the selection used here — the equilibrium branch continuous in fundamentals at $\xi=0$ — is a *convention*, flagged again in the Boundary section.

## III.L5 — Frictions as explicit deltas

Each friction is written as the modification it makes to a frictionless condition derived earlier. Frictionless benchmark (discrete-step notation for clarity of the perturbation argument; $X_{t+1} \equiv P_{t+1}+D_{t+1}$):
$\ P_t = \mathbb E_t[m_{t+1}X_{t+1}]$ — equation (II.23)'s level form.

**Δ1. Proportional transaction costs $\chi$.** Buying costs $(1+\chi)P_t$, selling yields $(1-\chi)P_t$. The buy-now/sell-next-period perturbation must not raise utility, and symmetrically for shorting:

$$\text{(III.19)}\qquad \mathbb E_t\big[m_{t+1}\,(D_{t+1} + (1-\chi)P_{t+1})\big] \;\le\; (1+\chi)\,P_t,
\qquad \mathbb E_t\big[m_{t+1}\,(D_{t+1}+(1+\chi)P_{t+1})\big] \;\ge\; (1-\chi)\,P_t .$$
**[FOC, inequality form]** The Euler *equation* becomes a **band**; inside it the agent does not trade (the no-trade region of Constantinides 1986, *JPE*; Davis–Norman 1990, *Math. OR*, characterize the continuous-time cone). Pricing with bid–ask spreads admits a sublinear extension of the FTAP (Jouini–Kallal 1995, *JET* — *from memory*).

**Δ2. Margin / leverage limits.** Already derived as (III.8): the delta to the frictionless Euler equation is the additive term $\xi_t h_j\,dt$. Two assets with **identical cash flows** but margins $h_1<h_2$ must satisfy, by differencing (III.8),

$$\text{(III.20)}\qquad \mathbb E_t[dR_2 - dR_1] = \xi_t\,(h_2-h_1)\,dt \;>\;0 \ \text{ when } \xi_t>0 ,$$
**[derived]** — a **violation of the law of one price sustained in equilibrium**: the basis is the present value of expected funding-tightness differentials (Gârleanu–Pedersen 2011; the post-2008 persistence of covered interest parity deviations is the canonical empirical instance: Du–Tepper–Verdelhan 2018, *JF*).

**Δ3. Short-sale constraint $\theta^k_j \ge 0$.** KKT turns the Euler equation into

$$\text{(III.21)}\qquad \mathbb E_t[m^k_{t+1} R_{j,t+1}] \le 1,\quad \text{with equality iff } \theta^k_j > 0 .$$
**[FOC, complementary slackness]** With heterogeneous beliefs, pessimists are sidelined and the price reflects optimists only (Miller 1977, *JF*); dynamically, the option to resell to future optimists pushes the price *above even the most optimistic* static valuation — the speculative premium (Harrison–Kreps 1978, *QJE*).

**Δ4. Funding liquidity and spirals.** Make margins endogenous: $h_j = h(\hat\sigma_{j,t})$ increasing in measured volatility **[P]**. Then a price drop → higher measured volatility → higher $h$ → tighter (III.7) → forced liquidation → further price drop: the demand curve can bend backward and **two stable equilibria** (liquid/illiquid) coexist for the same fundamentals (Brunnermeier–Pedersen 2009, *RFS*). The model's selection convention is stated in III.L4.

**Δ5. Intermediary balance-sheet constraint.** Already (III.5)–(III.6): the delta is that the *household* Euler equation fails to price risky assets with equality when (III.5) binds (households cannot reach the exposure), so

$$\text{(III.22)}\qquad \mathbb E_t[m^H\,dR_j] \le r_t\,\frac{dt}{1} \ \text{(inequality for households)};\qquad \text{(III.6) holds with equality for } I .$$
**[FOC, segmented]** Pricing migrates from the household kernel to the intermediary kernel exactly in the states where intermediary capital is scarce — the identifying insight of intermediary asset pricing (He–Krishnamurthy 2013; empirical kernel: He–Kelly–Manela 2017 **[EC]**). Limits to arbitrage in the Shleifer–Vishny (1997, *JF*) sense sit on top of Δ2/Δ5: arbitrage capital is *performance-sensitive* — losses trigger outflows precisely when spreads are widest, so $W^A$ shrinks when $\xi_t h_j$ compensation is highest, and convergence trades can diverge before they converge.

## III.L6 — Aggregate dynamics and the pricing kernel

For the closed-form spine, take the CRRA special case of (III.3) ($\psi = 1/\gamma$, rate $\varrho$) in the **unconstrained region** (where households are marginal and aggregation as in Part II applies); the constrained-region kernel swaps in $W^I$ per (III.6). Equilibrium goods clearing forces $C_t = D_t$, so the kernel is **endogenous**:

$$\text{(III.13)}\qquad m_t = e^{-\varrho t}\,C_t^{-\gamma} = e^{-\varrho t}\,D_t^{-\gamma} .$$
**[FOC + EQ]** Apply Itô's formula with jumps to (III.13) using (III.1):

$$\text{(III.14)}\qquad \frac{dm_t}{m_{t^-}} = \Big[-\varrho - \gamma g + \frac{\gamma(\gamma+1)}{2}v_t\Big]dt \;-\; \underbrace{\gamma\sqrt{v_t}}_{\eta^D_t}\,dB^D_t \;+\; \int_{\mathbb R}\big(\underbrace{e^{-\gamma z}}_{Y(z)} - 1\big)\,N(dt,dz) .$$
**[derived]** The market prices of risk are read off: diffusive consumption risk $\eta^D_t = \gamma\sqrt{v_t}$; **variance risk $\eta^v_t = 0$ under CRRA** — a sharp, honest implication: time-additive power utility attaches no premium to volatility shocks per se; a non-trivial variance risk premium requires $\gamma \neq 1/\psi$ (recursive utility: Drechsler–Yaron 2011, *RFS*) or habits. Jump risk is priced by the kernel's jump kernel $Y(z) = e^{-\gamma z}$: the risk-neutral jump measure is $\ell^{\mathbb Q}(dz) = e^{-\gamma z}\ell(dz)$ — bad jumps ($z<0$) are *over-weighted*, the rare-disaster premium channel (Rietz 1988, *JME*; Barro 2006, *QJE*).

**Endogenous riskless rate** — no drift in $m$ beyond $-r_t$ (definition of the shadow rate, equivalently the bond's Euler equation):

$$\text{(III.15)}\qquad r_t = \varrho + \gamma g - \frac{\gamma(\gamma+1)}{2}\,v_t - \int_{\mathbb R}\big(e^{-\gamma z}-1\big)\,\ell(dz) .$$
**[EQ]** Precautionary terms: high spot variance and negative-jump risk *lower* the equilibrium rate.

**Euler equation as a martingale restriction.** For any claim with price $P_t$ and cash-flow rate $D_t$:

$$\text{(III.16)}\qquad m_tP_t + \int_0^t m_sD_s\,ds \ \text{ is a } \mathbb P\text{-martingale}.$$
**[FOC]** (The continuous-time $\mathbb E[mR]=1$.) For the equity claim, the ansatz $P_t = D_t\,\Phi(v_t)$ with $\Phi(v) = \int_0^\infty H(n;v)\,dn$, where $H(n;v) = \mathbb E_t[(m_{t+n}D_{t+n})/(m_tD_t)]$ is the $n$-maturity dividend strip, gives via Feynman–Kac the exponential-affine solution $H = e^{A_\Phi(n) + B_\Phi(n)v}$ with Riccati ODEs

$$\text{(III.17)}\qquad
\begin{aligned}
A_\Phi' &= -\varrho + (1-\gamma)g + \int\big(e^{(1-\gamma)z}-1\big)\ell(dz) + \kappa\bar v\,B_\Phi,\\
B_\Phi' &= -\frac{\gamma(1-\gamma)}{2} + \big(\rho\sigma_v(1-\gamma) - \kappa\big)B_\Phi + \frac{\sigma_v^2}{2}B_\Phi^2,
\end{aligned}\qquad A_\Phi(0)=B_\Phi(0)=0,$$
**[derived]** (affine machinery: Duffie–Pan–Singleton 2000, *Econometrica*), with the transversality requirement that $A_\Phi(n)+B_\Phi(n)v \to -\infty$ fast enough for $\int_0^\infty H\,dn < \infty$ (a joint restriction on $\varrho, g,\gamma,\ell$). *Verification special case:* $\gamma=1$ (log utility) gives $B_\Phi' = 0 \Rightarrow B_\Phi \equiv 0$, $A_\Phi' = -\varrho$, so $\Phi = 1/\varrho$: the price–dividend ratio is constant — the known log-utility result, a check that (III.17) is not mis-derived.

**Cross-sectional restriction.** For any asset with $dR_j = \mu_j\,dt + \sigma_{jD}\,dB^D + \sigma_{jv}\,dB^v + \int J_j(z)\,N(dt,dz)$, the martingale property of $m\times(\text{gain})$ yields

$$\text{(III.18)}\qquad \mu_j + \int J_j(z)\ell(dz) - r_t = \underbrace{\gamma\,v_t\,\frac{\sigma_{jD}}{\sqrt{v_t}}}_{=\ \eta^D_t\,\sigma_{jD}} + \underbrace{0\cdot\sigma_{jv}}_{\eta^v_t\,\sigma_{jv}} + \int J_j(z)\,\big(1 - e^{-\gamma z}\big)\,\ell(dz).$$
**[derived]** Equity itself ($\sigma_{jD} = \sqrt{v_t}$, $J_j = e^z-1$, plus the $\Phi'(v)/\Phi$ loading on $B^v$ priced at zero under CRRA) earns $\gamma v_t + \int(e^z-1)(1-e^{-\gamma z})\ell(dz)$: a time-varying premium moving with $v_t$ — **endogenous, state-dependent expected returns from constant preferences**, plus a level premium for jump risk.

**Reduction to geometric Brownian motion.** Set $\sigma_v = 0$ with $v_0=\bar v$ (variance constant) and $\ell \equiv 0$ (no jumps): then $B_\Phi$ is irrelevant, $\Phi$ is constant, $P_t \propto D_t$, and $dP/P = (g)\,dt + \sqrt{\bar v}\,dB^D$ — exact GBM, the Black–Scholes environment, with $r$ constant by (III.15). **Every assumption needed for GBM is individually known false** (volatility clusters; jumps happen); GBM is the measure-zero corner of this model, which is the precise sense in which Black–Scholes is a limiting case rather than a foundation.

**Fundamental theorem of asset pricing.** The strictly positive kernel (III.13)/(III.14) defines an equivalent (local) martingale measure $\mathbb Q$ via $d\mathbb Q/d\mathbb P\big|_{\mathcal F_t} = m_t e^{\int_0^t r_s ds}$; conversely, absence of arbitrage in the NFLVR sense is equivalent to the existence of such a measure (Harrison–Kreps 1979, *JET*; Harrison–Pliska 1981, *SPA*; general semimartingale version: Delbaen–Schachermayer 1994, *Mathematische Annalen*). With the L5 frictions, equality-FTAP relaxes to no-arbitrage *bands* (Δ1) and kernel *segmentation* (Δ5): there exist positive kernels, but not a unique one shared by all agents — each agent's shadow kernel prices what that agent can trade frictionlessly.

---

# Boundary of the formalization

What this construction cannot capture, and why — concretely.

1. **Non-stationarity and regime change.** Every parameter ($g,\kappa,\bar v,\sigma_v,\ell,\iota,\sigma_u,\vartheta$, the demand coefficients $b_{p,i}$) is a constant of a single stationary law $\mathbb P$. Decimalization, electronic and high-frequency market making, post-2008 dealer regulation, and QE each *moved* the "constants" of Parts II–III ($\lambda$, spreads, $\zeta$, $\vartheta$). Regime-switching extensions (Hamilton 1989, *Econometrica*) remain stationary meta-models — a known switching law is just a bigger $\mathbb P$. Genuine structural change — policy regimes altering decision rules (Lucas 1976, *Carnegie-Rochester*) — is outside the model class, by construction.

2. **Reflexivity and adaptation-to-measurement.** The GS fixed point (II.14) already shows informativeness is self-limiting; the stronger empirical fact is that the *map itself decays under observation*: published return predictors lose a large fraction of their premium after publication (McLean–Pontiff 2016, *JF*, document post-publication declines on the order of half — magnitudes quoted from memory). Formally, the model would need an equilibrium over the space of *models agents hold*, with the act of estimating an equation like (III.9) shifting the coefficients of (III.9). Nothing in L0–L6 has that structure (informal antecedents: Soros's reflexivity; Lo 2004, *J. Portfolio Management*, adaptive markets). Any econometric use of this document's [EC] equations is conditional on the strategy not yet having eaten its own signal.

3. **Knightian / model uncertainty.** All agents know $\mathbb P$ exactly — including the jump measure $\ell(dz)$ governing events that may never have occurred in sample. Under ambiguity, the Euler equation becomes a worst-case condition over a prior set (maxmin: Gilboa–Schmeidler 1989, *JME*; robust control: Hansen–Sargent 2008, *Robustness*), and ambiguity premia are observationally entangled with the risk premia of (III.18); this model cannot separate them, and *no* model can without auxiliary identifying assumptions.

4. **Heavy tails and rare disasters outside the assumed law.** The model has jumps, but with *known, fixed* $\ell$. Empirical return tails are approximately power-law with exponent near 3 (Gabaix–Gopikrishnan–Plerou–Stanley 2003, *Nature*); finite samples cannot pin the tail of $\ell$ (the peso problem), so the disaster-premium terms in (III.15)/(III.18) are identified mostly by prior, not data (Rietz 1988; Barro 2006 uses cross-country century-scale data precisely because within-market samples cannot). A disaster whose *possibility* is not in $\ell$'s support is not "risk" in this model at all.

5. **The joint-hypothesis problem.** Every empirical test of (I.14)/(II.23)/(III.18) is a joint test of the kernel specification *and* market efficiency/informational assumptions (Fama 1970, *JF*); Hansen–Jagannathan (1991) bounds discipline candidate kernels but do not escape the joint hypothesis. Microstructure tests inherit the same problem through their auxiliary structure (e.g., (III.12) identifies "noise" only relative to the assumed efficient-price dynamics). No test of this model is independent of this model.

6. **Equilibrium multiplicity and selection.** The document *selects* throughout: linear equilibria in (II.13)/(II.17) (nonlinear ones not ruled out); the fundamentals-continuous branch in Δ4's spiral region; uniqueness of the demand-system fixed point only under $b_{p,i}<1$. Sunspot equilibria exist in related economies even with complete markets broken only slightly (Cass–Shell 1983, *JPE*), and run-type multiplicity is endemic to liquidity provision (Diamond–Dybvig 1983, *JPE*). The model has no positive theory of selection; comparative statics in the multiple-equilibrium regions are conditional on the conventions stated.

7. **Architecture limits (self-inflicted, disclosed).** Block M prices a single security in partial equilibrium under an in-window normalization; production, endogenous cash flows, taxes, and agency problems inside the institutions of (III.9) are absent; noise traders' budgets are open (see check 4 below).

---

# Self-consistency check

Run against the construction; tensions are reported, not smoothed.

1. **Markets clear / Walras's law (Part I).** Goods clearing was *imposed* in deriving (I.9); asset clearing must follow. Check: summing (I.11) over $k$, using $\sum_k \Gamma/\gamma_k = 1$ and $\sum_k\mathcal W^k = C_0 + \sum_j\bar\theta_jp_j$, gives $\sum_k c_0^k = C_0 \iff \sum_j \bar\theta_j p_j = \sum_\omega q(\omega)C_1(\omega)$ — which holds term-by-term by (I.3)–(I.4). ✔ Portfolios (I.12) sum to $\bar\theta$ since $\sum_k\Gamma/\gamma_k=1$. ✔

2. **Budget constraints bind.** Strict monotonicity of all utilities ⇒ multipliers strictly positive ⇒ (I.5) and Block-C budgets hold with equality. ✔ Inequality constraints (III.5), (III.7), Δ3 satisfy complementary slackness by construction of the FOCs. ✔

3. **Kyle zero-sum audit.** With (II.17): $f - p = \tfrac12(f-\mu_f) - \lambda u$, so $\mathbb E[\pi_{\text{insider}}] = \mathbb E[(f-p)X] = \sigma_u\sqrt{\Sigma_0}/2$; $\mathbb E[\pi_{\text{noise}}] = \mathbb E[(f-p)u] = -\lambda\sigma_u^2 = -\sigma_u\sqrt{\Sigma_0}/2$; dealers: zero by (II.7). Sum: $0$. ✔ The trading game conserves wealth.

4. **Noise-trader budgets — open tension (T1).** In GS, noise traders' expected loss is $B_p\varpi\sigma_u^2 > 0$ per round (they buy into their own price impact); in GM they pay the spread every round. Their wealth process and participation rationale are outside the model: repeated play requires an unmodeled resource inflow. This is a genuine non-conservation at the boundary of the model, standard in the literature and still a real loose end (Black 1986).

5. **SDF positivity.** (I.13): exponential ⇒ $>0$. ✔ (II.4)/(II.21): powers of positive consumption and value ⇒ $>0$ given $C_t>0$, which (II.1) guarantees. ✔ (III.13): $>0$ given $D_t>0$, guaranteed by (III.1) (geometric dynamics) and Feller for $v$. ✔ **Tension (T2):** the CARA-normal Block M allows $f<0$ and $p<0$ with positive probability — limited liability is violated inside the microstructure block even while the macro blocks enforce positivity. Known false assumption, retained for tractability, quarantined to Block M.

6. **Units and dimensions.** $q(\omega)$: date-0 goods per unit of state-$\omega$ goods; $m = q/\pi$: same units (probabilities dimensionless); $\mathbb E[mR]$: dimensionless = 1. ✔ Kyle: $[\lambda] = \$/\text{share}^2$ and $\lambda = \tfrac12\sqrt{\Sigma_0}/\sigma_u = (\$/\text{share})/\text{share}$ ✔; $[b_I] = \text{share}^2/\$$ and $b_I = \sigma_u/\sqrt{\Sigma_0}$ ✔; precisions $\tau$ carry inverse squared price units, so $\varpi = \gamma_I/(\varphi\tau_\varepsilon)$ has units (price²·tolerance/shares)… consistent with $\varpi u$ being price-of-signal units. ✔ Continuous time: $r,\varrho,\kappa,\ell(\mathbb R), v$ all per unit time; $\sqrt{v}\,dB$ is dimensionless per √time·√time. ✔

7. **FOCs consistent with equilibrium prices.** Part I: prices (I.9) were *constructed* from the FOCs plus clearing; substituting back, (I.6) holds for both agents simultaneously because (I.10) makes their MRSs equal state by state. ✔ Part II-GS: the conjectured information content of the price was *derived* from clearing at (II.11) and re-verified at (II.13) ($B_p>0$, invertible). ✔ Kyle: SOC $\lambda>0$ holds at (II.17). ✔ GM: quotes are fixed points of (II.8) by construction of (II.18). ✔

8. **Cross-block kernel consistency — quantified wedge (T3).** Block C's Euler (II.23) applied to Block M's asset requires $p = \mathbb E_t[m_{t+1}f]\,/\,\mathbb E_t[m_{t+1}]\cdot(1/R_f)\cdot R_f = \mathbb E_t[f]/R_f + \mathrm{Cov}_t(m_{t+1},f)$, while Block M's dealers set $p = \mathbb E[f\mid\text{flow}]$ with $R_f\equiv1$ in-window. The blocks agree iff (a) the window is short ($R_f\approx1$ over hours: fine) and (b) $\mathrm{Cov}_t(m_{t+1},f)\approx0$ (idiosyncratic asset). For an asset with systematic exposure, the microstructure block misprices by exactly $-R_f\,\mathrm{Cov}_t(m_{t+1},f)$ — visible in (II.13) as the $-\bar x$ risk-discount term that survives in the Walrasian price but vanishes from the risk-neutral dealer price (II.20). Integration of the two is *partial by design*; the wedge is the price of tractability and is now on the record.

9. **Who is marginal where — segmentation bookkeeping (T4).** With binding constraints, agents' shadow kernels differ. The model's assignment: households price the bond always and risky claims in the unconstrained region; intermediaries price the risky menu when (III.5) binds (III.6); arbitrageurs price the redundant pair (III.8). Consistency requires that no asset is priced *with equality* by two agents whose kernels disagree on its payoff span — enforced here by the participation structure (households cannot hold the risky menu directly in the constrained region; only arbitrageurs trade the basis pair). This is an assumption about market access doing real work, and it is what makes (III.20)'s law-of-one-price violation an equilibrium rather than a contradiction. ✔ as bookkeeping; flagged as economics.

10. **Elasticity clash (T5).** The optimizing CARA blocks imply per-capita demand slopes $\partial x/\partial p = -(\tau_f+\tau_\varepsilon)/\gamma_I$ — calibrated, enormous aggregate elasticities — while (III.9)–(III.10) impose small finite $\zeta$ with multiplier $\mathcal M\approx5$. Both cannot describe the same investors at the same frequency. The model reconciles them only by population weights (most wealth in inelastic mandates, a thin elastic fringe), which is the inelastic-markets hypothesis itself — currently an unresolved frontier dispute, not a settled fact. Reported as the model's largest live tension.

11. **Endogeneity of the discount rate.** $R_f$ in (I.14), $R_{f,t}$ in (II.22), $r_t$ in (III.15): each is derived from preferences + clearing, never assumed. The only exogenous rate anywhere is the in-window normalization $R_f=1$ of Block M, disclosed at the top of Part II and bounded in check 8. ✔

12. **Special-case verifications.** $\gamma=1/\psi$ collapses (II.4) to CRRA ✔; i.i.d. growth collapses EZ pricing to (II.21) ✔ (Kocherlakota 1990); $\gamma=1$ gives constant price–dividend ratio in (III.17) ✔; $\sigma_v=0,\ell=0$ gives exact GBM ✔. The model degrades gracefully to its classical corners.

---

# References

Confidence key: ★ = standard result, attribution confident from memory; ◐ = attribution confident, details (year/journal/exact scope) should be verified; ○ = reconstructed from memory or unverified status — verify before citing onward.

- ★ Arrow, K. & G. Debreu (1954), "Existence of an Equilibrium for a Competitive Economy," *Econometrica*. — Existence, welfare theorems (I.L4).
- ★ Back, K. (1992), "Insider Trading in Continuous Time," *RFS*. — Continuous-time Kyle (III.L3).
- ◐ Adrian, T., E. Etula & T. Muir (2014), "Financial Intermediaries and the Cross-Section of Asset Returns," *JF*.
- ★ Bansal, R. & A. Yaron (2004), "Risks for the Long Run," *JF*. — Where EZ matters (II.L6).
- ★ Barro, R. (2006), "Rare Disasters and Asset Prices in the Twentieth Century," *QJE*.
- ★ Black, F. (1986), "Noise," *JF*. — The noise-trader apology (II.9).
- ★ Breeden, D. (1979), "An Intertemporal Asset Pricing Model…," *JFE*. — Consumption CAPM.
- ○ Breon-Drish, B. (2015), "On Existence and Uniqueness of Equilibrium in a Class of Noisy Rational Expectations Models," *REStud*. — Year/scope from memory.
- ★ Brunnermeier, M. & L. Pedersen (2009), "Market Liquidity and Funding Liquidity," *RFS*. — Margin spirals, multiplicity (Δ2, Δ4).
- ★ Brunnermeier, M. & Y. Sannikov (2014), "A Macroeconomic Model with a Financial Sector," *AER*. — Global dynamics, volatility paradox.
- ◐ Cass, D. & J. Stiglitz (1970), "The Structure of Investor Preferences and Asset Returns…," *JET*. — Fund separation.
- ★ Cass, D. & K. Shell (1983), "Do Sunspots Matter?" *JPE*.
- ★ Constantinides, G. (1986), "Capital Market Equilibrium with Transaction Costs," *JPE*.
- ◐ Davis, M. & A. Norman (1990), "Portfolio Selection with Transaction Costs," *Mathematics of Operations Research*.
- ★ Debreu, G. (1959), *Theory of Value*, Yale UP.
- ★ DeGroot, M. (1970), *Optimal Statistical Decisions*, McGraw-Hill. — Lemma N.
- ★ Delbaen, F. & W. Schachermayer (1994), "A General Version of the Fundamental Theorem of Asset Pricing," *Mathematische Annalen*.
- ★ Diamond, D. & P. Dybvig (1983), "Bank Runs, Deposit Insurance, and Liquidity," *JPE*.
- ◐ Drechsler, I. & A. Yaron (2011), "What's Vol Got to Do with It," *RFS*. — Variance risk premium needs non-CRRA.
- ◐ Du, W., A. Tepper & A. Verdelhan (2018), "Deviations from Covered Interest Rate Parity," *JF*.
- ★ Duffie, D. & L. Epstein (1992), "Stochastic Differential Utility," *Econometrica*.
- ◐ Duffie, D. & C.-F. Huang (1985), "Implementing Arrow–Debreu Equilibria by Continuous Trading of Few Long-Lived Securities," *Econometrica*.
- ★ Duffie, D., J. Pan & K. Singleton (2000), "Transform Analysis and Asset Pricing for Affine Jump-Diffusions," *Econometrica*.
- ○ Duffie, D. & C. Skiadas (1994), "Continuous-Time Security Pricing: A Utility Gradient Approach," *J. Math. Econ.* — Details from memory.
- ★ Engle, R. (1982), "Autoregressive Conditional Heteroscedasticity…," *Econometrica*.
- ★ Epstein, L. & S. Zin (1989), "Substitution, Risk Aversion, and the Temporal Behavior of Consumption and Asset Returns: A Theoretical Framework," *Econometrica*; and (1991) empirical companion, *JPE*.
- ★ Fama, E. (1970), "Efficient Capital Markets: A Review of Theory and Empirical Work," *JF*. — Joint hypothesis.
- ★ Gabaix, X., P. Gopikrishnan, V. Plerou & H.E. Stanley (2003), "A Theory of Power-Law Distributions in Financial Market Fluctuations," *Nature*.
- ○ Gabaix, X. & R. Koijen (c. 2021), "In Search of the Origins of Financial Fluctuations: The Inelastic Markets Hypothesis," NBER working paper. — Working-paper status and multiplier magnitude unverified here.
- ★ Gârleanu, N. & L. Pedersen (2011), "Margin-Based Asset Pricing and Deviations from the Law of One Price," *RFS*.
- ★ Gilboa, I. & D. Schmeidler (1989), "Maxmin Expected Utility with Non-Unique Prior," *JME*.
- ★ Glosten, L. & P. Milgrom (1985), "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders," *JFE*.
- ★ Grossman, S. & J. Stiglitz (1980), "On the Impossibility of Informationally Efficient Markets," *AER*.
- ◐ Hamilton, J. (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series…," *Econometrica*.
- ★ Hansen, L.P. & R. Jagannathan (1991), "Implications of Security Market Data for Models of Dynamic Economies," *JPE*.
- ★ Hansen, L.P. & T. Sargent (2008), *Robustness*, Princeton UP.
- ★ Harrison, J.M. & D. Kreps (1978), "Speculative Investor Behavior in a Stock Market with Heterogeneous Expectations," *QJE*; (1979), "Martingales and Arbitrage in Multiperiod Securities Markets," *JET*.
- ★ Harrison, J.M. & S. Pliska (1981), "Martingales and Stochastic Integrals in the Theory of Continuous Trading," *Stochastic Processes and their Applications*.
- ◐ Hasbrouck, J. (1993), "Assessing the Quality of a Security Market: A New Approach to Transaction-Cost Measurement," *RFS*.
- ★ He, Z. & A. Krishnamurthy (2013), "Intermediary Asset Pricing," *AER*.
- ★ He, Z., B. Kelly & A. Manela (2017), "Intermediary Asset Pricing: New Evidence from Many Asset Classes," *JFE*.
- ★ Hellwig, M. (1980), "On the Aggregation of Information in Competitive Markets," *JET*.
- ★ Heston, S. (1993), "A Closed-Form Solution for Options with Stochastic Volatility…," *RFS*.
- ◐ Jouini, E. & H. Kallal (1995), "Martingales and Arbitrage in Securities Markets with Transaction Costs," *JET*.
- ○ Karatzas, I., J. Lehoczky & S. Shreve (1990), "Existence and Uniqueness of Multi-Agent Equilibrium in a Stochastic, Dynamic Consumption/Investment Model," *Mathematics of Operations Research*. — Details from memory.
- ◐ Kocherlakota, N. (1990), "Disentangling the Coefficient of Relative Risk Aversion from the Elasticity of Intertemporal Substitution: An Irrelevance Result," *JF*.
- ★ Koijen, R. & M. Yogo (2019), "A Demand System Approach to Asset Pricing," *JPE*.
- ★ Kyle, A. (1985), "Continuous Auctions and Insider Trading," *Econometrica*; (1989), "Informed Speculation with Imperfect Competition," *REStud*.
- ★ Lo, A. (2004), "The Adaptive Markets Hypothesis," *Journal of Portfolio Management*.
- ★ Lucas, R. (1976), "Econometric Policy Evaluation: A Critique," *Carnegie-Rochester Conference Series*; (1978), "Asset Prices in an Exchange Economy," *Econometrica*.
- ★ McLean, R.D. & J. Pontiff (2016), "Does Academic Research Destroy Stock Return Predictability?" *JF*. — Decay magnitudes quoted from memory.
- ★ Mehra, R. & E. Prescott (1985), "The Equity Premium: A Puzzle," *JME*.
- ★ Merton, R. (1971), "Optimum Consumption and Portfolio Rules in a Continuous-Time Model," *JET*; (1973), "An Intertemporal Capital Asset Pricing Model," *Econometrica*; (1976), "Option Pricing when Underlying Stock Returns Are Discontinuous," *JFE*.
- ★ Miller, E. (1977), "Risk, Uncertainty, and Divergence of Opinion," *JF*.
- ◐ Negishi, T. (1960), "Welfare Economics and Existence of an Equilibrium for a Competitive Economy," *Metroeconomica*.
- ★ Radner, R. (1972), "Existence of Equilibrium in Plans, Prices, and Price Expectations…," *Econometrica*.
- ★ Rietz, T. (1988), "The Equity Risk Premium: A Solution," *JME*.
- ◐ Rubinstein, M. (1974), "An Aggregation Theorem for Securities Markets," *JFE*.
- ★ Shleifer, A. & R. Vishny (1997), "The Limits of Arbitrage," *JF*.
- ★ Weil, P. (1989), "The Equity Premium Puzzle and the Risk-Free Rate Puzzle," *JME*.
- ★ Wilson, R. (1968), "The Theory of Syndicates," *Econometrica*. — Risk-tolerance aggregation (I.9)–(I.10).
- ★ Knight, F. (1921), *Risk, Uncertainty and Profit*, Houghton Mifflin.
