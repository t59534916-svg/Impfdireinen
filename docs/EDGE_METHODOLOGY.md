# The Positive Theory of Edge — Where Skill Comes From, and Why It Survives

The companion documents are, by volume, a catalogue of failure modes. That
asymmetry is honest but incomplete: it tells you how to *avoid being wrong*
without telling you how to *be right*. This document supplies the missing half
with the same mathematical seriousness — and its central result dissolves the
apparent pessimism. The small per-bet edge the empirical chapters treat as
near-futility is precisely the raw material of a world-class return stream, once
it is multiplied by breadth and executed with discipline.

Tags: **[LAW]** a theorem/identity of active management · **[DEF]** definition ·
**[FOC]** optimality condition · **[EC]** empirical/stylized specification ·
**[PRINCIPLE]** a structural economic principle.

## 1. The Fundamental Law of Active Management — the optimistic half of the theorem

The information ratio IR (active return per unit of active risk) decomposes
(Grinold 1989, *JPM* 15:30–37) as

$$\text{(1)}\qquad \mathrm{IR} \;\approx\; \mathrm{IC}\,\sqrt{\mathrm{BR}},$$
**[LAW]** where **IC** is the information coefficient — the cross-sectional
correlation of forecast with realised return, the very statistic of §8 of the
time-series chapter — and **BR** is breadth, the number of *independent* bets per
year. Under realistic constraints (no shorting, leverage caps, costs) this becomes
(Clarke–de Silva–Thorley 2002, *FAJ* 58(5):48–66)

$$\text{(2)}\qquad \mathrm{IR}\;\approx\;\mathrm{TC}\cdot\mathrm{IC}\,\sqrt{\mathrm{BR}},$$
**[LAW]** with **TC** ∈ [0,1] the *transfer coefficient* — the correlation between
the ideal portfolio and the one you can actually implement after constraints.

**The reconciliation.** The empirical chapters establish that honest directional
IC sits around 0.02–0.10 — the "~50–55% accuracy" restated as a correlation, via
the rule of thumb $\mathrm{IC}\approx 2\,(\text{hit rate})-1$ (so 52.5% maps to
$\mathrm{IC}\approx0.05$ and 55% to $\mathrm{IC}\approx0.10$; the bridge is a
heuristic, the exact map depends on the signal's distribution). Read
through (1), that is not a verdict of futility — it is a *per-bet* skill that
compounds with breadth:

$$\text{(3)}\qquad \mathrm{IC}=0.05,\ \mathrm{BR}=1000 \;\Longrightarrow\; \mathrm{IR}\approx 0.05\sqrt{1000}\approx 1.58.$$
**[LAW]** An information ratio of ~1.6 is institutional-grade. **The skeptical fact
("the edge per bet is tiny") and the constructive fact ("you can build a great
fund on it") are the same fact at two scales.** This is exactly how a
statistical-arbitrage operation — and Renaissance's Medallion — works: a
microscopic edge per trade, *industrialised* across an enormous number of trades,
implemented at high transfer coefficient and amplified with leverage. **Breadth,
not per-bet accuracy, is the lever.** The directional-accuracy chapter measured
the wrong thing to feel hopeful about; (1) measures the right thing.

## 2. Breadth is harder than it looks — the independence deflation

Equation (1) counts *independent* bets, and independence is the binding
constraint. $N$ positions sharing a common factor are not $N$ bets. With average
pairwise correlation $\rho$ among the bets, effective breadth collapses:

$$\text{(4)}\qquad \mathrm{BR}_{\text{eff}}\;\approx\;\frac{N}{1+(N-1)\rho}\;\xrightarrow{N\ \text{large}}\;\frac{1}{\rho}.$$
**[DEF, approx]** So 1000 names at $\rho=0.1$ give $\mathrm{BR}_{\text{eff}}\approx
10$, not 1000. This is why the lever is so hard to pull, and why genuine
diversity of *signals* (not just of names) is the scarce input — most apparent
"breadth" is disguised exposure to one factor, and one factor is one bet.

## 3. Capacity and decay — every edge is finite and perishable

Two forces bound the lever. **Capacity:** trading moves prices (the price-impact
$\lambda$ of the microstructure chapter), so the marginal dollar earns less and
beyond a capacity $A_{\text{cap}}$ the edge is impact-eaten. **Decay:** an edge
erodes as it is discovered and crowded (McLean–Pontiff). A realistic edge is
therefore a *decaying, capacity-bounded* asset:

$$\text{(5)}\qquad \alpha(A,t)\;\approx\;\alpha_0\Big(1-\tfrac{A}{A_{\text{cap}}}\Big)\,e^{-t/\tau},$$
**[EC, stylized]** with $A$ = assets, $A_{\text{cap}}$ = capacity, $\tau$ = decay
timescale. This dictates behaviour: size *below* capacity, harvest *fast*, and
treat the edge as a depleting reserve, not an annuity. It also explains
Medallion's defining choice — capping the fund near \$10B and returning outside
capital: they sized to capacity and refused to dilute the edge. An edge you cannot
defend against your own growth is not an edge for long.

## 4. Sizing — Kelly, and the ruin of over-betting even a real edge

Given a genuine edge, *how much* to bet is its own optimisation. The
growth-optimal fraction (Kelly 1956, *Bell System Technical Journal*) for an edge
with expected excess return $\mu$ and variance $\sigma^2$ is

$$\text{(6)}\qquad f^\star \;=\; \frac{\mu}{\sigma^2},$$
**[FOC]** which maximises the long-run growth rate of log-wealth. The crucial
asymmetry: **over-betting a real edge still ruins you** — past $f^\star$ the growth
rate falls, and past $2f^\star$ it turns *negative* even with a true positive
edge. Practitioners bet fractional Kelly (a quarter to a half) for drawdown
control. Note the unity with the asset-pricing chapter: $f^\star$ is the Sharpe
ratio divided by volatility — the same Euler/first-order logic that prices the
market, now applied to your own capital. The kernel was endogenous there; your
risk aversion is the kernel here.

## 5. Why edges survive — the economics of persistence

If markets were the frictionless ideal, every edge would be arbitraged instantly
and the pessimists would be right. They are not, for three structural reasons —
each already developed in this book:

- **Limits to arbitrage** (Shleifer–Vishny 1997; Part III of the model).
  **[PRINCIPLE]** Arbitrage capital is finite and performance-sensitive: losses
  force liquidation precisely when spreads are widest, so mispricings persist and
  can widen before they close. The edge survives because the capital that would
  compete it away is itself constrained.
- **A counterparty must exist.** **[PRINCIPLE]** Every edge is someone's willing
  loss: hedgers paying for insurance (the variance premium), inelastic indexers
  (Koijen–Yogo / Part III), forced sellers (margin calls, redemptions, flows), and
  behaviourally-driven retail. Where a *structural, non-profit-maximising*
  counterparty exists, the edge is durable; where it does not, it decays. This is
  the constructive use of the inelastic-markets idea — the same mechanism the
  model flags as its largest open tension is, from the other side, the *source* of
  durable edge.
- **Barriers to entry.** **[PRINCIPLE]** Speed (co-location), information
  (proprietary data), and *secrecy plus capacity discipline* (Medallion) keep an
  edge from being copied. The moat, not the signal, is what makes an edge persist;
  a published signal has, by construction, no moat (McLean–Pontiff).

## 6. The constructive checklist — the mirror of the red flags

Where the time-series chapter gives a checklist for detecting self-deception, here
is its positive mirror — the conditions for a real, survivable edge:

1. A nameable **source** — information, structure, speed, constraint, or behaviour
   — not a backtest pattern in search of a story.
2. A **counterparty** who is structurally willing to be on the other side.
3. **Breadth** of genuinely independent signals — the lever (eqs. 1, 4).
4. A **transfer coefficient** near 1 — do not let constraints and costs strangle
   the signal (eq. 2).
5. **Capacity** respected and **decay** expected — size below $A_{\text{cap}}$,
   harvest before $\tau$ (eq. 5).
6. **Kelly-disciplined** sizing — never over-bet, even a real edge (eq. 6).
7. A **moat** — speed, data, or secrecy — so the edge is not instantly copied.
8. *Only then*, the negative gate of the time-series chapter: prove it survives
   purged cross-validation, realistic costs, and the deflated Sharpe before
   trusting it.

The two checklists are one method. The §10 gate keeps you from fooling yourself;
this one tells you what you are actually looking for. An edge that passes **both**
— a real source, a willing counterparty, breadth, a moat, *and* a clean
out-of-sample, cost-aware, multiple-testing-corrected signal — is what Medallion
and Buffett found and defended. Rare, specific, real, and perishable — and,
demonstrably, enough.

## References

- ★ Grinold, R. (1989), "The Fundamental Law of Active Management," *Journal of Portfolio Management* 15(3):30–37. — Verified: IR = IC·√BR.
- ★ Clarke, R., H. de Silva & S. Thorley (2002), "Portfolio Constraints and the Fundamental Law of Active Management," *Financial Analysts Journal* 58(5):48–66. — Verified: the transfer-coefficient refinement.
- ★ Kelly, J.L. (1956), "A New Interpretation of Information Rate," *Bell System Technical Journal* 35:917–926. — Growth-optimal sizing (eq. 6).
- ★ Shleifer, A. & R. Vishny (1997), "The Limits of Arbitrage," *Journal of Finance* 52:35–55. — Why mispricings persist.
- ★ McLean, R.D. & J. Pontiff (2016), "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance* 71:5–32. — Decay of published edges.
- *See also* `MARKET_EQUILIBRIUM_MODEL.md` (limits to arbitrage, inelastic demand) and `TIMESERIES_MATH.md` (the information coefficient, the deflated-Sharpe gate).
