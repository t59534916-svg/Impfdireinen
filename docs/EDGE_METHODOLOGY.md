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
rank-correlation of forecast with realised return *across names* (the §8
statistic) — and **BR** is breadth, the number of *independent* bets per
year. Under realistic constraints (no shorting, leverage caps, costs) this becomes
(Clarke–de Silva–Thorley 2002, *FAJ* 58(5):48–66)

$$\text{(2)}\qquad \mathrm{IR}\;\approx\;\mathrm{TC}\cdot\mathrm{IC}\,\sqrt{\mathrm{BR}},$$
**[LAW]** with **TC** ∈ [0,1] the *transfer coefficient* — the correlation between
the ideal portfolio and the one you can actually implement after constraints.

**The reconciliation.** Skill is small however you measure it — but mind *which* measure. The law (1) consumes the **cross-sectional rank-IC** — the §8 statistic: the correlation of forecast with realised return *across names* each rebalance — and an honest one sits around 0.02–0.10. A *single-series directional* hit rate (the SWOT's "~50–55%") is a **different** statistic; the rough bridge $\mathrm{IC}\approx 2(\text{hit rate})-1$ shows only that it is of *comparable smallness* (52.5% ↔ 0.05). Do not conflate them — a per-date cross-sectional IC is not a one-series up/down hit rate — but both carry the same lesson: the per-bet edge is tiny. Read through (1), tiny is not futile — it is a *per-bet* skill that compounds with breadth:

$$\text{(3)}\qquad \mathrm{IC}=0.05,\ \mathrm{BR}=1000 \;\Longrightarrow\; \mathrm{IR}\approx 0.05\sqrt{1000}\approx 1.58.$$
**[LAW]** An information ratio of ~1.6 is institutional-grade — *provided* the 1000 here are **genuinely independent** bets. §2 shows that a naive trade count collapses under cross-correlation to $\mathrm{BR}_{\text{eff}}\approx 1/\rho$ (1000 names at $\rho=0.1$ give ≈ 10, not 1000), so (3) is an **upper bound** whose hard part is manufacturing real independence. With that caveat: **the skeptical fact
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
  counterparty exists, the edge is durable; where it does not, it decays
  (operationalised concretely in §5b). This is
  the constructive use of the inelastic-markets idea — the same mechanism the
  model flags as its largest open tension is, from the other side, the *source* of
  durable edge.
- **Barriers to entry.** **[PRINCIPLE]** Speed (co-location), information
  (proprietary data), and *secrecy plus capacity discipline* (Medallion) keep an
  edge from being copied. The moat, not the signal, is what makes an edge persist;
  a published signal has, by construction, no moat (McLean–Pontiff).

## 5b. Finding and defending edges in current market structure

The "a counterparty must exist" principle is abstract until you can name the
counterparty, observe their flow, and size what is left after everyone else has
noticed. This section operationalises it for **post-2020 US market structure**:
five archetypes of a structural counterparty *willing or compelled to lose on one
side*, each with the predictable price pressure it creates, the observable proxies
and data a practitioner could actually pull, and an honest read on remaining
capacity and moat. The unifying object is the inelastic-markets multiplier — a
dollar of inelastic flow moves aggregate prices by roughly five (Gabaix–Koijen
2021, *flagged as a working-paper estimate*), and the rise of passive has made
individual-stock demand ~11% *more* inelastic over two decades (Haddad–Huebner–
Loualiche 2025, *AER* 115(3)), so these flow channels are structurally *larger*
than they were — and, as §5b.6 warns, correspondingly more crowded.

**5b.1 — Retail options demand.** Retail now exceeds half of US options volume,
routed through payment-for-order-flow to a handful of wholesalers (~90% to three),
and concentrated in cheap, short-dated calls and weeklies/0DTE on which retail
*loses on average* (Bryzgalova–Pavlova–Sikorskaya 2023, *JF* 78(6); average
bid–ask spread ~12.6% on the favoured contracts). The willing loss is the premium
paid for lottery-like convexity; the harvestable edge is the **short variance/tail
premium** (Gârleanu–Pedersen–Poteshman 2009 demand-based option pricing), and the
dealer delta-hedging it induces has a *pervasive* impact on the underlying (Ni–
Pearson–Poteshman–White 2021, *RFS* 34(4)). **Proxies/data:** OCC volume by
trade-size (small-lot ≈ retail), OptionMetrics IvyDB (IV surface, open interest by
strike), Rule 606 PFOF disclosures, dealer-gamma estimates (signed open interest ×
gamma), 0DTE volume share, and the Boehmer–Jones–Zhang signed retail order
imbalance from sub-penny prints. **Capacity/moat:** capacity-rich but
tail-risk-laden — the moat is *risk-bearing balance sheet and execution*, not
secrecy; crowding is directly visible as a compressed variance risk premium
(VRP = implied − realised variance), which is the live capacity gauge.

**5b.2 — ETF creation/redemption flows.** Non-fundamental demand hits an ETF, the
authorised participant creates or redeems shares to keep price near NAV, and the
basket is temporarily dislocated and then mean-reverts. A portfolio short
high-creation and long high-redemption ETFs earned **1.1–2.0% per month**
gross (Brown–Davies–Ringgenberg 2021, *Review of Finance* 25(4)); ETF ownership
also adds non-fundamental volatility that reverses (Ben-David–Franzoni–Moussawi
2018, *JF* 73(6)). **Proxies/data:** daily ETF shares-outstanding (the
create/redeem series), premium/discount to iNAV, and flow = Δshares × price,
attributed down to constituents for the single-name version. **Capacity/moat:**
bounded by AP arbitrage capital and easy to watch, so the edge is moderate and
decaying; the moat is data latency and basket modelling, strongest in less-liquid
or fixed-income ETFs where the arbitrage is frictional (the liquidity-mismatch
channel).

**5b.3 — Index rebalancing.** Index funds must buy additions and sell deletions on
the effective date regardless of price — mechanical, price-insensitive demand, the
textbook front-runnable flow. It is also the cautionary tale: the S&P 500 addition
"index effect" has **collapsed from ~7.4% (1990s) to under 1%** as the trade was
crowded and issuers fought to minimise tracking cost (Greenwood–Sammon 2025, *JF*
80(2)). The durable residue is *benchmarking intensity* — the float share held by
benchmarked investors — which prices inelastic demand into index members (Pavlova–
Sikorskaya 2023, *RFS* 36(3)). **Proxies/data:** index methodologies and
reconstitution calendars (S&P, Russell June reconstitution, MSCI), the
announcement-to-effective window, and a constructed benchmarking-intensity measure
from fund holdings. **Capacity/moat:** the canonical *eroded* edge — near zero in
mega-cap US indices, with whatever remains living in thinner indices (small-cap,
thematic, international) and in the slower-moving benchmarking-intensity mispricing
rather than the event pop.

**5b.4 — Pension and insurance mandates.** Liability- and mandate-constrained
investors trade for non-price reasons: variable-annuity insurers dynamically hedge
the equity guarantees they wrote (VAs were ~$1.5T, ~35% of US life-insurer
liabilities, and these insurers took large equity drawdowns in COVID — Koijen–Yogo
2022, *JF* 77(2)); LDI pensions buy duration to match liabilities; rating- and
benchmark-constrained funds rebalance to fixed weights (Koijen–Yogo 2019 demand
system). Their hedging and rebalancing flows are predictable in *direction and
trigger* even when slow. **Proxies/data:** insurer statutory filings (NAIC
Schedule D holdings), VA guarantee disclosures in 10-Ks, pension funding ratios,
and the interaction of an equity/rate drawdown with disclosed hedging need.
**Capacity/moat:** large, slow, and episodic — the edge is *providing the
insurance or liquidity* during the forced episode, and the moat is balance-sheet
capacity and patience, not speed.

**5b.5 — Margin-driven forced selling.** Leverage plus an adverse move triggers a
margin call and forced deleveraging at any price — a fire sale that overshoots and
reverts (Gârleanu–Pedersen 2011 margin-CAPM; Brunnermeier–Pedersen 2009 margin
spirals; Coval–Stafford 2007 fund-flow fire sales). The contemporary set pieces are
vivid: the **UK LDI/gilt crisis (Sept–Oct 2022)**, where pension hedging funds were
forced to dump gilts into a falling market until the Bank of England intervened,
and the **Archegos unwind (March 2021)**, a forced liquidation of concentrated
swap positions. **Proxies/data:** FINRA aggregate margin debt, the Coval–Stafford
flow-driven *fire-sale price-pressure* measure (flow-weighted overlapping mutual-
fund holdings), prime-broker and dealer stress indicators, and, for crises,
Bank of England / regulatory financial-stability reports (*event-sourced, not
peer-reviewed*). **Capacity/moat:** episodic but very large in stress; the edge is
liquidity provision when others cannot, and the *only* moat that matters is being
unlevered and patient enough to not be the forced seller yourself — the
limits-to-arbitrage point (Shleifer–Vishny) turned into an operating rule.

**5b.6 — Estimating remaining capacity and moat.** Across all five, the same
back-of-envelope governs what is left:

$$\text{(5b.1)}\qquad \text{edge} \;\approx\; \underbrace{(\text{structural flow }\$)}_{\text{counterparty size}}\times\underbrace{(\text{price-insensitivity})}_{\sim\,\text{multiplier }M}\times\underbrace{(1-\text{crowding})}_{\text{competing capital}},$$

**[EC, stylized]** with capacity ∝ (counterparty flow ÷ your price impact $\lambda$)
and moat strength ∝ (how hard the flow is to *observe* + how little arbitrage
capital is aimed at it). The two measurable gauges are therefore the **magnitude**
of the flow (shares-outstanding deltas, margin debt, benchmarking intensity, VA
hedging need) and its **crowding/decay** (the McLean–Pontiff post-publication
fade, the compressed variance premium, and — the sharpest warning — the
*disappearing index effect*: a flow edge, once named and chased, goes to zero).
The constructive reading of the inelastic-markets literature is that passive's rise
has made $M$ larger, so these flows move prices *more*; the defensive reading is
that the same literature is now widely read, so the easy versions are crowded and
the surviving edge lives where the flow is **hard to observe, slow to arbitrage, or
costly to finance** — which is exactly the moat the §6 checklist demands.

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

**Post-2020 market-structure and flow-driven predictability (§5b)** — venues verified against the published record on 2026-06-14 (title/author/year/journal/volume); claim-level figures are from abstracts/summaries, not full-text reads.

- ★ Bryzgalova, S., A. Pavlova & T. Sikorskaya (2023), "Retail Trading in Options and the Rise of the Big Three Wholesalers," *Journal of Finance* 78(6):3465–3514. — Retail options demand, PFOF, retail loses on average (§5b.1).
- ◐ Ni, S.X., N. Pearson, A. Poteshman & J. White (2021), "Does Option Trading Have a Pervasive Impact on Underlying Stock Prices?" *Review of Financial Studies* 34(4):1952–1986. — Dealer delta-hedging moves the underlying (§5b.1).
- ◐ Barber, B., X. Huang, T. Odean & C. Schwarz (2022), "Attention-Induced Trading and Returns: Evidence from Robinhood Users," *Journal of Finance* 77(6):3141–3190. — Behavioural retail herding (§5b.1).
- ★ Brown, D., S. Davies & M. Ringgenberg (2021), "ETF Arbitrage, Non-Fundamental Demand, and Return Predictability," *Review of Finance* 25(4):937–972. — ETF flow predicts reversals, 1.1–2.0%/month (§5b.2).
- ★ Ben-David, I., F. Franzoni & R. Moussawi (2018), "Do ETFs Increase Volatility?" *Journal of Finance* 73(6):2471–2535. — ETF non-fundamental demand and reversal (§5b.2).
- ★ Greenwood, R. & M. Sammon (2025), "The Disappearing Index Effect," *Journal of Finance* 80(2). — The S&P 500 addition effect fell from ~7.4% to <1%; the canonical eroded edge (§5b.3, §5b.6).
- ★ Pavlova, A. & T. Sikorskaya (2023), "Benchmarking Intensity," *Review of Financial Studies* 36(3):859–903. — Benchmarked-investor float share prices inelastic index demand (§5b.3).
- ★ Koijen, R. & M. Yogo (2022), "The Fragility of Market Risk Insurance," *Journal of Finance* 77(2):815–862. — Variable-annuity insurer hedging flows; COVID drawdowns (§5b.4).
- ★ Coval, J. & E. Stafford (2007), "Asset Fire Sales (and Purchases) in Equity Markets," *Journal of Financial Economics* 86(2):479–512. — Flow-driven fire-sale price pressure measure (§5b.5).
- ★ Haddad, V., P. Huebner & E. Loualiche (2025), "How Competitive Is the Stock Market? Theory, Evidence from Portfolios, and Implications for the Rise of Passive Investing," *American Economic Review* 115(3):975–1018. — Passive rise made demand ~11% more inelastic (§5b intro).
- ◐ Bank of England (2022), *Financial Stability Report*, November — the LDI/gilt-market intervention. *Event-sourced regulatory report, not peer-reviewed* (§5b.5).
- *See also* `MARKET_EQUILIBRIUM_MODEL.md` (limits to arbitrage; Gârleanu–Pedersen margin-CAPM; Gârleanu–Pedersen–Poteshman demand-based option pricing; Brunnermeier–Pedersen margin spirals; Koijen–Yogo demand system; Gabaix–Koijen inelastic markets) and `TIMESERIES_MATH.md` (the information coefficient, the deflated-Sharpe gate).
