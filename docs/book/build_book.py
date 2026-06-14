"""Assemble the whole session into one scientific book → PDF.

Pipeline: source Markdown (+ authored front/back matter)
  → protect code & math → Markdown→HTML → re-render math with KaTeX (node) and
  code with Pygments → academic CSS + cover + linked TOC + chapter plates with
  figures → WeasyPrint → docs/Session-Compendium.pdf

Prereqs (installed during the build session):
    pip install markdown weasyprint pygments matplotlib pymupdf
    cd docs/book && npm install katex
    python docs/book/generate_book_figures.py
    python docs/book/build_book.py
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from weasyprint import HTML

BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parents[1]
DOCS = ROOT / "docs"
IMG = BOOK / "img"
DIST = BOOK / "node_modules" / "katex" / "dist"
OUT = DOCS / "Session-Compendium.pdf"
DATE = "13 June 2026"


# --------------------------------------------------------------------------- #
# math + code protection
# --------------------------------------------------------------------------- #
def _protect(text: str):
    code_blocks, inline_code, math = [], [], []

    def fenced(m):
        code_blocks.append((m.group(1).strip(), m.group(2)))
        return f"\n\n@@CODE{len(code_blocks) - 1}@@\n\n"
    text = re.sub(r"```([^\n]*)\n(.*?)```", fenced, text, flags=re.S)

    def inlc(m):
        inline_code.append(m.group(1))
        return f"@@INLC{len(inline_code) - 1}@@"
    text = re.sub(r"`([^`\n]+)`", inlc, text)

    def disp(m):
        math.append((m.group(1).strip(), True))
        return f"\n\n@@MATH{len(math) - 1}@@\n\n"
    text = re.sub(r"\$\$(.+?)\$\$", disp, text, flags=re.S)

    # inline $...$ (pandoc heuristic: open not before space; close not after space,
    # not followed by a digit; reject only on an UNESCAPED inner $ — currency — so
    # that an escaped \$ inside the math, e.g. units like "$/share", is allowed).
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "$" and (i == 0 or text[i - 1] != "\\") and i + 1 < n and not text[i + 1].isspace():
            k, found = i + 1, -1
            while k < n:
                if text[k] == "$" and text[k - 1] not in " \t\\" and (k + 1 >= n or not text[k + 1].isdigit()):
                    found = k
                    break
                if text[k] == "\n" and k + 1 < n and text[k + 1] == "\n":
                    break
                k += 1
            if found > 0 and not re.search(r"(?<!\\)\$", text[i + 1:found]):
                math.append((text[i + 1:found], False))
                out.append(f"@@MATH{len(math) - 1}@@")
                i = found + 1
                continue
        out.append(c)
        i += 1
    return "".join(out), code_blocks, inline_code, math


def _render_math(math):
    if not math:
        return []
    payload = json.dumps([{"tex": t, "display": d} for t, d in math])
    res = subprocess.run(["node", str(BOOK / "render_math.js")], input=payload,
                         capture_output=True, text=True, cwd=BOOK)
    if res.returncode != 0:
        raise RuntimeError("KaTeX render failed: " + res.stderr[:500])
    return json.loads(res.stdout)


def _hl(lang: str, code: str) -> str:
    try:
        lexer = get_lexer_by_name(lang or "text")
    except Exception:
        try:
            lexer = guess_lexer(code)
        except Exception:
            lexer = get_lexer_by_name("text")
    return highlight(code, lexer, HtmlFormatter(cssclass="codehl", nowrap=False))


def md_to_html(text: str) -> str:
    text, code_blocks, inline_code, math = _protect(text)
    body = markdown.markdown(
        text, extensions=["tables", "sane_lists", "attr_list", "md_in_html"])
    rendered = _render_math(math)
    for i, ((_, disp), h) in enumerate(zip(math, rendered)):
        if disp:
            block = f'<div class="eq">{h}</div>'
            body = body.replace(f"<p>@@MATH{i}@@</p>", block).replace(f"@@MATH{i}@@", block)
        else:
            body = body.replace(f"@@MATH{i}@@", f'<span class="mi">{h}</span>')
    for i, (lang, code) in enumerate(code_blocks):
        block = _hl(lang, code)
        body = body.replace(f"<p>@@CODE{i}@@</p>", block).replace(f"@@CODE{i}@@", block)
    for i, c in enumerate(inline_code):
        esc = c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body = body.replace(f"@@INLC{i}@@", f"<code>{esc}</code>")
    return body


def figure(fname: str, num: str, caption: str, cls: str = "lead") -> str:
    return (f'<figure class="{cls}"><img src="file://{IMG / fname}"/>'
            f'<figcaption><b>Figure {num}.</b> {caption}</figcaption></figure>')


# --------------------------------------------------------------------------- #
# front / back matter (authored)
# --------------------------------------------------------------------------- #
PREFACE = """
**Abstract.** This compendium assembles, into a single structured volume, the
work of one interactive research session. It moves from pure theory to running,
tested code and is held together by a single thread: the gap between what prices
*are* and what they *say*. Chapter 2 builds a general-equilibrium asset-pricing
and microstructure model whose one guardrail is that the pricing kernel *exists
but is not identified from prices*. Chapter 3 develops the mathematics of
analysing financial time series, ending in the evaluation discipline (purged
cross-validation, the deflated Sharpe ratio) that the theory demands. Chapter 4
is an evidence-based SWOT of AI and machine-learning architectures for
*directional* stock forecasting, concluding that honest out-of-sample directional
accuracy is ~50–55% across every model family. Chapter 5 turns the defensible
recipe into a dependency-free, unit-tested implementation. Chapter 6 supplies the
constructive counterpart — the Fundamental Law of Active Management, which shows
the very edge the empirical chapters call tiny becomes a world-class information
ratio once multiplied by breadth, plus why such edges survive. Chapter 7 states the
synthesis. The unifying claim, proved from two directions, is that *markets price
risk, not odds; models estimate odds, not edge; and the unidentified pricing
kernel is exactly what makes both asset pricing hard and machine-learning
backtests lie.* The purpose is constructive, not pessimistic: to find the model,
regime, and horizon where an edge is real — and the conclusion is explicit that
markets *are* beatable, rarely and specifically, as Medallion and Buffett prove.
The severity throughout is the filter that protects scarce time from what cannot
work; it is not a verdict that nothing does.

**How to read it.** Each chapter is self-contained and carries its own equations
(tagged by logical type), figures, and references. Equations are rendered with
KaTeX; figures are original. A methodology appendix documents how the volume was
built, including the tooling limits encountered and a candid account of a bug —
an in-sample leak — caught inside the project's own demonstration.

**Provenance & disclaimer.** Compiled by Claude Code from an interactive session
on 13 June 2026. It is a research and education artefact, not financial advice.
Load-bearing citations were verified against the published record where possible;
items that could not be independently re-verified are flagged in place.
"""

READER_GUIDE = """
This volume is dense by construction — it moves from expected-utility theory to
stochastic calculus to gradient-boosted trees — so read it for your purpose, not
cover to cover. Its aim is **constructive**: to identify which model fits which
sector, regime, and horizon, and to supply a filter severe enough that scarce time
is not spent on what cannot work. It is critical because the domain punishes
credulity, not because it is pessimistic — its conclusion is that markets *are*
beatable, rarely and specifically (Chapter 6 supplies the positive theory; Chapter
7 the synthesis), and the severity is in service of finding where.

- **The thesis in two pages:** the Plain-Language Summary, then the opening of Chapter 6.
- **What a price means (theory):** Chapter 2 — assumes comfort with expected
  utility and Itô calculus; its boundary and self-consistency sections need no
  calculus.
- **The empirical and methodological core:** Chapters 3 and 4 — assume basic
  statistics; the evaluation gate (§10 of Chapter 3) and the SWOT (Chapter 4) are
  self-contained.
- **The working code:** Chapter 5 — assumes Python; the results table stands alone.
- **A reading aid throughout:** every equation is tagged by logical type —
  *primitive*, *optimality*, *equilibrium/market-clearing*, *definition*, or
  *estimating specification* — so the structure of an argument is legible even
  without parsing every symbol. Assumptions carry their empirical status inline
  (*holds / contested / known false*).

A note on the tensions the book reports rather than hides (Chapter 2's
self-consistency section): they are of three kinds, and only one would be a
defect. *Logical inconsistencies within one model* — none; the consistency check
verifies clearing, binding budgets, and a positive kernel. *Seams between
deliberately incompatible idealisations* used in different blocks (e.g. the
negative prices allowed by the tractable Gaussian microstructure) — the real
limit, intrinsic to modelling, since no single tractable model is at once a
complete-markets equilibrium, a noisy rational-expectations economy, and a
limits-to-arbitrage economy. *Genuinely open empirical problems* (the elasticity
clash) — correctly left open, because the field has not settled them and faking a
resolution would be the true cop-out.
"""

APPENDIX = """
## A.1 — How this book was built

The volume is generated by a reproducible pipeline (`docs/book/build_book.py`):
source Markdown is parsed, LaTeX math is pre-rendered to HTML with **KaTeX**
(run under Node), code is highlighted with **Pygments**, the original figures are
produced by **matplotlib** (`docs/book/generate_book_figures.py`), and the whole
is styled with an academic stylesheet and rendered to PDF by **WeasyPrint**.
WeasyPrint has no JavaScript or MathML engine, so the KaTeX-to-HTML step is what
makes the dense mathematics render; the choice was validated by rasterising a
sample page and inspecting it before the full build.

## A.2 — Tooling limits encountered

The research for Chapter 4 was carried out by parallel web-search agents. During
that work, direct page-fetching was blocked (uniform HTTP 403), so the agents
grounded their claims on search-engine extractions of the primary papers rather
than full-text reads. Every load-bearing figure was then re-verified in a
separate pass; two citations initially flagged as uncertain — a recent
foundation-models-in-finance study and a controlled deep-learning comparison —
were confirmed on that second pass and upgraded, and one preprint that could not
be confirmed was demoted from anchor to corroboration. Precise decimals that
remained unconfirmed are flagged in Chapter 4 itself. The container also began
with no scientific Python stack; numpy, scipy, pandas, and the build tools were
installed during the session.

## A.3 — A leak caught in our own demo

Chapter 5's discipline is not hypothetical. The first version of the
significance-gate demonstration computed the deflated Sharpe ratio on an
*in-sample* refit of the model. On a *null* panel — data with no planted signal —
that version reported a deflated Sharpe of 1.000, "significant": the model had
memorised noise, and the in-sample evaluation dutifully rewarded it. This is
exactly the look-ahead leakage the book warns against, reproduced accidentally in
the book's own code. The fix was to run the gate on the **out-of-sample** rank-IC
stream produced by the purged cross-validation; the null then correctly collapsed
to a deflated Sharpe of 0.000. The episode is the strongest possible argument for
the verification gate: the leak is easy to introduce, it produces a confident and
entirely false number, and only out-of-sample evaluation exposes it.

## A.4 — Reproducibility

All sources, figures, code, and this builder live in the repository under
`docs/` and `vpts/`. The implemented model and significance gate ship with 15
unit tests (signal-detection and null-clearing for every estimator); the full
suite is 150 offline, deterministic tests.
"""

CHAPTERS = [
    dict(id="ch1", num="1", title="Introduction — Four Artefacts, One Argument",
         abstract="The session's thread and the unifying thesis.",
         lead=("fig9_session_map.png", "1.1",
               "The four technical artefacts of the session and the single argument that unifies them."),
         body=None, src=DOCS / "SYNTHESIS.md", strip_title=True, inline=[]),
    dict(id="ch2", num="2", title="A General-Equilibrium Model of a Market and Its Participants",
         abstract="Levels 1–4: from a static exchange economy to options, the risk-neutral density, and the non-identification of the physical measure.",
         lead=("fig1_pq_wedge.png", "2.1",
               "The risk-neutral density is the physical density tilted toward bad states by marginal utility — a valuation, not a forecast."),
         src=DOCS / "MARKET_EQUILIBRIUM_MODEL.md", strip_title=True,
         inline=[("## III.L6 — Aggregate dynamics and the pricing kernel",
                  figure("fig8_gbm_vs_jump.png", "2.2",
                         "Geometric Brownian motion is the measure-zero corner: the real world is jumps, stochastic volatility, and heavy tails.", "inline")),
                 ("## IV.L6 — The kernel, recovery, and the implication for $\\mathbb P$",
                  figure("fig2_pricing_kernel.png", "2.3",
                         "The empirical pricing kernel is the ratio of the two densities; its observed non-monotonicity is the pricing-kernel puzzle.", "inline"))]),
    dict(id="ch3", num="3", title="The Mathematics of Analysing Financial Time Series",
         abstract="From stationarity and the Wold decomposition to the purged-CV and deflated-Sharpe evaluation gate.",
         lead=("fig5_cpcv.png", "3.1",
               "Combinatorial purged cross-validation: contiguous groups, every k-subset as the test set, with purging and embargo to stop label leakage."),
         src=DOCS / "TIMESERIES_MATH.md", strip_title=True,
         inline=[("## §10 — Forecast evaluation and the overfitting problem (the core of the harness)",
                  figure("fig12_insample_oos.png", "3.2",
                         "Overfitting made visceral: the same strategy soars on the data it was tuned to and collapses the moment it meets data it has never seen. The vertical line is where the backtest ends and reality begins.", "inline")
                  + figure("fig7_deflated_sharpe.png", "3.3",
                         "Multiple testing inflates the significance hurdle: the benchmark Sharpe grows with the number of trials, so an observed Sharpe loses significance as the search widens.", "inline"))]),
    dict(id="ch4", num="4", title="AI Architectures for Directional Forecasting — a SWOT",
         abstract="An evidence-based survey anchored to directional accuracy: ~50–55% across every model family, and why.",
         lead=("fig3_directional_accuracy.png", "4.1",
               "Realistic out-of-sample directional accuracy by model family — all near coin-flip, with the up-rate (not 50%) as the real baseline."),
         src=DOCS / "AI_TIMESERIES_FORECASTING_SWOT.md", strip_title=True,
         inline=[("## 3. Cross-model synthesis",
                  figure("fig4_bias_variance.png", "4.2",
                         "On a low-SNR target the irreducible-noise floor is high, so the variance term dominates and the optimal model is simple — why bigger models lose.", "inline"))]),
    dict(id="ch5", num="5", title="From Theory to Code — the Model and the Significance Gate",
         abstract="The defensible recipe as dependency-free, unit-tested code, with a live demonstration.",
         lead=("fig6_gbrt_demo.png", "5.1",
               "The implemented recipe on synthetic data: with a planted signal it is found; with none, the OOS IC, the directional edge, and the deflated Sharpe all collapse."),
         body="CH5", src=None, strip_title=False, inline=[]),
    dict(id="ch6", num="6", title="The Positive Theory of Edge — Where Skill Comes From",
         abstract="The Fundamental Law (IR = IC·√breadth), breadth, capacity and decay, Kelly sizing, and why edges survive — the constructive counterpart to the failure modes.",
         lead=("fig10_fundamental_law.png", "6.1",
               "The Fundamental Law of Active Management: a per-bet edge of IC≈0.05 — the ~52–53% directional level the empirical chapters call near-futile — becomes an institutional information ratio once multiplied by breadth. The pessimistic and the constructive fact are the same fact at two scales."),
         src=DOCS / "EDGE_METHODOLOGY.md", strip_title=True,
         inline=[("## 2. Breadth is harder than it looks",
                  figure("fig11_lln_convergence.png", "6.2",
                         "The same law as a simulation: gamblers with a tiny 53% edge scatter over a few bets but fan into reliable profit over thousands — and the share in profit climbs toward certainty with breadth. A small edge industrialised is the engine of every statistical-arbitrage fund.", "inline"))]),
    dict(id="ch7", num="7", title="Synthesis — and the Boundary of the Whole Enterprise",
         abstract="The seven cross-cutting insights and what no method can repair.",
         body="CH6", src=None, strip_title=False, inline=[]),
]

CH5_BODY = """
Insights from the survey are not advice alone; they are implemented in this
repository as a runnable, tested recipe. Three components turn the theory into code.

**A dependency-free gradient-boosted-trees learner** (`vpts/ml/gbrt.py`). The
survey's best-evidence family for noisy, medium-N tabular financial data is tree
ensembles. Rather than add a heavy dependency, the learner is written in plain
NumPy, following the standard squared-error gradient-boosting recipe: start from
the mean, then iteratively fit shallow regression trees to the residual and add a
shrunk step. The split search is vectorised:

```python
def _best_split(X, y, min_leaf):
    n, p = X.shape
    parent_sse = float((y * y).sum()) - y.sum() ** 2 / n
    best_feat, best_thr, best_gain = -1, 0.0, 0.0
    for f in range(p):
        order = np.argsort(X[:, f], kind="mergesort")
        xs, ys = X[order, f], y[order]
        csum, csum2 = np.cumsum(ys), np.cumsum(ys * ys)
        nl = np.arange(1, n, dtype=float)
        sl, sr = csum[:-1], csum[-1] - csum[:-1]
        sse_l = csum2[:-1] - sl * sl / nl
        sse_r = (csum2[-1] - csum2[:-1]) - sr * sr / (n - nl)
        gain = parent_sse - (sse_l + sse_r)        # variance reduction
        ...
```

**A cross-sectional evaluator** (`gbrt_cross_sectional_eval`) that plugs the
learner into the existing purged + embargoed combinatorial cross-validation and —
crucially — scores directional accuracy against the **unconditional up-rate**,
never against 50%.

**A significance gate** (`vpts/ml/significance.py`) implementing the deflated and
probabilistic Sharpe ratios (which correct a Sharpe for non-normality, sample
length, and the *number of strategy trials*) and a directional-skill measure that
reports accuracy minus the majority-direction baseline. The deflated Sharpe:

$$\\text{DSR}=\\Phi\\!\\left(\\frac{(\\widehat{\\text{SR}}-\\text{SR}_0)\\sqrt{T-1}}{\\sqrt{1-\\gamma_3\\widehat{\\text{SR}}+\\frac{\\gamma_4-1}{4}\\widehat{\\text{SR}}^2}}\\right),\\qquad \\text{SR}_0=\\sigma_{\\text{SR}}\\Big[(1-\\gamma_E)\\Phi^{-1}\\!\\big(1-\\tfrac1M\\big)+\\gamma_E\\Phi^{-1}\\!\\big(1-\\tfrac1{Me}\\big)\\Big].$$

**What the demonstration shows.** Run end-to-end on synthetic data with a known
signal strength, the harness behaves exactly as the theory predicts. With a real
signal planted (β = 0.6) the out-of-sample rank IC is +0.37, directional accuracy
is 63.6% against an up-rate of 50.5% (a +13-point edge), and the deflated Sharpe
is 1.000 — significant after a 50-trial correction. With the signal switched off
(β = 0) all three collapse: IC ≈ 0, directional accuracy 50.1% versus 50.0%, and
the deflated Sharpe is **0.000**. The gate runs on the out-of-sample IC stream,
not an in-sample refit — which, as the methodology appendix recounts, is the whole
point: an earlier in-sample version let the null falsely pass.

| Metric | Signal (β = 0.6) | Null (β = 0) |
|---|---|---|
| OOS rank IC (mean) | +0.369 | −0.007 |
| Directional accuracy | 63.6% (up-rate 50.5%) | 50.1% (up-rate 50.0%) |
| Edge over baseline | +13.1 pp | +0.1 pp |
| Deflated Sharpe | 1.000 — significant | 0.000 — not significant |
"""

CH6_BODY = """
## The seven cross-cutting insights

**1. The two halves are one theorem from two sides.** The asset-pricing model's
deepest result — the kernel exists but is *not identified* from prices, because
the stochastic discount factor's permanent component is unrecoverable — is the
same wall the machine-learning models hit as the ~50–55% directional ceiling. To
turn a price into a forecast you need the kernel; the kernel is not recoverable;
so a model "predicting returns" cannot separate the market's belief from the risk
premium. The non-identification of the physical measure *is* the coin-flip wall.

**2. "Price ≠ probability" and "accuracy ≠ edge" are the same caution.** A
51%-accurate model posting a large pre-cost Sharpe is the empirical shadow of the
theoretical point that the risk-neutral density is a valuation, not a forecast.
Hit-rate is nearly orthogonal to profit; sizing and ranking over the risk-weighted
distribution are what pay.

**3. The binding constraint is identification and data — never model capacity.**
The between-family accuracy spread (~50–55%) is smaller than the leaked-versus-clean
evaluation spread within any one family. More model is the wrong lever; better
data, honest evaluation, and naming what is structurally unknowable are the right
ones.

**4. Bias–variance, made economic.** On a near-random-walk, low signal-to-noise
target, flexible models spend their capacity fitting noise — which is why
transformers and deep reinforcement learning lose to trees and regularised
baselines. "Geometric Brownian motion is a measure-zero corner" is the
continuous-time twin of the same point.

**5. The endogenous-kernel rule and the no-look-ahead rule are one discipline.**
Both forbid smuggling in the answer: do not exogenise the discount rate (you would
assume the risk premium you are explaining); do not let future data touch training
(you would assume the prediction you are claiming). Purged cross-validation is the
econometric image of the endogenous-kernel requirement.

**6. Where edge plausibly lives is consistent across both analyses:** relative
(cross-sectional ranking) over absolute (directional timing); risk premia
harvested knowingly rather than mistaken for alpha; structural features over
generic price-pattern learning. And the honest ceiling is low and decaying — a
signal, once traded, erodes.

**7. The verification gate is not bureaucracy.** Backtest overfitting yields
*negative* out-of-sample returns, not merely zero, so the naive number is
guaranteed to look good and be false. The gate — purged cross-validation, the
up-rate baseline, transaction costs, the deflated Sharpe and its kin, point-in-time
data, and reporting profit-and-loss rather than accuracy — is the only thing
standing between an analyst and a confidently wrong result.

## The constructive program — and why "the market is unbeatable" is false

The purpose of this volume is constructive: to find the right model, and the
sector, regime, horizon, and usage where it works. The criticality is a means to
that end — in a domain that punishes credulity, an unsparing filter protects
scarce time and capital, because a hallucinated edge trusted or a dead method
pursued costs more than the filter that would have caught it.

That criticality must not curdle into the false dogma that **the market is
unbeatable**, or that **no one can consistently beat it**. Those statements are
true only in an idealised, frictionless, stationary market that does not exist —
and, as Keynes warned, "in the long run," in which "we are all dead." In the
real, finite, frictional, non-stationary market they are false, and the
counterexamples are decisive:

- **Renaissance's Medallion Fund** compounded at roughly 63% gross (about 39% net
  of its 5/44 fees) from 1988 to 2018, without a single losing year across the
  dot-com crash and the financial crisis, and — decisively — with *negative*
  market beta and *negative* factor loadings, so its returns provably cannot be a
  premium for bearing risk. Cornell (2019, *Journal of Portfolio Management*)
  calls it, correctly, "the ultimate counterexample" to market efficiency: there
  is no adequate rational-market explanation. This is genuine, unexplained alpha.
- **Warren Buffett** beat the market for sixty years, with a Sharpe ratio near
  0.79 — roughly double the market's. Frazzini, Kabiller and Pedersen (2018,
  *Financial Analysts Journal*) show the alpha is *explainable* — leverage of
  about 1.7×, applied to cheap, safe, high-quality stocks (the betting-against-beta
  and quality-minus-junk factors) — but explainable is not the same as available:
  he found and sized those durable edges *decades before they were named*, and
  financed them with patient, near-zero-cost insurance float.

These are two different species of beating the market, and the distinction is the
whole point of the book. Medallion is **true anomaly** — real, unexplained,
short-horizon edge in a specific niche, *capacity-constrained* (the fund is capped
near \\$10B and returns outside capital; Renaissance's funds open to outsiders have
not replicated it) and *fiercely defended* against decay and imitation. Buffett is
**premium-harvesting done supremely well** — a durable, identifiable edge, found
early, sized large with cheap leverage, and held with permanent capital through
the decades it took to pay off. Both *exemplify* this book's constructive core
rather than contradict it: the claim was never that edge is impossible, but that
it is **rare, specific, real-when-it-exists, and perishable**. Medallion and
Buffett are what success under that description looks like.

The base rate is not the ceiling. The ~50–55% directional accuracy, the fragility,
the decay — those describe the median attempt and the naive method, the bar
against which any new claim must be judged. Skill in this domain is real and
*heavy-tailed*: that ~80–90% of active managers underperform over fifteen years
and that Medallion exists are *both* true. The honest posture holds both — most
fail, a few persistently win — and this book's severity is in service of reaching
the thin right tail, not of denying it exists.

And the joint-hypothesis problem cuts **both ways**. It disciplines edge-claims:
apparent predictability may be a risk premium or an artifact. But it disciplines
*efficiency*-claims with exactly equal force — because no test of efficiency is
independent of the asset-pricing model assumed, market efficiency is not a proven
null but an untestable maintained hypothesis. Cornell's Medallion result is the
sharper form: with negative factor loadings there is no risk model under which its
returns are fair compensation, so the efficiency null is not merely untested
there — it is refuted. The asymmetry the sceptic smuggles in ("treat efficiency as
the default and make edge prove itself") is unjustified. Both must prove
themselves, and sometimes, demonstrably, edge wins.

## The boundary of the whole enterprise

What no method in this book can repair: **non-stationarity** (every estimator
assumes a law that does not move; structural breaks are that assumption failing);
**reflexivity and signal decay** (the map decays once it is measured and traded);
**Knightian uncertainty** (the agents and the econometrician alike assume the
probability law is known, including the tails that may never have been sampled);
**the joint-hypothesis problem** (no test of predictability is independent of the
model used to measure it); and **equilibrium and model multiplicity** (selection
is a convention, not a result). These are not gaps to be closed with more data or
a larger model. They are the edge of the map — and naming them precisely is the
most honest thing a quantitative framework can do.

*End of volume.*
"""


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def katex_css() -> str:
    css = (DIST / "katex.min.css").read_text()
    css = re.sub(r"@font-face\s*\{[^}]*\}", "", css)
    faces = []
    for ttf in sorted((DIST / "fonts").glob("*.ttf")):
        fam, _, variant = ttf.stem.partition("-")
        faces.append(f"@font-face{{font-family:'{fam}';src:url('file://{ttf}');"
                     f"font-weight:{'bold' if 'Bold' in variant else 'normal'};"
                     f"font-style:{'italic' if 'Italic' in variant else 'normal'};}}")
    return "\n".join(faces) + "\n" + css


STYLE = """
@page {
  size: A4; margin: 22mm 18mm 20mm 18mm;
  @top-left { content: "A Session Compendium · Asset Pricing, Time Series, and the Limits of Prediction";
    font: 7.5pt Georgia; color: #9aa3b2; }
  @bottom-center { content: counter(page); font: 8pt Georgia; color: #6b7382; }
}
@page :first { margin: 0; @top-left { content: ""; } @bottom-center { content: ""; } }
@page cover { margin: 0; }
html { font-size: 10.4pt; }
body { font-family: Georgia, 'Times New Roman', serif; color: #1c2330; line-height: 1.46; }
h1, h2, h3, h4 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #16263a; line-height: 1.2; }
h1 { font-size: 16pt; page-break-before: always; border-bottom: 2px solid #1a3a5c; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 12.5pt; margin-top: 1.3em; color: #1a3a5c; }
h3 { font-size: 11pt; margin-top: 1.1em; }
h4 { font-size: 10.2pt; }
p { margin: 0.5em 0; text-align: justify; }
a { color: #1a5f9e; text-decoration: none; }
strong { color: #16263a; }
code { font-family: 'DejaVu Sans Mono', Menlo, monospace; font-size: 8.6pt;
  background: #f3f5f8; padding: 0.5px 3px; border-radius: 3px; }
.codehl { background: #f6f8fa; border: 1px solid #e2e8f0; border-left: 3px solid #1a3a5c;
  border-radius: 4px; padding: 7px 10px; margin: 9px 0; font-size: 8.3pt; overflow-wrap: anywhere; }
.codehl pre { margin: 0; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 8.8pt; }
th, td { border: 1px solid #cdd5e0; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-family: Helvetica, Arial, sans-serif; }
.eq { margin: 9px 0; text-align: center; overflow-wrap: anywhere; }
.eq .katex-display { margin: 0; }
.katex { font-size: 1.0em; }
.mi .katex { font-size: 0.98em; }
blockquote { border-left: 3px solid #b8860b; margin: 9px 0; padding: 2px 12px; color: #44505f; background:#fbf8f0; }
hr { border: none; border-top: 1px solid #d6dde6; margin: 14px 0; }
figure { margin: 14px auto; text-align: center; page-break-inside: avoid; }
figure.lead { margin: 6px auto 16px; }
figure img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }
figure.inline img { max-width: 86%; }
figcaption { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; color: #55606f;
  margin-top: 5px; text-align: center; }

/* cover */
.cover { page: cover; height: 297mm; box-sizing: border-box; padding: 46mm 26mm 24mm;
  background: linear-gradient(160deg, #16263a 0%, #1a3a5c 46%, #24506f 100%); color: #fff;
  page-break-after: always; }
.cover .rule { width: 64px; height: 4px; background: #b8860b; margin: 0 0 26px; }
.cover h1 { font-size: 30pt; border: none; page-break-before: avoid; color: #fff; margin: 0 0 10px;
  line-height: 1.12; }
.cover .sub { font-size: 13pt; color: #cdd9e6; font-family: Helvetica, Arial, sans-serif; margin-bottom: 32px; }
.cover .by { font-size: 10pt; color: #9fb4c8; font-family: Helvetica, Arial, sans-serif; }
.cover .tag { position: absolute; }
.cover ul { color: #dce6f0; font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.7;
  list-style: none; padding: 0; margin-top: 30px; }
.cover ul li::before { content: "—  "; color: #b8860b; }

/* front matter */
.frontmatter { page-break-after: always; }
.frontmatter h1 { page-break-before: avoid; }
.toc h1 { page-break-before: always; }
.toc ol { list-style: none; padding: 0; font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; }
.toc li { margin: 9px 0; border-bottom: 1px dotted #c7cfda; padding-bottom: 5px; }
.toc a { color: #16263a; }
.toc .cnum { color: #b8860b; font-weight: bold; margin-right: 8px; }
.toc a::after { content: target-counter(attr(href), page); float: right; color: #6b7382; }
.toc .cabs { display: block; color: #6b7382; font-size: 8.6pt; font-style: italic; margin-top: 2px; }

/* chapter plate */
.chapter { page-break-before: always; }
.chapter h1 { page-break-before: avoid; }
.chapter .kicker { font-family: Helvetica, Arial, sans-serif; font-size: 9pt; letter-spacing: .16em;
  text-transform: uppercase; color: #b8860b; font-weight: 700; }
.chapter .cabstract { font-style: italic; color: #44505f; margin: 4px 0 8px; }
.section-body h1 { /* source-level '# Part' headings already styled by h1 */ }
"""


def build():
    css = STYLE + "\n" + katex_css()

    # cover
    cover = f"""
    <section class="cover">
      <div class="rule"></div>
      <h1>Asset Pricing, Time Series,<br/>and the Limits of Prediction</h1>
      <div class="sub">A scientific compendium of one research session — from
        general-equilibrium theory to tested, dependency-free code</div>
      <ul>
        <li>A four-level model of a market and its participants</li>
        <li>The mathematics of analysing financial time series</li>
        <li>A SWOT of AI architectures for directional forecasting</li>
        <li>An implemented model with an honest significance gate</li>
        <li>A positive theory of where edge comes from — and why it survives</li>
      </ul>
      <div style="position:absolute; bottom:24mm;">
        <div class="by">Compiled by Claude Code &nbsp;·&nbsp; {DATE}</div>
        <div class="by" style="color:#7e94a8; margin-top:4px;">Research &amp; education — not financial advice</div>
      </div>
    </section>"""

    # front matter (plain-language summary + preface + reader's guide)
    primer = re.sub(r"^#\s+.*\n", "", (DOCS / "PRIMER.md").read_text(), count=1)
    summary = f'<section class="frontmatter"><h1>Plain-Language Summary</h1>{md_to_html(primer)}</section>'
    front = f'<section class="frontmatter"><h1>Preface</h1>{md_to_html(PREFACE)}</section>'
    guide = f'<section class="frontmatter"><h1>Reader’s Guide</h1>{md_to_html(READER_GUIDE)}</section>'

    # toc
    toc_items = "".join(
        f'<li><span class="cnum">{c["num"]}</span>'
        f'<a href="#{c["id"]}">{c["title"]}</a>'
        f'<span class="cabs">{c["abstract"]}</span></li>'
        for c in CHAPTERS)
    toc_items += ('<li><span class="cnum">A</span>'
                  '<a href="#appendix">Appendix — Methodology, Tooling, and a Caught Leak</a></li>')
    toc = f'<section class="toc"><h1>Contents</h1><ol>{toc_items}</ol></section>'

    # chapters
    parts = [cover, summary, front, guide, toc]
    for c in CHAPTERS:
        if c["src"]:
            text = c["src"].read_text()
            if c.get("strip_title"):
                text = re.sub(r"^#\s+.*\n", "", text, count=1)
            for anchor, fightml in c["inline"]:
                text = text.replace(anchor, fightml + "\n\n" + anchor, 1)
            body_html = md_to_html(text)
        else:
            body_html = md_to_html(CH5_BODY if c["body"] == "CH5" else CH6_BODY)
        lead = figure(c["lead"][0], c["lead"][1], c["lead"][2], "lead") if c.get("lead") else ""
        plate = (f'<section class="chapter" id="{c["id"]}">'
                 f'<div class="kicker">Chapter {c["num"]}</div>'
                 f'<h1>{c["title"]}</h1>'
                 f'<div class="cabstract">{c["abstract"]}</div>'
                 f'{lead}<div class="section-body">{body_html}</div></section>')
        parts.append(plate)

    parts.append(f'<section class="chapter" id="appendix"><div class="kicker">Appendix A</div>'
                 f'<h1>Methodology, Tooling, and a Caught Leak</h1>'
                 f'<div class="section-body">{md_to_html(APPENDIX)}</div></section>')

    doc = (f'<!doctype html><html><head><meta charset="utf-8">'
           f'<style>{css}\n{HtmlFormatter(cssclass="codehl").get_style_defs()}</style>'
           f'</head><body>{"".join(parts)}</body></html>')

    (BOOK / "_build").mkdir(exist_ok=True)
    (BOOK / "_build" / "book.html").write_text(doc)
    HTML(string=doc, base_url=str(BOOK)).write_pdf(str(OUT))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    build()
