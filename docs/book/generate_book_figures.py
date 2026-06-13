"""Generate the original figures for the combined session book (print-style PNGs).

Matplotlib (Agg), white background, academic palette. Each figure illustrates one
load-bearing idea from the session. Run:

    python docs/book/generate_book_figures.py     # writes figN_*.png into docs/book/img/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)

NAVY, MAROON, GREY, GOLD, GREEN, SLATE = "#1a3a5c", "#8c2d2d", "#8a93a3", "#b8860b", "#2e6e4e", "#34495e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.edgecolor": "#444",
    "axes.linewidth": 0.8, "figure.dpi": 170, "savefig.dpi": 170,
    "axes.titlesize": 12.5, "axes.titleweight": "bold",
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig_pq_wedge():
    x = np.linspace(60, 140, 600)
    fP = np.exp(-0.5 * ((x - 103) / 11) ** 2)
    fP /= np.trapezoid(fP, x)
    m = np.exp(-0.045 * (x - 100))                 # SDF: decreasing in wealth
    fQ = fP * m
    fQ /= np.trapezoid(fQ, x)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(x, fP, color=NAVY, lw=2.2, label=r"physical density  $f^{\mathbb{P}}$  (beliefs)")
    ax.plot(x, fQ, color=MAROON, lw=2.2, label=r"risk-neutral density  $f^{\mathbb{Q}}$  (prices)")
    crash = x < 85
    ax.fill_between(x[crash], fP[crash], color=NAVY, alpha=0.10)
    ax.fill_between(x[crash], fQ[crash], color=MAROON, alpha=0.18)
    ax.axvline(100, color=GREY, ls=":", lw=1)
    ax.annotate("crash region:\n" + r"$\mathbb{Q}$(crash) > $\mathbb{P}$(crash)", (78, 0.004),
                fontsize=9.5, color=MAROON, ha="center")
    ax2 = ax.twinx()
    ax2.plot(x, m, color=GOLD, lw=1.6, ls="--", label="pricing kernel  m(x)")
    ax2.set_ylabel("kernel  m(x)  =  marginal utility", color=GOLD)
    ax2.tick_params(axis="y", colors=GOLD)
    ax.set_xlabel("future price / wealth  x")
    ax.set_ylabel("density")
    ax.set_title("The $\\mathbb{P}$ / $\\mathbb{Q}$ wedge — a price is a risk-weighted valuation, not a probability")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.6, loc="upper right", framealpha=0.9)
    ax.set_yticks([])
    _save(fig, "fig1_pq_wedge.png")


def fig_pricing_kernel():
    x = np.linspace(-0.25, 0.25, 400)               # index return
    theory = np.exp(-3.2 * x)                        # monotone decreasing (risk aversion)
    puzzle = np.exp(-3.2 * x) * (1 + 0.9 * np.exp(-((x - 0.0) / 0.04) ** 2)) * 0.0 + \
        (0.9 + 6.0 * x ** 2 - 1.5 * x)               # U-shaped empirical EPK
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax.plot(x * 100, theory / theory.mean(), color=NAVY, lw=2.2,
            label="theory: monotone-decreasing kernel (risk aversion)")
    ax.plot(x * 100, puzzle / puzzle.mean(), color=MAROON, lw=2.2, ls="--",
            label="empirical EPK: non-monotone (the pricing-kernel puzzle)")
    ax.axhline(1, color=GREY, ls=":", lw=1)
    ax.set_xlabel("index return  (%)")
    ax.set_ylabel(r"$m(x)=B\,f^{\mathbb{Q}}/f^{\mathbb{P}}$  (normalised)")
    ax.set_title("The kernel IS the gap between the two densities")
    ax.legend(fontsize=8.8, loc="upper center")
    _save(fig, "fig2_pricing_kernel.png")


def fig_directional_accuracy():
    fams = ["Regularized\nlinear/logistic", "Gradient-boosted\ntrees", "LSTM / GRU / TCN",
            "Transformers\n(PatchTST etc.)", "N-BEATS / TFT", "Foundation models\n(zero-shot)"]
    acc = [52.0, 53.5, 51.5, 50.2, 50.3, 51.0]
    err = [1.5, 1.6, 1.6, 1.3, 1.3, 0.8]
    y = np.arange(len(fams))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axvspan(50, 55, color=NAVY, alpha=0.06, label="honest OOS band (50–55%)")
    ax.barh(y, acc, xerr=err, color=NAVY, alpha=0.85, height=0.6,
            error_kw=dict(ecolor=SLATE, lw=1))
    ax.axvline(50, color=GREY, ls=":", lw=1.2)
    ax.axvline(54, color=MAROON, ls="--", lw=1.4)
    ax.text(54.05, 5.4, "up-rate baseline ~53–55%\n(the real bar, not 50%)",
            color=MAROON, fontsize=8.4, va="top")
    ax.axvspan(60, 66, color=MAROON, alpha=0.08)
    ax.text(63, 0.0, ">60% =\nleakage zone", color=MAROON, fontsize=8.2, ha="center", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(fams, fontsize=8.6)
    ax.set_xlim(48, 66)
    ax.set_xlabel("out-of-sample directional accuracy  (%)")
    ax.set_title("Directional accuracy by model family — all near coin-flip")
    ax.legend(fontsize=8.2, loc="lower right")
    _save(fig, "fig3_directional_accuracy.png")


def fig_bias_variance():
    c = np.linspace(0.3, 10, 400)                   # model complexity
    bias2 = 3.0 / c
    var = 0.10 * c
    noise = 2.4                                      # HIGH irreducible noise (low SNR)
    total = bias2 + var + noise
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(c, bias2, color=NAVY, lw=1.8, label="bias$^2$")
    ax.plot(c, var, color=MAROON, lw=1.8, label="variance")
    ax.axhline(noise, color=GREY, lw=1.6, ls=":", label="irreducible noise (low-SNR floor)")
    ax.plot(c, total, color="black", lw=2.4, label="total test error")
    imin = int(np.argmin(total))
    ax.axvline(c[imin], color=GOLD, ls="--", lw=1.4)
    ax.annotate("optimum is SIMPLE\n(variance dominates here)", (c[imin] + 0.3, total[imin] + 0.6),
                fontsize=8.6, color=GOLD)
    ax.set_xlabel("model complexity / flexibility  →")
    ax.set_ylabel("expected test error")
    ax.set_title("Bias–variance on a low-SNR target: why bigger models lose")
    ax.legend(fontsize=8.6, loc="upper center", ncol=2)
    ax.set_ylim(0, 8)
    _save(fig, "fig4_bias_variance.png")


def fig_cpcv():
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    N = 6
    w = 1.0
    test = {1, 4}
    for g in range(N):
        x0 = g * (w + 0.08)
        if g in test:
            ax.add_patch(FancyBboxPatch((x0, 0.4), w, 0.5, boxstyle="round,pad=0.01",
                                        fc=MAROON, ec="none", alpha=0.85))
            ax.text(x0 + w / 2, 0.65, "TEST", color="white", ha="center", va="center", fontsize=8, weight="bold")
            # purge before, embargo after
            ax.add_patch(plt.Rectangle((x0 - 0.22, 0.4), 0.22, 0.5, fc="none", ec=GOLD, hatch="////", lw=0.8))
            ax.add_patch(plt.Rectangle((x0 + w, 0.4), 0.18, 0.5, fc=GREEN, ec="none", alpha=0.5))
        else:
            ax.add_patch(FancyBboxPatch((x0, 0.4), w, 0.5, boxstyle="round,pad=0.01",
                                        fc=NAVY, ec="none", alpha=0.8))
            ax.text(x0 + w / 2, 0.65, "train", color="white", ha="center", va="center", fontsize=8)
        ax.text(x0 + w / 2, 0.32, f"group {g}", ha="center", va="top", fontsize=7.5, color=SLATE)
    ax.text(0, 1.12, "Combinatorial Purged CV  (one split shown):  "
            r"$\binom{N}{k}$ splits,  $\binom{N-1}{k-1}$ OOS paths",
            fontsize=10.5, weight="bold")
    ax.add_patch(plt.Rectangle((6.6, 0.95), 0.22, 0.12, fc="none", ec=GOLD, hatch="////", lw=0.8))
    ax.text(6.9, 1.01, "purge (label overlap removed)", fontsize=7.6, va="center")
    ax.add_patch(plt.Rectangle((6.6, 0.78), 0.22, 0.12, fc=GREEN, ec="none", alpha=0.5))
    ax.text(6.9, 0.84, "embargo (serial-corr. bleed removed)", fontsize=7.6, va="center")
    ax.set_xlim(-0.4, 11)
    ax.set_ylim(0.2, 1.25)
    ax.axis("off")
    _save(fig, "fig5_cpcv.png")


def fig_gbrt_demo():
    rng = np.random.default_rng(0)
    sig = rng.normal(0.369, 0.142, 5000)
    nul = rng.normal(-0.007, 0.157, 5000)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.hist(nul, bins=40, color=GREY, alpha=0.6, density=True, label="null (β=0)")
    ax.hist(sig, bins=40, color=NAVY, alpha=0.7, density=True, label="signal (β=0.6)")
    ax.axvline(0, color="black", lw=1, ls=":")
    ax.set_title("OOS per-date rank IC", fontsize=10.5)
    ax.set_xlabel("IC"); ax.set_yticks([]); ax.legend(fontsize=7.6)
    ax = axes[1]
    ax.bar([0, 1], [63.6, 50.1], color=[NAVY, GREY], width=0.6)
    ax.axhline(54, color=MAROON, ls="--", lw=1.3)
    ax.text(1.5, 54.4, "up-rate", color=MAROON, fontsize=7.8, ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["signal", "null"])
    ax.set_ylim(45, 70); ax.set_ylabel("dir. accuracy (%)")
    ax.set_title("Directional accuracy", fontsize=10.5)
    ax = axes[2]
    ax.bar([0, 1], [1.0, 0.0], color=[GREEN, MAROON], width=0.6)
    ax.axhline(0.95, color="black", ls=":", lw=1)
    ax.text(1.5, 0.96, "0.95", fontsize=7.8, ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["signal", "null"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("deflated Sharpe")
    ax.set_title("Significance gate", fontsize=10.5)
    fig.suptitle("The implemented recipe: signal is found, the null collapses on all three metrics",
                 fontsize=11, weight="bold")
    _save(fig, "fig6_gbrt_demo.png")


def fig_deflated_sharpe():
    from scipy.stats import norm
    gE = 0.5772156649
    N = np.arange(1, 2000)
    sr_std = 0.05
    sr0 = sr_std * ((1 - gE) * norm.ppf(1 - 1.0 / N) + gE * norm.ppf(1 - 1.0 / (N * np.e)))
    sr0[0] = 0
    sr_obs = 0.12
    Tn = 500
    dsr = norm.cdf((sr_obs - sr0) * np.sqrt(Tn - 1))
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.plot(N, sr0, color=MAROON, lw=2, label="benchmark SR$_0$ = expected max of N trials")
    ax.axhline(sr_obs, color=NAVY, lw=1.6, ls="--", label="observed Sharpe (0.12)")
    ax.set_xscale("log")
    ax.set_xlabel("number of strategy trials  N  (log scale)")
    ax.set_ylabel("Sharpe (per obs)")
    ax.set_title("Multiple testing inflates the hurdle: SR$_0$ grows with the search")
    ax2 = ax.twinx()
    ax2.plot(N, dsr, color=GREEN, lw=1.6, label="deflated Sharpe (P[true > SR$_0$])")
    ax2.axhline(0.95, color=GREY, ls=":", lw=1)
    ax2.set_ylabel("deflated Sharpe", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    ax2.set_ylim(0, 1.05)
    cross = N[np.argmax(dsr < 0.95)]
    ax2.annotate(f"loses significance\nbeyond ~{cross} trials", (cross, 0.6), color=GREEN, fontsize=8.2)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    _save(fig, "fig7_deflated_sharpe.png")


def fig_gbm_vs_jump():
    rng = np.random.default_rng(3)
    T, dt = 252, 1 / 252
    t = np.arange(T) * dt
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    # GBM
    gbm = np.exp(np.cumsum((0.06 - 0.5 * 0.16) * dt + 0.4 * np.sqrt(dt) * rng.standard_normal(T)))
    ax.plot(t, gbm, color=NAVY, lw=1.6, label="GBM (smooth, constant vol)")
    # jump-diffusion + stoch vol
    v = 0.16 + np.zeros(T)
    rets = np.zeros(T)
    for i in range(1, T):
        v[i] = max(0.01, v[i - 1] + 5 * (0.16 - v[i - 1]) * dt + 0.5 * np.sqrt(v[i - 1] * dt) * rng.standard_normal())
        jump = rng.standard_normal() * 0.0
        if rng.random() < 0.02:
            jump = rng.normal(-0.06, 0.04)
        rets[i] = (0.06 - 0.5 * v[i]) * dt + np.sqrt(v[i] * dt) * rng.standard_normal() + jump
    jd = np.exp(np.cumsum(rets))
    ax.plot(t, jd, color=MAROON, lw=1.6, label="jump-diffusion + stochastic vol (the real world)")
    ax.set_xlabel("years"); ax.set_ylabel("price"); ax.legend(fontsize=8)
    ax.set_title("GBM is the measure-zero corner")
    # tails
    big = rng.standard_t(3, 20000) * 0.01
    gauss = rng.standard_normal(20000) * 0.01
    axh.hist(gauss, bins=80, color=NAVY, alpha=0.5, density=True, label="Gaussian")
    axh.hist(big, bins=200, color=MAROON, alpha=0.5, density=True, label="heavy-tailed (real)")
    axh.set_xlim(-0.05, 0.05); axh.set_yscale("log"); axh.set_yticks([])
    axh.set_xlabel("daily return"); axh.legend(fontsize=7.6)
    axh.set_title("tails", fontsize=10)
    _save(fig, "fig8_gbm_vs_jump.png")


def fig_session_map():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axis("off")
    boxes = [
        (0.5, 3.4, "Ch. 2\nAsset-pricing\nmodel (L1–L4)", NAVY),
        (0.5, 1.0, "Ch. 3\nTime-series\nanalysis math", NAVY),
        (6.2, 3.4, "Ch. 4\nAI/ML SWOT\n(directional)", MAROON),
        (6.2, 1.0, "Ch. 5\nGBRT + significance\ngate (code)", GREEN),
    ]
    for x, y, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.0, 1.3, boxstyle="round,pad=0.06",
                                    fc=col, ec="none", alpha=0.9))
        ax.text(x + 1.0, y + 0.65, txt, color="white", ha="center", va="center", fontsize=9, weight="bold")
    ax.add_patch(FancyBboxPatch((3.15, 2.05), 2.0, 1.5, boxstyle="round,pad=0.08",
                                fc="white", ec=GOLD, lw=2.2))
    ax.text(4.15, 2.8, "THE SPINE", color=GOLD, ha="center", fontsize=9.5, weight="bold")
    ax.text(4.15, 2.35, "existence ≠ identification\nprice ≠ probability\naccuracy ≠ edge",
            color=SLATE, ha="center", fontsize=8.2)
    for x, y in [(2.5, 4.05), (2.5, 1.65), (6.2, 4.05), (6.2, 1.65)]:
        tx = 3.15 if x < 4 else 5.15
        ax.add_patch(FancyArrowPatch((x, y), (tx, 2.8 if y > 2.5 else 2.8),
                                     arrowstyle="-", color=GOLD, lw=1.3, alpha=0.7,
                                     connectionstyle="arc3,rad=0.1"))
    ax.set_xlim(0, 8.4); ax.set_ylim(0.5, 5)
    ax.set_title("Session map — four artefacts, one argument", fontsize=12, weight="bold")
    _save(fig, "fig9_session_map.png")


def fig_fundamental_law():
    BR = np.logspace(0, 3.7, 400)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for ic, col, ls in [(0.02, GREY, ":"), (0.05, NAVY, "-"), (0.10, MAROON, "--"), (0.20, GREEN, "-.")]:
        ax.plot(BR, ic * np.sqrt(BR), color=col, ls=ls, lw=2, label=f"IC = {ic:.2f}")
    ax.axhline(1.0, color=GOLD, lw=1.4, alpha=0.85)
    ax.text(1.1, 1.04, "IR = 1.0  (institutional-grade)", color=GOLD, fontsize=8.4)
    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.set_xscale("log")
    ax.plot([1000], [0.05 * np.sqrt(1000)], "o", color=NAVY, ms=7)
    ax.annotate("IC 0.05  (≈53% directional)\n× breadth 1000  →  IR ≈ 1.6\nthe 'coin-flip' edge, industrialised",
                (1000, 1.58), (40, 1.75), fontsize=8.3, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlabel("breadth — number of independent bets per year  (log scale)")
    ax.set_ylabel("information ratio  IR = IC × √breadth")
    ax.set_title("The Fundamental Law: a tiny per-bet edge × breadth = a great fund")
    ax.legend(fontsize=8.6, loc="upper left", title="skill per bet (IC)")
    ax.set_ylim(0, 2.5)
    _save(fig, "fig10_fundamental_law.png")


if __name__ == "__main__":
    fig_pq_wedge()
    fig_pricing_kernel()
    fig_directional_accuracy()
    fig_bias_variance()
    fig_cpcv()
    fig_gbrt_demo()
    fig_deflated_sharpe()
    fig_gbm_vs_jump()
    fig_session_map()
    fig_fundamental_law()
    print("all figures written to", OUT)
