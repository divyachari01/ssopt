"""Paper figures from dataset/*.csv  (matplotlib only).

    python -m ssopt.figures --dataset dataset --out dataset/figures

fig1_rms_vs_Z        RMS and mean offset vs Z, one panel per isoelectronic sequence, per-l vs per-orbital
fig2_lambda_vs_Z     tuned per-orbital λ vs Z, one panel per orbital, one line per sequence
fig3_residuals       level residuals E_SS − E_NIST vs excitation energy, by sequence
fig4_convergence     best-so-far RMS vs number of structx runs, every ion (Sobol phase shaded)
fig5_avalues         A(SS)/A(NORAD) and A(SS)/A(NIST) distributions, split by the L/V flag
fig6_prune           ΔRMS of removing each correlation group, ion × group
fig7_predictor       RMS at the leave-one-ion-out predicted λ vs RMS at the tuned λ (needs verify_*.csv)

Sequences are ordinal (number of electrons), so they get a sequential colour
ramp and a direct label; the two tuning stages get two fixed hues.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGE = {"l": "#2a78d6", "orbital": "#eb6834"}     # per-l, per-orbital
NEUTRAL = "#52514e"
LV_OK = "#1baf7a"
LV_BAD = "#e34948"
ORB_RE = re.compile(r"(\d+[spdfg])=([-+\d.]+)")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a86",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "lines.linewidth": 1.4,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.constrained_layout.use": True,
})


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _seq_colors(seqs):
    """Sequential ramp over ordinal sequences (by electron number)."""
    order = sorted(seqs, key=lambda s: s[0])
    n = max(len(order), 2)
    cmap = plt.get_cmap("viridis")
    return {s: cmap(0.1 + 0.8 * i / (n - 1)) for i, s in enumerate(order)}


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"))
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)
    print("  ", name)


# ----------------------------------------------------------------------------
def fig_rms_vs_Z(ions, out):
    by = defaultdict(list)
    for r in ions:
        by[(int(r["N"]), r["sequence"])].append(r)
    keys = sorted(by)
    if not keys:
        return
    ncol = min(4, len(keys)); nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow), sharey=True, squeeze=False)
    for ax, key in zip(axes.flat, keys):
        rows = sorted(by[key], key=lambda r: int(r["Z"]))
        Z = [int(r["Z"]) for r in rows]
        for mode in ("l", "orbital"):
            rms = [_f(r.get(f"rms_{mode}")) for r in rows]
            off = [_f(r.get(f"mean_diff_{mode}")) for r in rows]
            ax.plot(Z, rms, "-o", ms=3.5, color=STAGE[mode], label=f"per-{mode} RMS")
            ax.plot(Z, np.abs(off), "--", lw=1, color=STAGE[mode], alpha=0.7, label=f"per-{mode} |offset|")
        ax.set_title(f"{key[1]}-like (N={key[0]})", loc="left")
        ax.set_xlabel("Z")
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Ry")
    axes.flat[0].legend(frameon=False, loc="upper left")
    fig.suptitle("RMS deviation from NIST (solid) and mean offset (dashed) along isoelectronic sequences",
                 x=0.01, ha="left", fontsize=10)
    _save(fig, out, "fig1_rms_vs_Z")


def fig_lambda_vs_Z(ions, out, mode="orbital"):
    pts = defaultdict(lambda: defaultdict(list))   # orbital -> seq -> [(Z, λ, rms)]
    seqs = set()
    for r in ions:
        lam = {k: float(v) for k, v in ORB_RE.findall(r.get(f"lambdas_{mode}", ""))}
        if not lam:
            continue
        s = (int(r["N"]), r["sequence"]); seqs.add(s)
        for o, v in lam.items():
            pts[o][s].append((int(r["Z"]), v, _f(r.get(f"rms_{mode}"))))
    if not pts:
        return
    orbs = sorted(pts, key=lambda o: (int(o[:-1]), "spdfg".index(o[-1])))
    col = _seq_colors(seqs)
    ncol = min(5, len(orbs)); nrow = int(np.ceil(len(orbs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.1 * nrow), squeeze=False)
    for ax, o in zip(axes.flat, orbs):
        for s, p in sorted(pts[o].items()):
            p.sort()
            Z = [q[0] for q in p]; v = [q[1] for q in p]
            ax.plot(Z, v, "-o", ms=3, color=col[s])
            ax.annotate(s[1], (Z[-1], v[-1]), xytext=(3, 0), textcoords="offset points",
                        fontsize=7, color=col[s], va="center")
        ax.set_title(f"λ({o})", loc="left")
        ax.set_xlabel("Z")
    for ax in axes.flat[len(orbs):]:
        ax.axis("off")
    fig.suptitle(f"Tuned per-{mode} scaling parameters along isoelectronic sequences", x=0.01, ha="left",
                 fontsize=10)
    _save(fig, out, "fig2_lambda_vs_Z")


def fig_residuals(levels, ions, out):
    if not levels:
        return
    seq_of = {r["ion"]: (int(r["N"]), r["sequence"]) for r in ions}
    by = defaultdict(list)
    for r in levels:
        by[seq_of.get(r["ion"], (0, "?"))].append((_f(r["E_nist"]), _f(r["diff"]), r["ion"]))
    keys = sorted(by)
    ncol = min(4, len(keys)); nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow), sharey=True, squeeze=False)
    for ax, key in zip(axes.flat, keys):
        E, d, ions_ = zip(*by[key])
        ax.axhline(0, color=NEUTRAL, lw=0.8)
        ax.scatter(E, d, s=6, color=STAGE["orbital"], alpha=0.6, edgecolors="none")
        ax.set_title(f"{key[1]}-like  ({len(set(ions_))} ions, {len(E)} levels)", loc="left")
        ax.set_xlabel("E(NIST) / Ry")
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("E(SS) − E(NIST) / Ry")
    fig.suptitle("Level residuals at the tuned λ", x=0.01, ha="left", fontsize=10)
    _save(fig, out, "fig3_residuals")


def fig_convergence(samples, out, mode="orbital"):
    by = defaultdict(list)
    for r in samples:
        if r["mode"] != mode:
            continue
        by[r["ion"]].append((int(r["sample_id"]), _f(r["rms"], np.inf), r["tag"]))
    if not by:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    curves = []
    n_init = 0
    for ion, s in by.items():
        s.sort()
        best = np.minimum.accumulate([x[1] for x in s])
        final = best[-1]
        if not np.isfinite(final) or final <= 0:
            continue
        y = best / final
        ax.plot(np.arange(1, len(y) + 1), y, color=STAGE[mode], alpha=0.25, lw=0.9)
        curves.append(y)
        n_init = max(n_init, sum(1 for x in s if x[2] in ("sobol", "init")))
    if curves:
        L = max(len(c) for c in curves)
        M = np.full((len(curves), L), np.nan)
        for i, c in enumerate(curves):
            M[i, :len(c)] = c; M[i, len(c):] = c[-1]
        med = np.nanmedian(M, axis=0)
        ax.plot(np.arange(1, L + 1), med, color="#0b0b0b", lw=2, label=f"median of {len(curves)} ions")
    if n_init:
        ax.axvspan(0, n_init, color="#e6e5e1", alpha=0.6, lw=0)
        ax.text(n_init / 2, ax.get_ylim()[1] * 0.95, "Sobol design", ha="center", va="top", fontsize=8,
                color=NEUTRAL)
    ax.set_yscale("log")
    ax.set_xlabel("structx runs")
    ax.set_ylabel("best RMS so far / final RMS")
    ax.set_title(f"Convergence of the surrogate optimiser (per-{mode} stage)", loc="left")
    ax.legend(frameon=False)
    _save(fig, out, "fig4_convergence")


def fig_avalues(av, out):
    if not av:
        return
    rows = [(np.log10(_f(r["ratio_norad"])), np.log10(_f(r["ratio_nist"])), _f(r["LV"])) for r in av]
    rows = [r for r in rows if np.isfinite(r[2])]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    bins = np.linspace(-1.5, 1.5, 31)
    for ax, idx, name in zip(axes, (0, 1), ("NORAD", "NIST")):
        x = np.array([r[idx] for r in rows]); lv = np.array([r[2] for r in rows])
        ok = np.isfinite(x)
        good = ok & (np.abs(lv - 1) <= 0.2)
        bad = ok & ~good
        ax.hist(x[good], bins=bins, color=LV_OK, alpha=0.85, label=f"|L/V−1| ≤ 0.2  (n={good.sum()})")
        ax.hist(x[bad], bins=bins, color=LV_BAD, alpha=0.6, label=f"|L/V−1| > 0.2  (n={bad.sum()})")
        ax.axvline(0, color=NEUTRAL, lw=0.8)
        for lim, ls in ((np.log10(1.3), ":"), (np.log10(2), "--")):
            ax.axvline(lim, color=NEUTRAL, lw=0.7, ls=ls); ax.axvline(-lim, color=NEUTRAL, lw=0.7, ls=ls)
        ax.set_xlabel(f"log10  A(SS) / A({name})")
        ax.set_title(f"vs {name}", loc="left")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("transitions")
    fig.suptitle("A-value agreement, split by the length/velocity flag (dotted ±30 %, dashed ×2)",
                 x=0.01, ha="left", fontsize=10)
    _save(fig, out, "fig5_avalues")


def fig_prune(prune, out):
    if not prune:
        return
    ions = sorted({r["ion"] for r in prune})
    groups = sorted({r["group"] for r in prune if r["group"] != "baseline"})
    if not groups:
        return
    M = np.full((len(ions), len(groups)), np.nan)
    for r in prune:
        if r["group"] in groups:
            M[ions.index(r["ion"]), groups.index(r["group"])] = _f(r["delta_rms"])
    fig, ax = plt.subplots(figsize=(0.32 * len(groups) + 1.8, 0.24 * len(ions) + 1.2))
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 0.01
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ions))); ax.set_yticklabels(ions, fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("ΔRMS when the group is removed / Ry")
    ax.set_title("Which correlation configurations matter (leave-group-out)", loc="left")
    _save(fig, out, "fig6_prune")


def fig_predictor(verify, out, model):
    if not verify:
        return
    x = np.array([_f(r["rms_tuned"]) for r in verify]); y = np.array([_f(r["rms_pred"]) for r in verify])
    ok = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lim = (min(x[ok].min(), y[ok].min()) * 0.8, max(x[ok].max(), y[ok].max()) * 1.3)
    ax.plot(lim, lim, color=NEUTRAL, lw=0.8)
    ax.plot(lim, [2 * l for l in lim], color=NEUTRAL, lw=0.6, ls="--")
    ax.scatter(x[ok], y[ok], s=14, color=STAGE["orbital"], edgecolors="white", linewidths=0.4)
    for r, xx, yy, o in zip(verify, x, y, ok):
        if o and yy > 1.5 * xx:
            ax.annotate(r["ion"], (xx, yy), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("RMS at tuned λ / Ry")
    ax.set_ylabel("RMS at predicted λ (one run) / Ry")
    med = np.median(y[ok] / x[ok])
    ax.set_title(f"λ predictor, leave-one-ion-out ({model}): median ratio {med:.2f}", loc="left")
    _save(fig, out, f"fig7_predictor_{model}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or os.path.join(a.dataset, "figures")
    os.makedirs(out, exist_ok=True)
    d = a.dataset
    ions = _read(os.path.join(d, "ions.csv"))
    print("figures ->", out)
    fig_rms_vs_Z(ions, out)
    fig_lambda_vs_Z(ions, out)
    fig_residuals(_read(os.path.join(d, "levels.csv")), ions, out)
    fig_convergence(_read(os.path.join(d, "samples.csv")), out)
    fig_avalues(_read(os.path.join(d, "avalues.csv")), out)
    fig_prune(_read(os.path.join(d, "prune.csv")), out)
    for model in ("gp", "seq"):
        fig_predictor(_read(os.path.join(d, f"verify_{model}.csv")), out, model)


if __name__ == "__main__":
    main()
