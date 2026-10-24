"""Uncertainties of the radiative data, and checks against data the tuning did not use.

    ssopt uncert "Ni XV" --mode orbital          (after `tune`; one Slurm job, see slurm/uncert.slurm)

Two sources of error are estimated for every E1 line and added in quadrature:

1. The flat valley of the energy fit. Many λ sets reproduce the NIST energies about
   equally well but give different A-values. The λ sets tried during tuning whose energy
   RMS lies inside the 68% confidence region of the fit,

       RMS <= RMS_best * sqrt(1 + chi2_68(d) / N)      (d λ values, N fitted levels),

   are run with radiative data, and s_ens is the standard deviation of ln A over them.
   (The level errors are taken to be of the size of the best-fit scatter, so this is the
   usual Δχ² region of a least-squares fit with d parameters.)
2. The quality of the wavefunction. Length and velocity forms of gf agree for an exact
   wavefunction, so s_lv = |ln(gf_L/gf_V)|. One line can agree by chance, so the mean of
   s_lv over the lines within a factor 10 in line strength is used as a floor, as in the
   MCDHF practice (Ekman et al. 2014, Kramida 2014).

       sigma = sqrt( (a s_ens)² + (b max(s_lv, bin mean))² + F² )          (in ln A)

a, b and F were fitted once against independent MCDHF and CI+MBPT calculations so that
68% of the actual differences fall within sigma in every cell of (s_ens, s_lv), separately
for strong lines (gf >= 0.01) of ions with charge >= 2, strong lines of neutral and singly
charged ions, and weak lines, and checked on ions left out of the fit (docs/details.md).
sigma is converted to a percent uncertainty and a NIST letter grade.

Lifetimes: s_ens from the spread of ln tau over the same λ sets; the length/velocity part
of each E1 decay channel is added in quadrature, weighted by its branching fraction.
Metastable levels that decay only by E2/M1 lines have no length/velocity check, so only
the spread and F enter. Lifetimes use the strong-line constants of the ion's charge group.

The checks that use data not involved in the tuning:
* held-out levels: the λ sets tried during tuning are re-ranked using 4/5 of the NIST
  levels, and the best one is scored on the other 1/5 (5 folds). No new runs are needed.
* the A-values at λ = 1 (no tuning) and at the tuned λ, both against NIST.
* A_obs: the A-value with the computed transition energy replaced by the NIST one
  (A ∝ ΔE³ for E1). It is a column for reference only; the tuned A stays the result.
"""
from __future__ import annotations

import math
import os
import shutil
from collections import defaultdict

import numpy as np

# (a, b, F), fitted on the benchmark ions, see docs/details.md ("Uncertainties")
CALIB = {"strong": (5.5, 0.2, 0.09),             # gf >= 0.01, ion charge >= 2
         "strong, charge 0-1": (3.5, 0.6, 0.33),  # gf >= 0.01, neutral and singly charged ions
         "weak": (5.0, 0.6, 1.02)}                # gf < 0.01


def calib(gf: float, charge: int) -> tuple:
    if gf < 0.01:
        return CALIB["weak"]
    return CALIB["strong"] if charge >= 2 else CALIB["strong, charge 0-1"]


def combine(s_ens: float, s_g: float, abf: tuple) -> float:
    a, b, F = abf
    s_e = s_ens if math.isfinite(s_ens) else 0.0
    s_g = s_g if math.isfinite(s_g) else 0.0
    return math.sqrt((a * s_e) ** 2 + (b * s_g) ** 2 + F ** 2)

GRADES = [(0.003, "AAA"), (0.01, "AA"), (0.02, "A+"), (0.03, "A"), (0.07, "B+"), (0.10, "B"),
          (0.18, "C+"), (0.25, "C"), (0.40, "D+"), (0.50, "D")]


def grade(frac: float) -> str:
    """NIST ASD accuracy letter for a fractional uncertainty."""
    if not math.isfinite(frac):
        return "-"
    for lim, g in GRADES:
        if frac <= lim:
            return g
    return "E"


def frac(sig: float) -> float:
    """sigma of ln A -> fractional uncertainty of A (the larger side, exp(sigma) - 1)."""
    return math.expm1(sig) if math.isfinite(sig) else float("nan")


# --------------------------------------------------------------------- the λ ensemble
def ensemble(samples, best, n_fit: int, max_runs: int = 16, level: float = 0.683):
    """λ sets inside the confidence region of the energy fit, `best` first.
    Returns (samples, threshold as a multiple of the best RMS)."""
    from scipy.stats import chi2
    fac = math.sqrt(1.0 + chi2.ppf(level, len(best.lam)) / max(n_fit, 1))
    pool, seen = [best], {tuple(round(x, 4) for x in best.lam)}
    for s in sorted((s for s in samples if s.ok and math.isfinite(s.rms)), key=lambda s: s.rms):
        if s.rms > fac * best.rms:
            break
        key = tuple(round(x, 4) for x in s.lam)
        if key not in seen:
            seen.add(key)
            pool.append(s)
    if len(pool) > max_runs:
        idx = np.unique(np.linspace(0, len(pool) - 1, max_runs).round().astype(int))
        pool = [pool[i] for i in idx]
    return pool, fac


def level_keys(levels) -> dict[int, tuple]:
    """K -> (parity, 2J, rank in energy among levels of that parity and J). Runs of the
    same deck have the same levels, so this identifies a level across λ sets; where two
    levels cross between λ sets their lines swap, which shows up as a large spread."""
    by = defaultdict(list)
    for lv in levels:
        by[(lv.parity, lv.twoJ)].append(lv)
    out = {}
    for k, v in by.items():
        for r, lv in enumerate(sorted(v, key=lambda x: x.E)):
            out[lv.K] = (k[0], k[1], r)
    return out


def lines_by_key(e1, keys) -> dict:
    return {(keys[r.K], keys[r.KP]): r for r in e1 if r.K in keys and r.KP in keys}


def lv_terms(e1) -> dict:
    """(K, KP) -> (s_lv, bin mean of s_lv over lines within a factor 10 in S)."""
    rows = [r for r in e1 if r.gf_vel > 0 and r.gf > 0 and r.S != 0]      # SD is printed with a sign
    if not rows:
        return {}
    lv = np.array([abs(math.log(r.gf / r.gf_vel)) for r in rows])
    ls = np.log10([abs(r.S) for r in rows])
    o = np.argsort(ls)
    ls_s, lv_s = ls[o], lv[o]
    cs = np.concatenate([[0.0], np.cumsum(lv_s)])
    lo = np.searchsorted(ls_s, ls - 0.5, "left")
    hi = np.searchsorted(ls_s, ls + 0.5, "right")
    mean = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return {(r.K, r.KP): (float(a), float(b)) for r, a, b in zip(rows, lv, mean)}


def line_sigma(s_ens: float, s_lv: float, s_lvb: float, gf: float, charge: int) -> float:
    g = max(s_lv, s_lvb) if math.isfinite(s_lv) else s_lvb
    return combine(s_ens, g, calib(gf, charge))


def components(best_e1, best_keys, member_lines: list[dict]) -> dict:
    """(K, KP) of the best run -> (s_ens, s_lv, s_lvb, n members with the line)."""
    lvt = lv_terms(best_e1)
    out = {}
    for r in best_e1:
        key = (best_keys.get(r.K), best_keys.get(r.KP))
        vals = [math.log(m[key].A) for m in member_lines if key in m and m[key].A > 0]
        s_ens = float(np.std(vals, ddof=1)) if len(vals) >= 3 else float("nan")
        s_lv, s_lvb = lvt.get((r.K, r.KP), (float("nan"), float("nan")))
        out[(r.K, r.KP)] = (s_ens, s_lv, s_lvb, len(vals))
    return out


# --------------------------------------------------------------------- held-out levels
def holdout(hist, nist_levels, objective: str = "rel", folds: int = 5, seed: int = 0, extra=()):
    """Re-rank the λ sets of the history on 4/5 of the levels, score the winner on the other 1/5.
    Returns (RMS on fitted levels, RMS on held-out levels, RMS of λ = 1 on all levels or nan, N)."""
    from . import match as M
    ok = [s for s in hist.samples if s.ok and s.E]
    if not ok:
        return None
    best = min(ok, key=lambda s: s.rms)
    lev = sorted(best.E)
    if len(lev) < 4:
        return None
    w = M.target_weights({k: nist_levels[k].E for k in lev}, objective)

    def err(s, sub):
        d = [w[k] * (s.E[k] - nist_levels[k].E) for k in sub if k in s.E]
        return d

    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(lev))
    nf = min(folds, len(lev))
    fit_sq, out_sq = [], []
    for f in range(nf):
        test = set(perm[f::nf])
        train = [k for k in lev if k not in test]
        cand = [s for s in ok if sum(k in s.E for k in train) >= 0.9 * len(train)]
        win = min(cand, key=lambda s: np.mean(np.square(err(s, train))))
        fit_sq += list(np.square(err(win, train)))
        out_sq += list(np.square(err(win, test)))
    one = float("nan")
    for s in list(ok) + [s for h in extra for s in h.samples if s.ok and s.E]:
        if all(abs(x - 1.0) < 1e-9 for x in s.lam):
            one = math.sqrt(np.mean(np.square(err(s, lev))))
            break
    return math.sqrt(np.mean(fit_sq)), math.sqrt(np.mean(out_sq)), one, len(lev)


def holdout_line(res, objective: str) -> str:
    if res is None:
        return "held-out levels: too few levels"
    a, b, one, n = res
    f = (lambda x: f"{100 * x:.2f} %") if objective == "rel" else (lambda x: f"{x:.5f} Ry")
    s = (f"held-out levels (5 folds over {n} NIST levels): RMS {f(a)} on the levels used to pick λ, "
         f"{f(b)} on the levels left out")
    if math.isfinite(one):
        s += f"; λ = 1 (no tuning) gives {f(one)}"
    return s


# --------------------------------------------------------------------- the command
def run(args, ion, iondir, log, nl, plan, hist, extra_hist=()):
    """extra_hist: the other tuning stage. With --mode orbital the λ sets of the l stage
    (one λ per l, i.e. equal λ for the orbitals of one l) are points of the same orbital
    space and join the ensemble when their energies are inside the region."""
    from . import match as M
    from . import nist as N
    from .avalues import _nist_line_index
    from .deck import e1_only, limit_transitions, write_ssin
    from .export import relabel_from_nist
    from .lifetimes import level_label, lifetimes, parse_rates
    from .runner import run_batch
    from .ssout import parse_e1, read

    from .deck import lambda_names
    from .surrogate import Sample
    best = hist.best()
    samples = list(hist.samples)
    if args.mode == "orbital":
        on, ln = lambda_names(plan.configs, "orbital"), lambda_names(plan.configs, "l")
        for h in extra_hist:
            if h.mode == "l" and h.objective == hist.objective:
                samples += [Sample([s.lam[ln.index(o[-1])] for o in on], s.E, s.rms, s.n_matched, s.elapsed,
                                   s.ok, "l") for s in h.samples if s.ok]
    pool, fac = ensemble(samples, best, best.n_matched, args.max_runs)
    lam1 = [1.0] * len(best.lam)
    lams = [list(s.lam) for s in pool] + [lam1]
    names = [f"m{i:02d}" for i in range(len(pool))] + ["lambda1"]
    log(f"uncertainty ensemble: {len(pool)} λ sets with energy RMS <= {fac:.2f} x best "
        f"(68% region, d={len(best.lam)}, N={best.n_matched}); plus λ = 1")

    def deck(lam):
        t = write_ssin(plan.configs, ion.Z, lam, mode=args.mode, mpnch=-1, closed_core=args.closed_core)
        if not args.all_transitions:
            t = limit_transitions(t, plan.tags.count("spec"))
        return e1_only(t) if args.e1_only else t

    root = os.path.join(args.scratch or iondir, "uncert_runs")
    res = run_batch([deck(l) for l in lams], root, os.path.abspath(args.binary), names=names,
                    n_workers=args.workers, stop_after_levels=False, timeout=args.timeout * 10)
    runs = []
    for r in res:
        if not r.ok or r.timed_out:
            runs.append(None)
            continue
        text = read(r.ssout)
        e1 = parse_e1(text)
        runs.append((r, e1, lifetimes(parse_rates(text)) if not args.e1_only else {}))
    if runs[0] is None:
        raise SystemExit("the run at the best λ failed")
    r0, e10, tau0 = runs[0]
    relabel_from_nist(r0.levels, ion, iondir, r0.configs, args.data, args.match_key)
    keys0 = level_keys(r0.levels)
    members = [x for x in runs[:-1] if x is not None]
    mlines = [lines_by_key(e1, level_keys(r.levels)) for r, e1, _ in members]
    comp = components(e10, keys0, mlines)

    # lifetimes: spread of ln tau over the members, and the L/V part of each E1 channel
    mtau = []
    for r, _, tau in members:
        kk = level_keys(r.levels)
        mtau.append({kk[k]: v[0] for k, v in tau.items() if k in kk})
    chan = defaultdict(list)
    for x in e10:
        s = comp[(x.K, x.KP)]
        chan[x.K].append((x.A, max(s[1], s[2]) if math.isfinite(s[1]) else s[2]))

    # NIST identities
    targets = M.nist_targets(nl, ion, plan.configs, e_max=args.e_max)
    m0 = M.match_levels(r0.levels, nl, ion, plan.configs, r0.configs, targets=targets, match_key=args.match_key)
    flagged = M.ambiguous(m0, r0.levels, nl, ion, plan.configs, r0.configs, targets, args.match_key)
    K2n = {r0.levels[m.comp_idx].K: m.nist_idx for m in m0}
    byK = {lv.K: lv for lv in r0.levels}

    def lab(K):
        if K in K2n:
            d = nl[K2n[K]]
            return f"{d.config} {d.term} {d.J:g}" + ("?" if K2n[K] in flagged else "")
        cfg, term, J = level_label(byK[K], r0.configs, ion.core)
        return f"{cfg} {term} {J:g} (SS)"

    try:
        nlines = N.load_lines(args.data, ion.spectroscopic, fetch_if_missing=False)
    except (FileNotFoundError, ValueError):
        nlines = []
    lidx = _nist_line_index(nlines, nl) if nlines else {}

    def vs_nist(run):
        if run is None or not lidx:
            return None
        r, e1, _ = run
        mm = M.match_levels(r.levels, nl, ion, plan.configs, r.configs, targets=targets, match_key=args.match_key)
        kn = {r.levels[m.comp_idx].K: m.nist_idx for m in mm}
        q = []
        for x in e1:
            if x.K in kn and x.KP in kn and x.gf >= 0.01:
                a, b = sorted((kn[x.K], kn[x.KP]), key=lambda i: nl[i].E)
                if a in flagged or b in flagged:
                    continue
                ln = lidx.get((a, b))
                if ln is not None and ln.A > 0:
                    q.append(abs(math.log10(x.A / ln.A)))
        if not q:
            return None
        q = np.array(q)
        return (f"n={len(q)}, within 10%: {np.mean(q < math.log10(1.1)) * 100:.0f}%, within 30%: "
                f"{np.mean(q < math.log10(1.3)) * 100:.0f}%, within x2: {np.mean(q < math.log10(2)) * 100:.0f}%, "
                f"median factor {10 ** np.median(q):.2f}")

    # ---------------------------------------------------------------- report
    out = [f"Uncertainties ({len(members)} of {len(pool)} ensemble runs usable; method in docs/details.md)",
           f"  ensemble : λ sets with energy RMS <= {fac:.2f} x the best ({best.rms:.5f}), the 68% region of the "
           f"energy fit with {len(best.lam)} λ and {best.n_matched} levels",
           f"  sigma    : sqrt((a s_ens)^2 + (b s_LV)^2 + F^2) in ln A; (a, b, F) = "
           f"{calib(1.0, ion.charge)} for gf >= 0.01, {calib(0.0, ion.charge)} for weaker lines",
           "  " + holdout_line(holdout(hist, nl, args.objective, extra=extra_hist), args.objective)]
    a1, ab = vs_nist(runs[-1]), vs_nist(runs[0])
    if ab:
        out.append(f"  E1 A-values (gf >= 0.01) vs NIST at λ = 1 : {a1 or 'run failed'}")
        out.append(f"  E1 A-values (gf >= 0.01) vs NIST, tuned   : {ab}")
    rows = []
    for x in e10:
        if x.gf < 1e-3:
            continue
        s_ens, s_lv, s_lvb, n = comp[(x.K, x.KP)]
        sig = line_sigma(s_ens, s_lv, s_lvb, x.gf, ion.charge)
        aobs = float("nan")
        if x.K in K2n and x.KP in K2n:
            de_n = abs(nl[K2n[x.K]].E - nl[K2n[x.KP]].E)
            de_c = abs(byK[x.K].E - byK[x.KP].E)
            if de_c > 0 and de_n > 0:
                aobs = x.A * (de_n / de_c) ** 3
        rows.append((x, sig, s_ens, s_lv, s_lvb, n, aobs))
    strong = [frac(t[1]) for t in rows if t[0].gf >= 0.01]
    if strong:
        cnt = defaultdict(int)
        for u in strong:
            cnt[grade(u)[0]] += 1
        out.append(f"  E1 lines with gf >= 0.01: {len(strong)}, median uncertainty {100 * np.median(strong):.0f}%, "
                   f"grades " + " ".join(f"{g}:{cnt[g]}" for g in "ABCDE" if cnt[g]))
    out.append("")
    out.append("E1 lines (unc = 1-sigma uncertainty of A; s_ens = spread over the λ ensemble; s_LV = length/velocity "
               "term, the larger of the line's own and its line-strength bin; A_obs = A at the NIST transition energy, "
               "for reference; '?' = uncertain identification; (SS) = SUPERSTRUCTURE label, no NIST level)")
    out.append(f"{'lower':<26}{'upper':<26}{'λ(Å)':>10}{'gf':>10}{'A(s-1)':>11}{'unc%':>7}{'grade':>6}"
               f"{'s_ens%':>8}{'s_LV%':>7}{'A_obs':>11}")
    for x, sig, s_ens, s_lv, s_lvb, n, aobs in sorted(rows, key=lambda t: (byK[t[0].KP].E, byK[t[0].K].E)):
        g = max(s_lv, s_lvb) if math.isfinite(s_lv) else s_lvb
        p = lambda v: f"{100 * v:8.1f}" if math.isfinite(v) else f"{'-':>8}"
        out.append(f"{lab(x.KP)[:25]:<26}{lab(x.K)[:25]:<26}{x.wavel:>10.2f}{x.gf:>10.4f}{x.A:>11.3e}"
                   f"{100 * frac(sig):>7.0f}{grade(frac(sig)):>6}{p(s_ens)}{p(g)[1:]}"
                   + (f"{aobs:>11.3e}" if math.isfinite(aobs) else f"{'-':>11}"))
    if tau0:
        out.append("")
        out.append("Lifetimes (unc = 1-sigma; 'forbidden' = decays only by E2/M1 lines, no length/velocity check)")
        out.append(f"{'level':<30}{'E(Ry)':>10}{'tau(s)':>12}{'unc%':>7}")
        for lv in sorted(r0.levels, key=lambda v: v.E):
            if lv.K not in tau0 or not math.isfinite(tau0[lv.K][0]) or tau0[lv.K][0] > 1.0:
                continue
            k = keys0[lv.K]
            vals = [math.log(t[k]) for t in mtau if k in t and 0 < t[k] < math.inf]
            s_ens = float(np.std(vals, ddof=1)) if len(vals) >= 3 else 0.0
            gam = tau0[lv.K][1]
            ch = chan.get(lv.K, [])
            g = math.sqrt(sum((a * (s if math.isfinite(s) else 0.0)) ** 2 for a, s in ch)) / gam if gam > 0 else 0.0
            sig = combine(s_ens, g, calib(1.0, ion.charge))
            note = "" if ch else "  forbidden"
            out.append(f"{lab(lv.K)[:29]:<30}{lv.E:>10.5f}{tau0[lv.K][0]:>12.3e}{100 * frac(sig):>7.0f}{note}")
    shutil.rmtree(root, ignore_errors=True)
    return "\n".join(out)
