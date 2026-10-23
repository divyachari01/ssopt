"""Read-only A-value check.

After λ tuning, one full structx run (radiative data on) gives the E1 table
(K, KP, AE1, gf, ...). K/KP are indices in the J-level table, so every matched
NIST level gives us a NIST identity for K. For each E1 row where both ends are
NIST-identified we look up:

  * NIST A_ki  — from the ASD lines cache, matched on the two level energies
  * NORAD A    — from Nahar's f.fs file, matched on quantum numbers + energies

and report ratios. Weak lines (gf below `gf_min`) are skipped unless NIST lists
them, since their A values are dominated by cancellation.
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import nist as N
from . import norad as R
from .ions import Ion
from .ssout import parse_e1, read as read_ssout


def _nist_line_index(lines: list[N.NistLine], levels: list[N.NistLevel], tol=2e-5):
    """(lower level idx, upper level idx) -> NIST A, by energy lookup."""
    E = np.array([d.E for d in levels])
    idx = {}
    for ln in lines:
        if ln.type:                       # only E1 lines compare with the E1 table
            continue
        if not (math.isfinite(ln.Ei) and math.isfinite(ln.Ek)):
            continue
        i = int(np.argmin(np.abs(E - ln.Ei)))
        k = int(np.argmin(np.abs(E - ln.Ek)))
        if abs(E[i] - ln.Ei) > tol or abs(E[k] - ln.Ek) > tol:
            continue
        # J must agree too (levels with identical energies are rare but exist)
        if math.isfinite(ln.J_i) and abs(levels[i].J - ln.J_i) > 0.1:
            continue
        if math.isfinite(ln.J_k) and abs(levels[k].J - ln.J_k) > 0.1:
            continue
        idx[(i, k)] = ln
    return idx


def _norad_level_index(trans: list[R.NoradTransition], levels: list[N.NistLevel],
                       tol_obs=3e-4, tol_calc=0.03):
    """NORAD fine-structure level -> NIST level idx.

    NORAD sub-rows carry the observed level energy (cm^-1 -> Ry) whenever NIST
    had it, so observed levels match by (parity, 2S+1, L, 2J) + energy to
    ~1e-4 Ry; levels marked calculated ('*') get a loose tolerance."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, d in enumerate(levels):
        if d.twoSp1 is not None and d.L is not None:
            buckets[d.key].append(i)
    cache = {}

    def find(qn, E, calc):
        key = (qn, round(E, 6))
        if key in cache:
            return cache[key]
        cands = buckets.get(qn, [])
        j = min(cands, key=lambda i: abs(levels[i].E - E)) if cands else None
        tol = tol_calc if calc else tol_obs
        cache[key] = j if j is not None and abs(levels[j].E - E) <= tol else None
        return cache[key]

    out = {}
    for t in trans:
        a = find(t.lower_qn, t.Ei, t.calc_i)
        b = find(t.upper_qn, t.Ef, t.calc_f)
        if a is None or b is None:
            continue
        lo, hi = (a, b) if levels[a].E <= levels[b].E else (b, a)
        out[(lo, hi)] = t
    return out


def compare(iondir: str, ion: Ion, run, matches, nist_levels, data_dir: str,
            gf_min: float = 1e-3, max_rows: int = 400, flagged=frozenset(), header: str = "") -> str:
    text = read_ssout(run.ssout)
    e1 = parse_e1(text)
    if not e1:
        return "A-values: no E1 table in ssout (was the run stopped early?)"
    comp2nist = {m.comp_idx: m.nist_idx for m in matches}
    K2nist = {run.levels[j].K: n for j, n in comp2nist.items()}

    nist_lines = []
    try:
        nist_lines = N.load_lines(data_dir, ion.spectroscopic, fetch_if_missing=False)
    except (FileNotFoundError, ValueError):     # no NIST A-value table for this ion (cached error page)
        pass
    line_idx = _nist_line_index(nist_lines, nist_levels) if nist_lines else {}

    ntrans, ip = R.load_transitions(data_dir, ion)
    norad_idx = _norad_level_index(ntrans, nist_levels) if ntrans else {}

    rows = []
    for r in e1:
        if r.K not in K2nist or r.KP not in K2nist:
            continue
        up, lo = K2nist[r.K], K2nist[r.KP]
        if nist_levels[up].E < nist_levels[lo].E:
            up, lo = lo, up
        a_nist = line_idx.get((lo, up))
        if r.gf < gf_min and a_nist is None:      # weak lines are listed only when NIST has them
            continue
        a_norad = norad_idx.get((lo, up))
        rows.append((lo, up, r, a_nist.A if a_nist else float("nan"),
                     a_nist.acc if a_nist else "", a_norad.A if a_norad else float("nan")))

    def stats(ratios):
        ratios = np.array([x for x in ratios if math.isfinite(x) and x > 0])
        if not len(ratios):
            return "n=0"
        lr = np.abs(np.log10(ratios))
        return (f"n={len(ratios)}  within 10%: {np.mean(lr < math.log10(1.1)) * 100:.0f}%  "
                f"within 30%: {np.mean(lr < math.log10(1.3)) * 100:.0f}%  "
                f"within x2: {np.mean(lr < math.log10(2)) * 100:.0f}%  "
                f"median |log10 ratio|={np.median(lr):.3f}")

    out = [f"A-value check ({len(rows)} E1 transitions between NIST-identified levels: gf >= {gf_min}, "
           f"plus every weaker line that NIST lists)"]
    sure = [t for t in rows if t[0] not in flagged and t[1] not in flagged]
    out.append("  vs NIST : " + stats([r.A / a for _, _, r, a, _, _ in sure if math.isfinite(a) and a > 0]))
    nq = sum(1 for t in rows if (t[0] in flagged or t[1] in flagged) and math.isfinite(t[3]))
    if nq:
        out.append(f"  ({nq} NIST line(s) involve a level listed as an uncertain identification at the end of "
                   f"levels_orbital.txt and are left out of these numbers)")
    out.append("  vs NORAD: " + stats([r.A / a for _, _, r, _, _, a in rows if math.isfinite(a) and a > 0]))
    if not nist_lines:
        out.append("  (no NIST lines cache — run `ssopt fetch` on a login node)")
    if not ntrans:
        out.append("  (no NORAD transition file for this ion)")
    out.append("")
    out.append(f"{'lower':<20}{'upper':<20}{'λ(Å)':>9}{'gf_SS':>9}{'L/V':>6}{'A_SS':>11}{'A_NIST':>11}{'acc':>4}"
               f"{'A_NORAD':>11}{'SS/NIST':>9}{'SS/NORAD':>10}")
    out.append("  (L/V = gf(length) / [4 gf(velocity)]. structx prints 'VEL.GF' in a convention that is 1/4 of the length "
               "form for exact wavefunctions — Si I resonance lines that agree with NIST sit at 1.00. Far from 1 means a "
               "poor wavefunction for that transition, not a λ problem)")
    rows.sort(key=lambda t: -t[2].gf)
    for lo, up, r, an, acc, ar in rows[:max_rows]:
        dl = nist_levels[lo]; du = nist_levels[up]
        l1 = f"{dl.config} {dl.term} {dl.J:g}"[:19]
        l2 = f"{du.config} {du.term} {du.J:g}"[:19]
        fa = lambda x: f"{x:11.3e}" if math.isfinite(x) else f"{'-':>11}"
        fr = lambda x: f"{x:9.2f}" if math.isfinite(x) else f"{'-':>9}"
        lv_ratio = r.gf / (4.0 * r.gf_vel) if r.gf_vel > 0 else float("nan")
        out.append(f"{l1:<20}{l2:<20}{r.wavel:>9.1f}{r.gf:>9.4f}{lv_ratio:>6.2f}{r.A:>11.3e}{fa(an)}{acc:>4}{fa(ar)}"
                   f"{fr(r.A / an if math.isfinite(an) and an > 0 else float('nan'))}"
                   f"{fr(r.A / ar if math.isfinite(ar) and ar > 0 else float('nan')):>10}")
    rep = "\n".join(out)
    with open(os.path.join(iondir, "avalues.txt"), "w") as f:
        f.write(header + rep + "\n")
    return rep
