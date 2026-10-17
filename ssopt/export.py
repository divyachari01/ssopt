"""Write an ion's tuned SUPERSTRUCTURE output as NORAD-style tables.

    python -m ssopt.export "Ne V" --workdir work_rel --out norad_ss

produces  norad_ss/ne5.en.ls.txt   LS terms, bound-state energies (negative, Ry, from the NIST limit)
          norad_ss/ne5.en.fs.txt   fine-structure levels grouped by J and parity
          norad_ss/ne5.f.ls.txt    LS-multiplet oscillator strengths: i j wl(A) Ei Ej f S A
          norad_ss/ne5.f.fs.txt    fine-structure f-values, level-index layout (NORAD Li-/He-like files)
          norad_ss/ne5.f.fsm.txt   fine-structure f-values, multiplet-header layout (NORAD Si I style)

Layouts follow the NORAD files (Nahar) column for column so existing readers
work unchanged.  What SUPERSTRUCTURE cannot supply is filled honestly:
  * the "target/core" columns hold the configuration minus its outermost
    electron and the parent term is written as "( - )" (SS gives the level's
    own LS term, not the core's); the level's own term is in the Terms column;
  * energies are relative to the ionization limit taken from the NIST level
    list (E = E_SS - IP); without a limit they stay ground-relative and positive
    and the header says so;
  * seniority letters follow NORAD's rule: within one (2S+1, L, pi) symmetry
    terms are lettered a, b, c ... by increasing energy for even parity and
    z, y, x ... for odd parity.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import string
from collections import defaultdict

import numpy as np

from .ions import build_ion, config_str, orb_str
from .ssout import parse_configs, parse_e1, parse_levels

RY_CM = 109737.31568
LNAME = "SPDFGHIKLMNOQ"


# ----------------------------------------------------------------------------
def nist_ionization_limit(data_dir: str, spectrum: str) -> float | None:
    """Lowest 'Limit' row of the cached NIST level CSV, in Ry (None if absent)."""
    from .nist import levels_cache_path
    p = levels_cache_path(data_dir, spectrum)
    if not os.path.exists(p):
        return None
    best = None
    with open(p) as f:
        for row in csv.reader(f):
            if len(row) > 4 and "Limit" in row[1]:
                try:
                    e = float(row[4].replace('="', "").replace('"', "").strip())
                except ValueError:
                    continue
                best = e if best is None else min(best, e)
    return best


def _split_outer(cfg: dict, core: dict) -> tuple[dict, tuple[int, int]]:
    """Configuration -> (core part, outermost electron (n,l)): the highest-n (then
    highest-l) orbital outside the closed core loses one electron."""
    val = {o: n for o, n in cfg.items() if o not in core}
    if not val:
        val = dict(cfg)
    outer = max(val, key=lambda o: (o[0], o[1]))
    rest = dict(cfg)
    rest[outer] -= 1
    if rest[outer] == 0:
        del rest[outer]
    return rest, outer


def _term(twoSp1: int, L: int, parity: int) -> str:
    return f"{twoSp1}{LNAME[L]}{'o' if parity < 0 else 'e'}"


def _seniority(levels_or_terms):
    """NORAD lettering per (2S+1, L, parity) symmetry: a,b,c.. up for even; z,y,x.. down for odd."""
    groups = defaultdict(list)
    for i, x in enumerate(levels_or_terms):
        groups[(x["S"], x["L"], x["p"])].append((x["E"], i))
    letters = {}
    for key, items in groups.items():
        items.sort()
        seq = string.ascii_lowercase + string.ascii_uppercase
        if key[2] < 0:
            seq = string.ascii_lowercase[::-1] + string.ascii_uppercase[::-1]
        for k, (_, i) in enumerate(items):
            letters[i] = seq[k % len(seq)]
    return letters


def _frac_J(twoJ: int) -> str:
    return f"{twoJ // 2}" if twoJ % 2 == 0 else f"{twoJ}/2"


# ----------------------------------------------------------------------------
def load_run(iondir: str):
    with open(os.path.join(iondir, "ssout.full")) as f:
        text = f.read()
    cfgs = parse_configs(text)
    levels = parse_levels(text, cfgs)
    e1 = parse_e1(text)
    return text, cfgs, levels, e1


def relabel_from_nist(levels, ion, iondir, cf_configs, data_dir, match_key="j", factor=3.0, min_cost=0.25):
    """Give a level the NIST LS label when the energies force it.

    SUPERSTRUCTURE names a level after its largest LS component; for strongly mixed
    pairs of the same (configuration, parity, J) that name flips (Ni XV 3s2 3p2:
    3P2 and 1D2 come out as 1D2 and 3P2 at the right energies). The matcher pairs
    levels by J and energy inside the configuration without looking at the term, so
    a disagreement can also be an artefact: two terms of a light ion a few hundredths
    of a Ry apart, paired crosswise because the deck error is of the same size.
    The two cases are told apart by the energies. For a candidate (SS level with
    term X paired to NIST term Y) the label-respecting alternative (SS-X with NIST-X,
    SS-Y with NIST-Y, each the nearest in the same configuration/parity/J bucket) is
    costed against the swap (relative errors); the swap is accepted only if it is
    `factor` times cheaper AND the label-respecting pairing would be off by more
    than `min_cost` (25%) of the level energies, i.e. the disagreement is far larger
    than any deck error. Close terms that merely cross because of a few-percent deck
    error keep the SUPERSTRUCTURE label; so do levels without a NIST partner."""
    from . import match as M, nist
    try:
        nl = nist.load_levels(data_dir, ion.spectroscopic, fetch_if_missing=False)
        with open(os.path.join(iondir, "deck.json")) as f:
            configs = [{tuple(o): n for o, n in c} for c in json.load(f)["configs"]]
    except FileNotFoundError:
        return 0
    targets = M.nist_targets(nl, ion, configs)
    matches = M.match_levels(levels, nl, ion, configs, cf_configs, targets=targets, match_key=match_key)
    if not matches:
        return 0
    floor = 0.01 * max(nl[k].E for k in targets)

    def rel(dE, E):
        return abs(dE) / max(E, floor)

    def ss_bucket(lv):
        return (M.valence_key(cf_configs[lv.CF], ion.core), lv.parity, lv.twoJ)

    def nist_bucket(d):
        return (M.valence_key(d.cfg, ion.core), d.parity, int(round(2 * d.J)))

    ss_by = defaultdict(list)
    for lv in levels:
        if lv.CF in cf_configs:
            ss_by[ss_bucket(lv)].append(lv)
    nist_by = defaultdict(list)
    for i in targets:
        nist_by[nist_bucket(nl[i])].append(nl[i])

    def nearest(cands, E, term):
        c = [x for x in cands if (x.twoSp1, x.L) == term]
        return min(c, key=lambda x: abs(x.E - E)) if c else None

    decay = _decay_scorer(levels, matches, nl, ion, iondir, data_dir)

    new = {}
    for m in matches:
        d, lv = nl[m.nist_idx], levels[m.comp_idx]
        if d.twoSp1 is None or d.L is None or (lv.twoSp1, lv.L) == (d.twoSp1, d.L):
            continue
        if "?" in d.term:          # NIST marks the identification itself as uncertain
            continue
        X, Y = (lv.twoSp1, lv.L), (d.twoSp1, d.L)
        d2 = nearest(nist_by.get(ss_bucket(lv), []), lv.E, X)      # NIST level with SS's term
        lv2 = nearest(ss_by.get(nist_bucket(d), []), d.E, Y)       # SS level with NIST's term
        # Decay test first: NIST transition rates into and out of these levels say which
        # SS level is which, independently of how close the energies are. A level that
        # carries the wrong label decays like the other term (e.g. a 3S that talks to
        # 3P0,1,2 against a 1P that talks to 1D2), typically off by a factor of 10 or more.
        verdict = decay(lv, d, lv2, d2) if decay else None
        if verdict is not None:
            accept = verdict
        elif d2 is None and lv2 is None:
            accept = True                                          # nothing to respect: only option
        else:
            cost_keep = (rel(lv.E - d2.E, d2.E) if d2 else 0.0) + (rel(lv2.E - d.E, d.E) if lv2 else 0.0)
            cost_swap = rel(lv.E - d.E, d.E) + (rel(lv2.E - d2.E, d2.E) if (lv2 and d2) else 0.0)
            # forced: the label-respecting pairing is `factor` times worse than the swap AND
            # would put levels off by more than `min_cost` of their energy (a crossing of two
            # close terms caused by a few-percent deck error never gets near that)
            accept = cost_swap * factor < cost_keep and cost_keep > min_cost
        if accept:
            new[m.comp_idx] = Y
    for i, (S, L) in new.items():
        levels[i].twoSp1, levels[i].L = S, L
    return len(new)


def relabel_ssout(iondir: str, ion, data_dir: str, match_key: str = "j") -> int:
    """Rewrite the fine-structure level table of ssout.full so that every level paired
    with a NIST level carries the NIST (2S+1, L). The original is kept as ssout.full.ss
    (made once, never overwritten). Energies, eigenvectors, the LS term table and the
    radiative tables are untouched; the E1 rows refer to levels by index K, so they stay
    consistent. Returns the number of changed rows."""
    from .ssout import LEVEL_HEADER, TERMS_MARKER
    path = os.path.join(iondir, "ssout.full")
    backup = path + ".ss"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(backup) as f:
        text = f.read()
    cfgs = parse_configs(text)
    levels = parse_levels(text, cfgs)
    n = relabel_from_nist(levels, ion, iondir, cfgs, data_dir, match_key)
    new = {lv.K: (lv.twoSp1, lv.L) for lv in levels}
    lines = text.splitlines(keepends=True)
    # the LAST level table in the file is the Breit-Pauli one that parse_levels keeps
    starts = [i for i, l in enumerate(lines) if LEVEL_HEADER in l and "K*CM" in l]
    changed = 0
    if starts:
        row = re.compile(r"^(\s*)(\d+)(\s+\d+\s+[\d.]+\s+)(\d)(\s+)(\d)(\s+\d+\s+\d+\s+)")
        for i in range(starts[-1] + 1, len(lines)):
            if TERMS_MARKER in lines[i]:
                break
            m = row.match(lines[i])
            if not m:
                continue
            K = int(m.group(2))
            if K in new and (int(m.group(4)), int(m.group(6))) != new[K]:
                S, L = new[K]
                lines[i] = (m.group(1) + m.group(2) + m.group(3) + str(S) + m.group(5) + str(L)
                            + lines[i][m.end(6):])
                changed += 1
    with open(path, "w") as f:
        f.writelines(lines)
    return changed


def build_terms(levels):
    """Group fine-structure levels into LS terms: (2S+1, L, parity, CF) with g-weighted mean energy."""
    groups = defaultdict(list)
    for lv in levels:
        groups[(lv.twoSp1, lv.L, lv.parity, lv.CF)].append(lv)
    terms = []
    for key, lvs in groups.items():
        g = np.array([lv.twoJ + 1 for lv in lvs], float)
        E = float(np.sum(g * np.array([lv.E for lv in lvs])) / g.sum())
        terms.append({"S": key[0], "L": key[1], "p": key[2], "CF": key[3], "E": E, "levels": lvs,
                      "g": int(g.sum())})
    terms.sort(key=lambda t: t["E"])
    return terms


# ----------------------------------------------------------------------------
def write_en_ls(path, ion, cfgs, terms, ip, note):
    let = _seniority(terms)
    with open(path, "w") as f:
        f.write(f"{'-'*26} ssopt {'-'*46}\n\n")
        f.write(f"Ref.  SUPERSTRUCTURE (Eissner et al.), lambda tuned to NIST energies with ssopt\n\n")
        f.write(f"{'-'*79}\n\n {ion.spectroscopic}: LS term energies\n\n{note}\n\n")
        f.write(f" Z = {ion.Z}, no of target/core electrons= {ion.n_electrons - 1:3d}\n\n")
        f.write(" ii) Table of bound state energies (with spectroscopic notation)\n")
        f.write(f"{'-'*70}\n\n")
        if ip is not None:
            f.write(f" Ion ground state = {ip * RY_CM:.1f}, Eo(Ry) = {-ip:.4E} =-IP\n\n")
        f.write(f"{'State':>18s}{'E(Ry)':>19s}\n\n")
        for i, t in enumerate(terms):
            core, outer = _split_outer(cfgs[t["CF"]], ion.core)
            label = f"{let[i]}{t['S']}{LNAME[t['L']]}{'o' if t['p'] < 0 else 'e'}"
            E = t["E"] - ip if ip is not None else t["E"]
            f.write(f"{config_str(core):<10s}( - ) {orb_str(*outer):<4s}{label:<6s}{E:12.5f}\n")


def write_en_fs(path, ion, cfgs, levels, ip, note):
    blocks = defaultdict(list)
    for lv in levels:
        blocks[(lv.twoJ, lv.parity)].append(lv)
    with open(path, "w") as f:
        f.write(f"{'-'*26} ssopt {'-'*46}\n\n")
        f.write("Ref.  SUPERSTRUCTURE (Eissner et al.), lambda tuned to NIST energies with ssopt\n\n")
        f.write(f"{'-'*79}\n\n {ion.spectroscopic} : Fine structure levels\n\n{note}\n\n")
        f.write(" iii) Fine structure energy levels of the ion, with spectroscopic designation,\n")
        f.write("      ordered in J\\pi and energy\n" + "-" * 58 + "\n\n")
        f.write(" Table Explanation:\n\n     T -> configuration of the level minus its outer electron\n")
        f.write("     Nlv=Number of levels\n     nl = Principal and orbital quantum numbers of the outer electron\n")
        f.write("     EQN = Effective Quantum Number.\n" + "-" * 38 + "\n\n")
        f.write(f"  i  T(config SLpi Jt)    nl    J      E(Ry)     EQN  Terms\n\n")
        z = ion.charge + 1
        for (twoJ, par) in sorted(blocks, key=lambda k: (k[0], -k[1])):
            lvs = sorted(blocks[(twoJ, par)], key=lambda l: l.E)
            f.write(f"  Nlv={len(lvs):4d}           J pi = {_frac_J(twoJ):>3s} {'o' if par < 0 else 'e'}\n\n")
            for i, lv in enumerate(lvs, 1):
                core, outer = _split_outer(cfgs[lv.CF], ion.core)
                E = lv.E - ip if ip is not None else lv.E
                eqn = z / np.sqrt(-E) if (ip is not None and E < 0) else float("nan")
                f.write(f"{i:3d} {config_str(core):<10s} ( - ) {'-':>2s} {orb_str(*outer):>4s} "
                        f"{_frac_J(twoJ):>4s} {E:12.5E} {eqn:5.2f}  {lv.twoSp1} {LNAME[lv.L]} "
                        f"{'o' if par < 0 else 'e'}\n")
            f.write("\n")


def write_f_ls(path, ion, cfgs, terms, levels, e1, ip):
    """LS multiplets: sum gf over the fine-structure lines between two terms; S summed; A = g-weighted."""
    K2term = {}
    for ti, t in enumerate(terms):
        for lv in t["levels"]:
            K2term[lv.K] = ti
    mult = defaultdict(lambda: {"gf": 0.0, "S": 0.0, "gA": 0.0})
    for tr in e1:
        a, b = K2term.get(tr.K), K2term.get(tr.KP)
        if a is None or b is None or a == b:
            continue
        lo, hi = (a, b) if terms[a]["E"] <= terms[b]["E"] else (b, a)
        m = mult[(lo, hi)]
        m["gf"] += tr.gf
        m["S"] += abs(tr.S)
        m["gA"] += tr.A * (levels_by_K(levels)[max(tr.K, tr.KP)].twoJ + 1)
    bysym = defaultdict(list)
    for (lo, hi), m in mult.items():
        tl, th = terms[lo], terms[hi]
        bysym[((tl["S"], tl["L"], tl["p"]), (th["S"], th["L"], th["p"]))].append((lo, hi, m))
    with open(path, "w") as f:
        f.write(f"{'-'*26} ssopt {'-'*46}\n\nRef.  SUPERSTRUCTURE (Eissner et al.), lambda tuned with ssopt\n\n")
        f.write(f"{'-'*79}\n\n {ion.spectroscopic}: Oscillator strengths for dipole allowed transitions - LS multiplets\n\n")
        f.write(" Line : (SLP)i - (SLP)j -> (2S+1), total L, pi (0 for even, 1 for odd)\n")
        f.write(" Line : ni nj ntrns -> number of terms in symmetries i, j and number of transitions\n")
        f.write(" Lines: i,j,wl(A),Ei(Ry),Ej(Ry),f,S,A(/s)   (f<0: first state is lower)\n")
        f.write(f"{'-'*70}\n{ion.Z:5d}{ion.n_electrons - 1:5d}\n")
        for (si, sj), rows in bysym.items():
            f.write(f"{si[0]:5d}{si[1]:5d}{0 if si[2] > 0 else 1:5d}{sj[0]:8d}{sj[1]:5d}{0 if sj[2] > 0 else 1:5d}  :isi,li,ipi,isj,lj,ipj\n")
            ni = len({lo for lo, _, _ in rows}); nj = len({hi for _, hi, _ in rows})
            f.write(f"{ni:5d}{nj:5d}{len(rows):6d}  : ni,nj,ntrs/i,j,wl(A),Ei(Ry),Ej(Ry),f,S,A(/s)\n")
            for lo, hi, m in sorted(rows, key=lambda r: (terms[r[0]]["E"], terms[r[1]]["E"])):
                tl, th = terms[lo], terms[hi]
                dE = th["E"] - tl["E"]
                wl = 1e8 / (dE * RY_CM) if dE > 0 else float("inf")
                Ei = tl["E"] - ip if ip is not None else tl["E"]
                Ej = th["E"] - ip if ip is not None else th["E"]
                fval = -m["gf"] / tl["g"]
                A = m["gA"] / th["g"]
                f.write(f"{lo + 1:5d}{hi + 1:5d} {wl:11.5E} {Ei:12.5E} {Ej:12.5E} {fval:10.3E} {m['S']:10.3E} {A:10.3E}\n")


def levels_by_K(levels):
    return {lv.K: lv for lv in levels}


def write_f_fs(path, ion, cfgs, terms, levels, e1, ip):
    """NORAD f.fs layout: per LS multiplet a header (term labels, Ei Ef in Ry-from-limit,
    dE, gi gf f S A summed) followed by fine-structure sub-rows in cm-1 above ground."""
    byK = levels_by_K(levels)
    K2term = {lv.K: ti for ti, t in enumerate(terms) for lv in t["levels"]}
    let = _seniority(terms)
    mult = defaultdict(list)
    for tr in e1:
        a, b = K2term.get(tr.K), K2term.get(tr.KP)
        if a is None or b is None or a == b:
            continue
        lo, hi = (a, b) if terms[a]["E"] <= terms[b]["E"] else (b, a)
        mult[(lo, hi)].append(tr)
    with open(path, "w") as f:
        ipstr = f"{ip:.5f}" if ip is not None else "n/a"
        f.write(f" I.P.(Ry)= {ipstr}  NOTE: energies from SUPERSTRUCTURE (ssopt-tuned); * = unmatched level\n\n")
        f.write(" TRANSITION        Ei       Ef      EDIFF    gi  gf    fif        S         Aji\n")
        f.write("                  cm-1     cm-1      Ry                                     sec-1\n")
        f.write("_" * 80 + "\n\n")
        for (lo, hi), trs in sorted(mult.items(), key=lambda kv: (terms[kv[0][0]]["E"], terms[kv[0][1]]["E"])):
            tl, th = terms[lo], terms[hi]
            lab = (f"{let[lo]}{tl['S']}{LNAME[tl['L']]}{'o' if tl['p'] < 0 else 'e'}->"
                   f"{let[hi]}{th['S']}{LNAME[th['L']]}{'o' if th['p'] < 0 else 'e'}")
            gf = sum(t.gf for t in trs); S = sum(abs(t.S) for t in trs)
            gA = sum(t.A * (byK[max(t.K, t.KP)].twoJ + 1) for t in trs)
            dE = th["E"] - tl["E"]
            Ei = (ip - tl["E"]) if ip is not None else tl["E"]
            Ef = (ip - th["E"]) if ip is not None else th["E"]
            f.write(f" {lab:<14s}{Ei:9.4f}{Ef:9.4f}  {dE:9.3E}{tl['g']:4d}{th['g']:4d}  {gf / tl['g']:9.3E}  "
                    f"{S:9.3E}  {gA / th['g']:9.3E}\n\n")
            for t in sorted(trs, key=lambda t: (byK[min(t.K, t.KP)].E, byK[max(t.K, t.KP)].E)):
                l1, l2 = byK[min(t.K, t.KP)], byK[max(t.K, t.KP)]
                f.write(f"{l1.E * RY_CM:13.3f}{l2.E * RY_CM:13.3f}  {l2.E - l1.E:9.3E}{l1.twoJ + 1:4d}{l2.twoJ + 1:4d}"
                        f"  {t.gf / (l1.twoJ + 1):9.3E}  {abs(t.S):9.3E}  {t.A:9.3E}\n")
            f.write("\n")


def write_f_fs_index(path, ion, cfgs, levels, e1, ip):
    """NORAD f.fs layout used for the Li-/He-like files (e.g. Ne VIII, O VI, O VII):
    a numbered level table  i j p iiii E(Ry) conf term jjpiiii, then transitions grouped by
    (2J+1, pi) symmetry pairs as  i j wl(A) Ei(Ry) Ej(Ry) f S A  with level indices within
    each symmetry (f negative when i is the lower level)."""
    byK = levels_by_K(levels)
    sym = defaultdict(list)
    for lv in sorted(levels, key=lambda l: l.E):
        sym[(lv.twoJ, lv.parity)].append(lv)
    idx = {}          # K -> (symmetry key, 1-based index within symmetry)
    for key, lvs in sym.items():
        for i, lv in enumerate(lvs, 1):
            idx[lv.K] = (key, i)
    with open(path, "w") as f:
        f.write(f"{'-'*26} ssopt {'-'*46}\n\nRef.  SUPERSTRUCTURE (Eissner et al.), lambda tuned with ssopt\n\n")
        f.write(f"{'-'*79}\n\n {ion.spectroscopic}: fine structure levels and E1 f-values (level-index layout)\n\n")
        f.write(" jj=2 x total angular momentum j, p=parity, iiii=level position\n\n")
        f.write(f"{'i':>5s} {'j':>5s} p {'iiii':>5s} {'E(Ry)':>13s}   conf         LS term      jjpiiii\n\n")
        n = 0
        for key in sorted(sym, key=lambda k: (k[0], -k[1])):
            for i, lv in enumerate(sym[key], 1):
                n += 1
                E = lv.E - ip if ip is not None else lv.E
                pc = 0 if lv.parity > 0 else 1
                f.write(f"{n:5d} {lv.twoJ / 2:5.1f} {'e' if lv.parity > 0 else 'o'} {i:5d} {E:13.5E}   "
                        f"{config_str({o: c for o, c in cfgs[lv.CF].items() if o not in ion.core}):<12s} "
                        f"{lv.twoSp1}{LNAME[lv.L]}{'e' if lv.parity > 0 else 'o':<8s} "
                        f"{lv.twoJ:2d}{pc:1d}{i:04d}\n")
        f.write(f"\n{'-'*40}\n\n Line : 2J+1, pi (0 even, 1 odd), 2J'+1, pi', Ni, Nj, NN\n"
                " Lines: i, j, wl(A), Ei(Ry), Ej(Ry), f, S, A   (f < 0: level i is the lower one)\n\n")
        f.write(f"{ion.Z:5d}{ion.n_electrons:5d} : Z, Number of electrons\n\n")
        f.write(f"{'i':>5s}{'j':>5s}{'wl(A)':>11s}{'Ei(Ry)':>13s}{'Ej(Ry)':>13s}{'fL':>12s}{'S':>12s}{'Aji(s-1)':>12s}\n\n")
        pairs = defaultdict(list)
        for tr in e1:
            a, b = byK.get(tr.K), byK.get(tr.KP)
            if a is None or b is None:
                continue
            lo, hi = (a, b) if a.E <= b.E else (b, a)
            pairs[(idx[lo.K][0], idx[hi.K][0])].append((lo, hi, tr))
        for (ki, kj), rows in sorted(pairs.items()):
            f.write(f"{ki[0] + 1:5d}{0 if ki[1] > 0 else 1:5d}{kj[0] + 1:6d}{0 if kj[1] > 0 else 1:5d}"
                    f"{len(sym[ki]):6d}{len(sym[kj]):5d}{len(rows):6d} : gi Pi gf Pf Ni Nf NN\n")
            for lo, hi, tr in sorted(rows, key=lambda r: (idx[r[0].K][1], idx[r[1].K][1])):
                dE = hi.E - lo.E
                wl = 1e8 / (dE * RY_CM) if dE > 0 else float("inf")
                Ei = lo.E - ip if ip is not None else lo.E
                Ej = hi.E - ip if ip is not None else hi.E
                f.write(f"{idx[lo.K][1]:5d}{idx[hi.K][1]:5d}{wl:11.2f}{Ei:13.4E}{Ej:13.4E}"
                        f"{-tr.gf / (lo.twoJ + 1):12.3E}{abs(tr.S):12.3E}{tr.A:12.3E}\n")
            f.write("\n")


# ----------------------------------------------------------------------------
def export(spectrum: str, workdir: str, data_dir: str, out: str):
    ion = build_ion(spectrum)
    iondir = os.path.join(workdir, ion.tag)
    text, cfgs, levels, e1 = load_run(iondir)
    n_relabel = relabel_from_nist(levels, ion, iondir, cfgs, data_dir)
    ip = nist_ionization_limit(data_dir, ion.spectroscopic)
    note = (f" Energies relative to the NIST ionization limit {ip:.5f} Ry (bound states negative)."
            if ip is not None else " No NIST ionization limit cached: energies are relative to the ground level.")
    terms = build_terms(levels)
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, ion.tag)
    write_en_ls(base + ".en.ls.txt", ion, cfgs, terms, ip, note)
    write_en_fs(base + ".en.fs.txt", ion, cfgs, levels, ip, note)
    write_f_ls(base + ".f.ls.txt", ion, cfgs, terms, levels, e1, ip)
    write_f_fs_index(base + ".f.fs.txt", ion, cfgs, levels, e1, ip)      # Li-/He-like NORAD layout
    write_f_fs(base + ".f.fsm.txt", ion, cfgs, terms, levels, e1, ip)     # multiplet-header layout (Si I style)
    print(f"{ion.spectroscopic}: {len(levels)} levels, {len(terms)} terms, {len(e1)} E1 lines, "
          f"{n_relabel} level labels taken from NIST, "
          f"IP={'%.4f Ry' % ip if ip is not None else 'n/a'} -> {base}.{{en.ls,en.fs,f.ls,f.fs,f.fsm}}.txt")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ions", nargs="+", help='e.g. "Ne V" "Ne VI"  or  all')
    p.add_argument("--workdir", default="work_rel")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="norad_ss")
    p.add_argument("--relabel-ssout", action="store_true",
                   help="also rewrite the level table in work/<ion>/ssout.full with the NIST LS labels "
                        "(original kept as ssout.full.ss)")
    a = p.parse_args(argv)
    ions = a.ions
    if ions == ["all"]:
        ions = [json.load(open(os.path.join(a.workdir, d, "deck.json")))["ion"] for d in sorted(os.listdir(a.workdir))
                if os.path.exists(os.path.join(a.workdir, d, "ssout.full"))]
    for s in ions:
        try:
            if a.relabel_ssout:
                ion = build_ion(s)
                n = relabel_ssout(os.path.join(a.workdir, ion.tag), ion, a.data)
                print(f"{ion.spectroscopic}: {n} rows of the ssout.full level table relabelled from NIST "
                      f"(original: {a.workdir}/{ion.tag}/ssout.full.ss)")
            export(s, a.workdir, a.data, a.out)
        except FileNotFoundError as e:
            print(f"{s}: skipped ({e})")




def _decay_scorer(levels, matches, nl, ion, iondir, data_dir, margin=0.15):
    """Return f(lv, d, lv2, d2) -> True (swap labels), False (keep), None (no evidence).

    E1 decays obey the LS selection rules for the dominant component of a level:
    dS = 0 and |dL| <= 1 (not L = 0 -> 0). A level that carries the wrong label therefore
    decays like the other term: a 3S1 sends its rate to 3P0,1,2, a 1P1 to 1D2 and 1S0;
    a 3D3 decays to 3P2, a 3F3 hardly does. For SS level K and a candidate term (S, L),
    allowed(K, S, L) is the fraction of K's E1 rate that goes to lower levels it could
    reach in LS coupling with that term (lower levels take the NIST label when matched,
    otherwise the SUPERSTRUCTURE label). The labels are swapped when the NIST-paired
    assignment explains the decays better than the label-respecting one by more than
    `margin` (summed fractions), kept when it is the other way round, and left to the
    energy rule when neither level has E1 decays to compare."""
    from .ssout import parse_e1
    src = os.path.join(iondir, "ssout.full.ss")
    if not os.path.exists(src):
        src = os.path.join(iondir, "ssout.full")
    try:
        with open(src, errors="replace") as f:
            e1 = parse_e1(f.read())
    except FileNotFoundError:
        return None
    if not e1:
        return None
    term_of = {lv.K: (lv.twoSp1, lv.L) for lv in levels}
    for m in matches:                                    # trust NIST for matched levels
        d = nl[m.nist_idx]
        if d.twoSp1 is not None and d.L is not None and "?" not in d.term:
            term_of[levels[m.comp_idx].K] = (d.twoSp1, d.L)
    out = defaultdict(list)                              # K -> [(lower K, A)]
    for tr in e1:
        if tr.A and tr.A > 0:
            out[tr.K].append((tr.KP, tr.A))

    def allowed(K, S, L):
        tot = sum(A for _, A in out.get(K, []))
        if tot <= 0:
            return None
        ok = sum(A for Kp, A in out[K] if Kp in term_of and term_of[Kp][0] == S
                 and abs(term_of[Kp][1] - L) <= 1 and not (L == 0 and term_of[Kp][1] == 0))
        return ok / tot

    def verdict(lv, d, lv2, d2):
        X, Y = (lv.twoSp1, lv.L), (d.twoSp1, d.L)
        sw = [allowed(lv.K, *Y)] + ([allowed(lv2.K, *X)] if lv2 is not None else [])
        kp = [allowed(lv.K, *X)] + ([allowed(lv2.K, *Y)] if lv2 is not None else [])
        pairs = [(a, b) for a, b in zip(sw, kp) if a is not None and b is not None]
        if not pairs:
            return None
        s_sw, s_kp = sum(a for a, _ in pairs), sum(b for _, b in pairs)
        if s_sw > s_kp + margin:
            return True
        if s_kp > s_sw + margin:
            return False
        return None
    return verdict

if __name__ == "__main__":
    main()
