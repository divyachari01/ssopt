"""Pair computed J-levels with NIST levels.

Pure nearest-energy matching inside (parity, 2S+1, L, 2J) buckets fails
whenever NIST has levels the deck cannot produce (5s, 5p, 5d ... when the deck
stops at n=4): the deck's levels get stolen by those and everything cascades.
So matching is configuration-aware:

  pass 1  bucket = (parity, 2S+1, L, 2J); computed candidates limited to the
          configurations NIST names inside that bucket; greedy nearest-energy,
          one-to-one. Only NIST levels whose configuration is in the deck are
          targets at all. Labels may be permuted among the bucket's configs
          (strongly mixed pairs come out of SUPERSTRUCTURE with swapped
          dominant configurations while the energies are fine) — reported as
          how='swap'; strict_config=True forbids it.
  pass 2  NIST levels without an LS term (jK-coupled Rydberg levels such as
          3s2.3p.4f 2[5/2]) are matched by (parity, 2J, configuration).
  pass 3  (optional, off by default) leftovers by quantum numbers only inside
          an energy window — for strongly mixed levels whose dominant CF in
          SUPERSTRUCTURE differs from the NIST label.

"valence configuration" = configuration minus the ion's frozen core, so the
NIST string '3s2.3p.4s' and the deck's 1s2 2s2 2p6 3s2 3p 4s compare equal.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .ions import Ion, config_key
from .nist import NistLevel
from .ssout import Level


@dataclass
class Match:
    nist_idx: int
    comp_idx: int
    key: tuple
    E_nist: float
    E_comp: float
    how: str = "cfg"

    @property
    def diff(self) -> float:
        return self.E_comp - self.E_nist


def valence_key(cfg: dict, core: dict) -> tuple:
    v = {}
    for o, occ in cfg.items():
        r = occ - core.get(o, 0)
        if r > 0:
            v[o] = r
    return config_key(v)


def deck_valence_keys(configs: list[dict], core: dict) -> set[tuple]:
    return {valence_key(c, core) for c in configs}


def nist_targets(nist: list[NistLevel], ion: Ion, configs: list[dict], *,
                 e_max: float | None = None, require_ls: bool = False) -> list[int]:
    """Indices of NIST levels the deck can in principle reproduce."""
    keys = deck_valence_keys(configs, ion.core)
    out = []
    for i, d in enumerate(nist):
        if not d.cfg or d.parity == 0:
            continue
        if require_ls and (d.twoSp1 is None or d.L is None):
            continue
        if e_max is not None and d.E > e_max:
            continue
        if valence_key(d.cfg, ion.core) not in keys:
            continue
        out.append(i)
    return out


def _assign(pairs, used_n, used_c, window):
    """Optimal one-to-one assignment (min Σ d²) among the candidate pairs of one
    bucket; `pairs` = [(d, i, j), ...]. Falls back to greedy if scipy is absent."""
    pairs = [(d, i, j) for d, i, j in pairs if i not in used_n and j not in used_c
             and (window is None or d <= window)]
    if not pairs:
        return []
    ni = sorted({i for _, i, _ in pairs})
    cj = sorted({j for _, _, j in pairs})
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        linear_sum_assignment = None
    out = []
    if linear_sum_assignment is not None:
        big = 1e6
        C = np.full((len(ni), len(cj)), big)
        ii = {i: a for a, i in enumerate(ni)}
        jj = {j: b for b, j in enumerate(cj)}
        for d, i, j in pairs:
            C[ii[i], jj[j]] = d * d
        r, c = linear_sum_assignment(C)
        for a, b in zip(r, c):
            if C[a, b] < big:
                out.append((ni[a], cj[b]))
    else:
        for d, i, j in sorted(pairs):
            if i in used_n or j in used_c:
                continue
            out.append((i, j))
            used_n.add(i); used_c.add(j)
    for i, j in out:
        used_n.add(i)
        used_c.add(j)
    return out


_greedy = _assign


def match_levels(computed: list[Level], nist: list[NistLevel], ion: Ion,
                 configs: list[dict], cf_configs: dict[int, dict], *,
                 targets: list[int] | None = None, e_max: float | None = None,
                 window: float | None = None, qn_fallback_window: float | None = None,
                 strict_config: bool = False, match_key: str = "j",
                 term_first: bool = True, max_offset: float = 0.10) -> list[Match]:
    """match_key='ls': buckets are (parity, 2S+1, L, 2J) — for LS-coupled light ions.
    match_key='j' : buckets are (parity, 2J) only — for highly charged ions where
    intermediate coupling makes LS labels arbitrary (e.g. Ni XVI 3s3p² ²S½/²P½
    come out of SUPERSTRUCTURE with the labels the other way round from NIST).
    Configurations are still respected exactly as in 'ls' mode; a pair whose
    LS term differs is reported as how='term'.

    term_first (j mode): inside each (parity, 2J) bucket, NIST levels are first
    paired with computed levels of the same configuration AND the same LS term,
    as long as the pair is within `max_offset` (10%) of the NIST energy; only the
    levels left over are paired by energy alone. Without it, a NIST level that is
    the only one listed from a mixed group (Ni XV 3s2 3p 4f 1G4 among the J=4
    levels 3G4, 3F4, 1G4) goes to whichever computed level is nearest in energy,
    even when that level has another term and different decays."""
    if targets is None:
        targets = nist_targets(nist, ion, configs, e_max=e_max)
    comp_vkey = {j: valence_key(cf_configs[lv.CF], ion.core) for j, lv in enumerate(computed)}

    used_n: set[int] = set()
    used_c: set[int] = set()
    out: list[Match] = []

    # pass 1: quantum numbers; computed candidates restricted to the
    # configurations NIST names in that bucket (labels may be permuted among
    # them — strongly mixed pairs like Si I 3s3p3 3D° / 3s2 3p3d 3D° come out of
    # SUPERSTRUCTURE with swapped dominant configurations while the energies
    # are fine; the swap is reported as how='swap').
    def nkey(d):
        return (d.parity, d.twoJ) if match_key == "j" else (d.parity, d.twoSp1, d.L, d.twoJ)

    def ckey(lv):
        return (lv.parity, lv.twoJ) if match_key == "j" else lv.key

    floor = 0.01 * max((nist[i].E for i in targets), default=1.0)
    nb: dict[tuple, list[int]] = defaultdict(list)
    for i in targets:
        d = nist[i]
        if match_key == "ls" and (d.twoSp1 is None or d.L is None):
            continue
        nb[nkey(d)].append(i)
    cb: dict[tuple, list[int]] = defaultdict(list)
    for j, lv in enumerate(computed):
        cb[ckey(lv)].append(j)
    for key, nis in nb.items():
        allowed = {valence_key(nist[i].cfg, ion.core) for i in nis}
        cjs = [j for j in cb.get(key, []) if comp_vkey[j] in allowed]
        if not cjs:
            continue
        if match_key == "j" and term_first:
            tp = [(abs(computed[j].E - nist[i].E), i, j) for i in nis for j in cjs
                  if nist[i].twoSp1 is not None and nist[i].L is not None and "?" not in nist[i].term
                  and comp_vkey[j] == valence_key(nist[i].cfg, ion.core)
                  and (computed[j].twoSp1, computed[j].L) == (nist[i].twoSp1, nist[i].L)
                  and abs(computed[j].E - nist[i].E) <= max_offset * max(nist[i].E, floor)]
            for i, j in _greedy(tp, used_n, used_c, window):
                out.append(Match(i, j, key, nist[i].E, computed[j].E, "cfg"))
        if strict_config:
            pairs = [(abs(computed[j].E - nist[i].E), i, j) for i in nis for j in cjs
                     if comp_vkey[j] == valence_key(nist[i].cfg, ion.core)]
        else:
            pairs = [(abs(computed[j].E - nist[i].E), i, j) for i in nis for j in cjs]
        for i, j in _greedy(pairs, used_n, used_c, window):
            d = nist[i]; lv = computed[j]
            if comp_vkey[j] != valence_key(d.cfg, ion.core):
                how = "swap"
            elif d.twoSp1 is None or d.L is None:
                how = "jK"
            elif (d.twoSp1, d.L) != (lv.twoSp1, lv.L):
                how = "term"
            else:
                how = "cfg"
            out.append(Match(i, j, key, d.E, lv.E, how))

    # pass 2 (ls mode): jK / unlabeled NIST levels by (parity, 2J, configuration)
    nb2: dict[tuple, list[int]] = defaultdict(list)
    for i in targets:
        d = nist[i]
        if match_key == "j" or (d.twoSp1 is not None and d.L is not None):
            continue
        nb2[(d.parity, d.twoJ, valence_key(d.cfg, ion.core))].append(i)
    if nb2:
        cb2: dict[tuple, list[int]] = defaultdict(list)
        for j, lv in enumerate(computed):
            if j in used_c:
                continue
            cb2[(lv.parity, lv.twoJ, comp_vkey[j])].append(j)
        for key, nis in nb2.items():
            cjs = cb2.get(key)
            if not cjs:
                continue
            if strict_config:
                cjs = [j for j in cjs if comp_vkey[j] == key[2]]
            pairs = [(abs(computed[j].E - nist[i].E), i, j) for i in nis for j in cjs]
            for i, j in _greedy(pairs, used_n, used_c, window):
                out.append(Match(i, j, (key[0], None, None, key[1]), nist[i].E, computed[j].E, "jK"))

    # pass 3: quantum numbers only, windowed
    if qn_fallback_window is not None:
        nb3: dict[tuple, list[int]] = defaultdict(list)
        for i in targets:
            d = nist[i]
            if i in used_n or d.twoSp1 is None or d.L is None:
                continue
            nb3[(d.parity, d.twoSp1, d.L, d.twoJ)].append(i)
        cb3: dict[tuple, list[int]] = defaultdict(list)
        for j, lv in enumerate(computed):
            if j not in used_c:
                cb3[lv.key].append(j)
        for key, nis in nb3.items():
            cjs = cb3.get(key)
            if not cjs:
                continue
            pairs = [(abs(computed[j].E - nist[i].E), i, j) for i in nis for j in cjs]
            for i, j in _greedy(pairs, used_n, used_c, qn_fallback_window):
                out.append(Match(i, j, key, nist[i].E, computed[j].E, "qn"))

    out.sort(key=lambda m: m.E_nist)
    return out


def target_weights(E_nist: dict[int, float], objective: str = "abs", floor_frac: float = 0.01) -> dict[int, float]:
    """Per-target weights for the objective.  'abs': 1 for every level (RMS in Ry).
    'rel': 1/max(E, floor) so every level counts by its *percent* error; the floor
    (1% of the highest target energy) keeps the near-degenerate ground fine structure
    from dominating through division by ~0."""
    if objective == "abs":
        return {k: 1.0 for k in E_nist}
    if objective != "rel":
        raise ValueError(f"unknown objective {objective!r}")
    floor = floor_frac * max(E_nist.values()) if E_nist else 1.0
    return {k: 1.0 / max(e, floor) for k, e in E_nist.items()}


def rms(matches: list[Match], weights: dict[int, float] | None = None) -> float:
    """RMS of (weighted) differences. With weights from target_weights(..., 'rel') this is
    the RMS *relative* error (dimensionless); without weights it is in Ry."""
    if not matches:
        return float("nan")
    d = np.array([m.diff * (weights.get(m.nist_idx, 1.0) if weights else 1.0) for m in matches])
    return float(np.sqrt(np.mean(d * d)))


def ambiguous(matches: list[Match], computed, nist, ion, configs, cf_configs, targets, match_key="j") -> set[int]:
    """NIST levels whose computed partner changes when the pairing ignores the LS term:
    for these strongly mixed levels neither the term nor the energy settles which
    computed level is the NIST one, so their lines are marked '?' in the tables."""
    by_energy = {m.nist_idx: m.comp_idx for m in match_levels(computed, nist, ion, configs, cf_configs,
                                                             targets=targets, match_key=match_key,
                                                             term_first=False)}
    return {m.nist_idx for m in matches if by_energy.get(m.nist_idx) != m.comp_idx}


def summary(matches: list[Match], nist: list[NistLevel], computed: list[Level] | None = None,
            cf_configs: dict | None = None, flagged=frozenset()) -> str:
    from .ions import config_str
    lines = [f"{'NIST config':<26}{'term':<12}{'J':>4}{'E_nist':>10}{'E_comp':>10}{'diff':>9}  {'how':<4}{'SS config (CF)'}"]
    for m in matches:
        d = nist[m.nist_idx]
        ss = ""
        if computed is not None and cf_configs is not None:
            cf = computed[m.comp_idx].CF
            ss = f"{config_str({o: n for o, n in cf_configs[cf].items() if o[0] >= 3})} ({cf})"
        lines.append(f"{d.config:<26}{d.term:<12}{d.J:>4.1f}{m.E_nist:>10.5f}{m.E_comp:>10.5f}"
                     f"{m.diff:>+9.5f}  {m.how:<4}{ss}")
    d = np.array([m.diff for m in matches]) if matches else np.zeros(0)
    excl = d[np.argsort(np.abs(d))[:-1]] if len(d) > 1 else d
    rel = rms(matches, target_weights({m.nist_idx: m.E_nist for m in matches}, "rel")) if matches else float("nan")
    lines.append(f"matched {len(matches)}   RMS = {rms(matches):.5f} Ry   "
                 f"mean diff = {d.mean() if len(d) else float('nan'):+.5f}   "
                 f"RMS about mean = {np.sqrt(np.mean((d - d.mean()) ** 2)) if len(d) else float('nan'):.5f}   "
                 f"RMS without worst level = {np.sqrt(np.mean(excl ** 2)) if len(excl) else float('nan'):.5f}   "
                 f"relative RMS = {100 * rel:.2f} %")
    if flagged:
        names = ", ".join(f"{nist[i].config} {nist[i].term} {nist[i].J:g}"
                          for i in sorted(flagged, key=lambda i: nist[i].E))
        lines.append(f"uncertain identifications ({len(flagged)} strongly mixed levels, where pairing by term and "
                     f"by energy give different computed levels): {names}")
    return "\n".join(lines)
