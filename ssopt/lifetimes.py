"""Level lifetimes from a full SUPERSTRUCTURE run, and comparison with measured values.

    python -m ssopt.lifetimes "Fe XVII" --workdir work_rel                 # every level with tau < 1 s
    python -m ssopt.lifetimes "Fe XVII" --workdir work_rel --all           # every level
    python -m ssopt.lifetimes all --workdir work_rel --bench benchmarks/lifetimes_exp.csv
                                                                           # measured vs computed

tau(K) = 1 / sum over all lower levels KP of [A_E1 + A_E2 + A_M1 + A_M2 + A_E3](K -> KP),
using every rate SUPERSTRUCTURE prints in the INTERMEDIATE-COUPLING RADIATIVE DATA
section: part 1 (parity changing: E1 rows, and 'E3:' / 'AM2 =' rows for the higher
multipoles) and part 2 (same parity: AQUAD = E2 and AMAG = M1). Energies and labels
come from the level table of ssout.full (use --relabel-ssout first so the labels are
NIST-consistent, which is how the benchmark file names levels).

Benchmark CSV columns: ion, config, term, J, tau, unc, unit, method, reference
  config in NIST style ("2s2.2p5.3s"), term with parity star ("3P*"), J as 0, 1, 1.5;
  unit one of fs ps ns us ms s.  Several rows for one level (different experiments) are fine.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

from .ions import build_ion, config_str
from .ssout import RAD_MARKER, parse_configs, parse_levels, parse_e1

PART2 = "DATA FOR ELECTRIC QUADRUPOLE AND MAGNETIC DIPOLE"
_E = r"([\d.]+E[+-]\d+)"
_ROW2 = re.compile(r"^\s*\d+\s+(?:E0:)?\s*(\d+)\s+(\d+)\s+" + _E + r"\s+" + _E)        # IOS [E0:] K KP AQUAD AMAG
_E3 = re.compile(r"^\s*E3:\s*(\d+)\s+(\d+)\s+" + _E + r"(?:\s+AM2\s*=\s*" + _E + ")?")  # E3: K KP AE3 [AM2 = x]
_M2 = re.compile(r"^\s*(\d+)\s+(\d+)\s+AM2\s*=\s*" + _E)                                # K KP AM2 = x
UNIT = {"fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
LN = "SPDFGHIK"


def parse_rates(text: str) -> dict[tuple[int, int], dict[str, float]]:
    """(K_upper, KP_lower) -> {'E1': A, 'E2': A, 'M1': A, 'M2': A, 'E3': A} in s^-1."""
    rates: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    start = text.find(RAD_MARKER)
    if start < 0:
        return rates
    for tr in parse_e1(text):
        rates[(tr.K, tr.KP)]["E1"] = tr.A
    body = text[start:]
    cut = body.find(PART2)
    part1, part2 = (body, "") if cut < 0 else (body[:cut], body[cut:])
    for line in part1.splitlines():
        m = _E3.match(line)
        if m:
            rates[(int(m.group(1)), int(m.group(2)))]["E3"] = float(m.group(3))
            if m.group(4):
                rates[(int(m.group(1)), int(m.group(2)))]["M2"] = float(m.group(4))
            continue
        m = _M2.match(line)
        if m:
            rates[(int(m.group(1)), int(m.group(2)))]["M2"] = float(m.group(3))
    for line in part2.splitlines():
        m = _ROW2.match(line)
        if m:
            k, kp = int(m.group(1)), int(m.group(2))
            rates[(k, kp)]["E2"] = float(m.group(3))
            rates[(k, kp)]["M1"] = float(m.group(4))
    return rates


def lifetimes(rates):
    """K -> (tau in s, total A, {multipole: summed A})."""
    tot: dict[int, float] = defaultdict(float)
    by: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (k, kp), d in rates.items():
        for mp, a in d.items():
            tot[k] += a
            by[k][mp] += a
    return {k: (1.0 / a if a > 0 else math.inf, a, dict(by[k])) for k, a in tot.items()}


def fmt_tau(t):
    if not math.isfinite(t):
        return "stable"
    for u in ("fs", "ps", "ns", "us", "ms", "s"):
        if t < 1000 * UNIT[u] or u == "s":
            return f"{t / UNIT[u]:.3g} {u}"


def level_label(lv, cfgs, core):
    cfg = config_str({o: c for o, c in cfgs[lv.CF].items() if o not in core})
    return cfg, f"{lv.twoSp1}{LN[lv.L]}{'*' if lv.parity < 0 else ''}", lv.twoJ / 2


def load_run(workdir, ion):
    iondir = os.path.join(workdir, ion.tag)
    text = open(os.path.join(iondir, "ssout.full"), errors="replace").read()
    cfgs = parse_configs(text)
    return cfgs, parse_levels(text, cfgs), parse_rates(text)


def _norm_cfg(s: str) -> str:
    """'2s2.2p5.3s' / '2s2 2p5 3s1' / '2s22p53s' -> canonical '2s2 2p5 3s1'."""
    parts = re.findall(r"(\d)([spdfghik])(\d*)", s.lower())
    return " ".join(f"{n}{l}{occ or 1}" for n, l, occ in parts)


def report_ion(ion, workdir, all_levels=False, tmax=1.0):
    cfgs, levels, rates = load_run(workdir, ion)
    tau = lifetimes(rates)
    print(f"\n== {ion.spectroscopic}  ({len(levels)} levels, {len(rates)} transitions with rates)")
    print(f"{'K':>5} {'config':<16}{'term':<5}{'J':>4} {'E(Ry)':>10}  {'tau':>10}   channels (summed A, s^-1)")
    for lv in levels:
        if lv.K == 1:
            continue
        t, a, by = tau.get(lv.K, (math.inf, 0.0, {}))
        if not all_levels and t > tmax:
            continue
        cfg, term, J = level_label(lv, cfgs, ion.core)
        ch = "  ".join(f"{mp}={v:.3g}" for mp, v in sorted(by.items(), key=lambda kv: -kv[1]))
        print(f"{lv.K:5d} {cfg:<16}{term:<5}{J:4g} {lv.E:10.5f}  {fmt_tau(t):>10}   {ch}")


def compare(ions, workdir, bench_path):
    rows = list(csv.DictReader(open(bench_path)))
    want = {b["ion"].strip() for b in rows}
    cache = {}
    out = []
    print(f"\n{'ion':<9}{'level':<28}{'tau_exp':>12} {'unc':>7}  {'tau_SS':>10} {'SS/exp':>7}  method / reference")
    for b in rows:
        name = b["ion"].strip()
        if ions and name not in ions:
            continue
        ion = build_ion(name)
        if name not in cache:
            try:
                cfgs, levels, rates = load_run(workdir, ion)
            except FileNotFoundError:
                print(f"{name:<9}(no ssout.full in {workdir}/{ion.tag})")
                cache[name] = None
                continue
            cache[name] = (cfgs, levels, lifetimes(rates))
        if cache[name] is None:
            continue
        cfgs, levels, tau = cache[name]
        bc, bt, bj = _norm_cfg(b["config"]), b["term"].strip().upper(), float(b["J"])
        cand = [lv for lv in levels
                if _norm_cfg(level_label(lv, cfgs, ion.core)[0]) == bc
                and level_label(lv, cfgs, ion.core)[1].upper() == bt and abs(lv.twoJ / 2 - bj) < 1e-6]
        texp = float(b["tau"]) * UNIT[b["unit"].strip()]
        unc = f"{b['unc']}".strip()
        lab = f"{b['config']} {b['term']} J={b['J']}"
        if not cand:
            print(f"{name:<9}{lab:<28}{fmt_tau(texp):>12} {unc:>7}  {'no such level':>10}          {b.get('method','')} {b.get('reference','')}")
            continue
        lv = cand[0]
        t = tau.get(lv.K, (math.inf, 0, {}))[0]
        ratio = t / texp if math.isfinite(t) else math.inf
        out.append(ratio)
        print(f"{name:<9}{lab:<28}{fmt_tau(texp):>12} {unc:>7}  {fmt_tau(t):>10} {ratio:7.2f}  {b.get('method','')} {b.get('reference','')}")
    if out:
        r = sorted(out)
        print(f"\n{len(out)} levels: median SS/exp = {r[len(r) // 2]:.2f}, "
              f"{100 * sum(abs(math.log10(x)) < math.log10(1.2) for x in r) / len(r):.0f}% within 20%, "
              f"{100 * sum(abs(math.log10(x)) < math.log10(1.5) for x in r) / len(r):.0f}% within 50%")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ions", nargs="+", help='"Fe XVII" ...  or  all')
    p.add_argument("--workdir", default="work_rel")
    p.add_argument("--bench", default=None, help="CSV of measured lifetimes (see module doc)")
    p.add_argument("--all", action="store_true", help="list every level, not only tau < --tmax")
    p.add_argument("--tmax", type=float, default=1.0, help="list levels with tau below this (s)")
    a = p.parse_args(argv)
    ions = None if a.ions == ["all"] else [build_ion(s).spectroscopic for s in a.ions]
    if a.bench:
        compare(ions, a.workdir, a.bench)
        return
    if ions is None:
        ions = [json.load(open(os.path.join(a.workdir, d, "deck.json")))["ion"] for d in sorted(os.listdir(a.workdir))
                if os.path.exists(os.path.join(a.workdir, d, "ssout.full"))]
    for s in ions:
        report_ion(build_ion(s), a.workdir, a.all, a.tmax)


if __name__ == "__main__":
    main()
