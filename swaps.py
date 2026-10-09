"""List every level whose LS label was changed by --relabel-ssout, with its NIST partner.
    python swaps.py work_rel > swaps.txt
"""
import os, json, sys
from ssopt.ions import build_ion
from ssopt.ssout import parse_configs, parse_levels
from ssopt import match as M, nist

W = sys.argv[1] if len(sys.argv) > 1 else "work_rel"
LN = "SPDFGHIK"


def lab(l):
    return "%d%s%g" % (l.twoSp1, LN[l.L], l.twoJ / 2)


total = 0
for tag in sorted(os.listdir(W)):
    d = os.path.join(W, tag)
    if not os.path.exists(os.path.join(d, "ssout.full.ss")):
        continue
    deck = json.load(open(os.path.join(d, "deck.json")))
    ion = build_ion(deck["ion"])
    orig = open(os.path.join(d, "ssout.full.ss")).read()
    new = open(os.path.join(d, "ssout.full")).read()
    cf = parse_configs(orig)
    L0 = parse_levels(orig, cf)
    L1 = {l.K: l for l in parse_levels(new, cf)}
    changed = [i for i, l in enumerate(L0) if (l.twoSp1, l.L) != (L1[l.K].twoSp1, L1[l.K].L)]
    if not changed:
        continue
    nl = nist.load_levels("data", ion.spectroscopic, fetch_if_missing=False)
    configs = [{tuple(o): n for o, n in c} for c in deck["configs"]]
    mm = {m.comp_idx: m for m in M.match_levels(L0, nl, ion, configs, cf, targets=M.nist_targets(nl, ion, configs))}
    total += len(changed)
    print("\n== %s: %d levels relabelled" % (ion.spectroscopic, len(changed)))
    print("%5s %10s  %-7s -> %-7s | NIST: %-15s %-6s %-4s %10s   diff" % ("K", "E_SS(Ry)", "SS", "new", "config", "term", "J", "E_NIST(Ry)"))
    for i in changed:
        a, b = L0[i], L1[L0[i].K]
        m = mm.get(i)
        if m:
            dd = nl[m.nist_idx]
            part = "%-15s %-6s %-4g %10.5f %+6.1f%%" % (dd.config, dd.term, dd.J, dd.E, 100 * (a.E - dd.E) / max(dd.E, 1e-9))
        else:
            part = "(no NIST partner)"
        print("%5d %10.5f  %-7s -> %-7s | %s" % (a.K, a.E, lab(a), lab(b), part))
print("\n%d levels relabelled in total" % total)
