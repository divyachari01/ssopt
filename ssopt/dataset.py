"""Collect everything under work/<tag>/ into a flat, model-ready dataset.

    python -m ssopt.dataset --workdir work --out dataset/

writes (one row per ... ):
  ions.csv          ion, Z, N, charge, sequence, core, valence, n_configs, n_targets,
                    rms_l, rms_orbital, mean_diff, rms_about_mean, lambdas_*, sensitivities
  levels.csv        ion, NIST config/term/J/E, computed E (tuned), diff, how, dominant CF config
  samples.csv       ion, mode, tag, λ_1..λ_n, rms, n_matched          (every structx run)
  level_samples.csv ion, mode, sample_id, nist_idx, E_comp             (the surrogate's raw data)
  prune.csv         ion, group, n_configs_removed, rms, delta_rms
  avalues.csv       ion, lower, upper, wavelength, gf_SS, LV, A_SS, A_NIST, acc, A_NORAD, ratios
  configs.csv       ion, config (valence), tag spec/corr, excitation level, parity, prune ΔRMS of its group

Everything is derived from files ssopt already writes; nothing is recomputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

from .configs import excitation_level, full_config, prune_groups, DeckPlan
from .ions import build_ion, config_str, parity
from .surrogate import History

ISOELECTRONIC = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
                 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
                 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
                 27: "Co", 28: "Ni"}


def _load_deck(iondir):
    with open(os.path.join(iondir, "deck.json")) as f:
        d = json.load(f)
    configs = [{tuple(o): n for o, n in c} for c in d["configs"]]
    return d, DeckPlan(configs, d["tags"], d["n_max"], d["l_max"])


def _parse_levels_txt(path):
    rows = []
    if not os.path.exists(path):
        return rows, {}
    with open(path) as f:
        lines = f.read().splitlines()
    for ln in lines[1:]:
        if ln.startswith("matched"):
            m = re.search(r"matched (\d+)\s+RMS = ([\d.]+).*mean diff = ([-+\d.]+).*RMS about mean = ([\d.]+)"
                          r".*RMS without worst level = ([\d.]+)", ln)
            summ = {"n": int(m.group(1)), "rms": float(m.group(2)), "mean_diff": float(m.group(3)),
                    "rms_about_mean": float(m.group(4)), "rms_wo_worst": float(m.group(5))} if m else {}
            return rows, summ
        p = ln.split()
        if len(p) < 7:
            continue
        # columns: config term J E_nist E_comp diff how [SS config (CF)]
        try:
            j = float(p[2]); en = float(p[3]); ec = float(p[4]); df = float(p[5])
        except ValueError:
            continue
        how = p[6][:4] if p[6][:4] in ("cfg", "swap", "term", "jK") else p[6]
        ss = " ".join(p[7:]) if len(p) > 7 else ""
        rows.append({"config": p[0], "term": p[1], "J": j, "E_nist": en, "E_comp": ec, "diff": df,
                     "how": how, "ss_config": ss})
    return rows, {}


def _parse_prune(path):
    out = []
    if not os.path.exists(path):
        return out
    for ln in open(path).read().splitlines()[1:]:
        p = ln.split()
        if len(p) >= 5:
            try:
                out.append({"group": p[0], "n_configs": int(p[1]), "matched": int(p[2]),
                            "rms": float(p[3]), "delta_rms": float(p[4])})
            except ValueError:
                pass
    return out


def _parse_avalues(path):
    out = []
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("lower")), None)
    if start is None:
        return out
    for ln in lines[start + 2:]:
        # fixed-width: lower(20) upper(20) then numbers
        lower, upper, rest = ln[:20].strip(), ln[20:40].strip(), ln[40:].split()
        if len(rest) < 8:
            continue
        def num(x):
            try:
                return float(x)
            except ValueError:
                return float("nan")
        # rest: wavel gf LV A_SS A_NIST [acc] A_NORAD SS/NIST SS/NORAD  (acc may be absent)
        if len(rest) == 9:
            wl, gf, lv, a_ss, a_nist, acc, a_norad, r1, r2 = rest
        else:
            wl, gf, lv, a_ss, a_nist, a_norad, r1, r2 = rest[:8]; acc = ""
        out.append({"lower": lower, "upper": upper, "wavelength": num(wl), "gf_SS": num(gf), "LV": num(lv),
                    "A_SS": num(a_ss), "A_NIST": num(a_nist), "acc": acc, "A_NORAD": num(a_norad),
                    "ratio_nist": num(r1), "ratio_norad": num(r2)})
    return out


def _sens(logpath, mode):
    if not os.path.exists(logpath):
        return {}
    txt = open(logpath, errors="replace").read()
    hits = re.findall(r"sensitivity dRMS/dλ: (.*)", txt)
    if not hits:
        return {}
    # last sensitivity line whose λ count matches the mode is the one we want; keep the last overall
    return {k: float(v) for k, v in re.findall(r"(\w+)=([-+\d.]+)", hits[-1])}


def collect(workdir: str, out: str):
    os.makedirs(out, exist_ok=True)
    ions, levels, samples, lsamples, prune, av, cfgs = [], [], [], [], [], [], []
    for tag in sorted(os.listdir(workdir)):
        iondir = os.path.join(workdir, tag)
        if not os.path.isdir(iondir) or not os.path.exists(os.path.join(iondir, "deck.json")):
            continue
        d, plan = _load_deck(iondir)
        ion = build_ion(d["ion"])
        row = {"ion": ion.spectroscopic, "tag": tag, "Z": ion.Z, "N": ion.n_electrons, "charge": ion.charge,
               "sequence": ISOELECTRONIC.get(ion.n_electrons, str(ion.n_electrons)),
               "core": config_str(ion.core), "valence": config_str(ion.valence),
               "n_configs": len(plan), "n_max": plan.n_max, "l_max": plan.l_max}
        best_mode = None
        for mode in ("l", "orbital"):
            hp = os.path.join(iondir, f"history_{mode}.json")
            if not os.path.exists(hp):
                continue
            h = History.load(hp)
            b = h.best()
            if b is None:
                continue
            row[f"rms_{mode}"] = b.rms
            row[f"n_runs_{mode}"] = len(h.samples)
            row[f"lambdas_{mode}"] = " ".join(f"{n}={v:.4f}" for n, v in zip(h.lambda_names, b.lam))
            for i, s in enumerate(h.samples):
                samples.append({"ion": ion.spectroscopic, "mode": mode, "sample_id": i, "tag": s.tag,
                                "ok": s.ok, "rms": s.rms, "n_matched": s.n_matched,
                                **{f"lam_{n}": v for n, v in zip(h.lambda_names, s.lam)}})
                for k, e in s.E.items():
                    lsamples.append({"ion": ion.spectroscopic, "mode": mode, "sample_id": i,
                                     "nist_idx": k, "E_comp": e})
            best_mode = mode
        for mode in ("l", "orbital"):
            rows, summ = _parse_levels_txt(os.path.join(iondir, f"levels_{mode}.txt"))
            if summ:
                row[f"n_targets_{mode}"] = summ["n"]
                row[f"mean_diff_{mode}"] = summ["mean_diff"]
                row[f"rms_about_mean_{mode}"] = summ["rms_about_mean"]
                row[f"rms_wo_worst_{mode}"] = summ["rms_wo_worst"]
            if mode == best_mode:
                for r in rows:
                    levels.append({"ion": ion.spectroscopic, "mode": mode, **r})
        sens = _sens(os.path.join(iondir, "log.txt"), best_mode)
        row["sensitivities"] = " ".join(f"{k}={v:+.4f}" for k, v in sens.items())
        pr = _parse_prune(os.path.join(iondir, "prune.txt"))
        for r in pr:
            prune.append({"ion": ion.spectroscopic, **r})
        pmap = {r["group"]: r["delta_rms"] for r in pr}
        groups = prune_groups(plan, ion)
        cfg2group = {i: g for g, idx in groups.items() for i in idx}
        ref = full_config(ion, ion.valence)
        for i, (c, t) in enumerate(zip(plan.configs, plan.tags)):
            v = {o: n for o, n in c.items() if o not in ion.core}
            g = cfg2group.get(i, "")
            cfgs.append({"ion": ion.spectroscopic, "config": config_str(v), "tag": t,
                         "excitation": excitation_level(c, ref), "parity": "o" if parity(c) < 0 else "e",
                         "group": g, "group_delta_rms": pmap.get(g, "")})
        for r in _parse_avalues(os.path.join(iondir, "avalues.txt")):
            av.append({"ion": ion.spectroscopic, **r})
        ions.append(row)

    def write(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  {name}: {len(rows)} rows")

    print(f"dataset -> {out}")
    write("ions.csv", ions)
    write("levels.csv", levels)
    write("samples.csv", samples)
    write("level_samples.csv", lsamples)
    write("prune.csv", prune)
    write("avalues.csv", av)
    write("configs.csv", cfgs)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="work")
    p.add_argument("--out", default="dataset")
    a = p.parse_args(argv)
    collect(a.workdir, a.out)


if __name__ == "__main__":
    main()
