"""λ predictor: learn the tuned per-orbital λ across ions, predict for a new ion.

    python -m ssopt.predict fit      --dataset dataset            # fit, print leave-one-ion-out table
    python -m ssopt.predict predict  --dataset dataset "Ni XV"    # λ for an ion (any ion, tuned or not)
    python -m ssopt.predict verify   --dataset dataset --workdir work --binary ./structx [--workers 48]
                                                                  # ONE structx run per ion at the
                                                                  # leave-one-out prediction -> real RMS

Two models, both trained on ions.csv (one row per tuned ion):

  gp    one Gaussian process over the features (Z, N_electrons, n, l, core?) of every
        (ion, orbital) pair, λ standardised.  Works for any ion, including sequences
        never tuned.  Uses ssopt.gp (numpy only).
  seq   per (isoelectronic sequence, orbital) polynomial in Z — the classic
        isoelectronic-smoothness assumption.  Needs ≥2 tuned ions in the sequence;
        falls back to the gp elsewhere.

Training rows are weighted by 1/(RMS + 0.01): a poorly converged ion says less
about the right λ.  λ that sit on the box boundary are kept (they are still the
best point found) but flagged in the table.

The leave-one-ion-out (LOIO) test is the honest one: for each ion the model is
refit without it, its λ predicted, and — with `verify` — structx is run once at
that λ.  The ratio RMS(predicted λ)/RMS(tuned λ) says how much of the tuning a
new ion still needs.  A ratio near 1 means one run instead of ~150.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import numpy as np

from .gp import MultiOutputGP
from .ions import build_ion, parse_orbital

ORB_RE = re.compile(r"(\d+[spdfg])=([-+\d.]+)")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_ions(dataset: str, mode: str = "orbital") -> list[dict]:
    """Rows of ions.csv with a parsed λ dict {orbital: value}. Ions without the mode are skipped."""
    rows = []
    with open(os.path.join(dataset, "ions.csv")) as f:
        for r in csv.DictReader(f):
            s = r.get(f"lambdas_{mode}", "")
            lam = {k: float(v) for k, v in ORB_RE.findall(s)}
            if not lam:
                continue
            rows.append({"ion": r["ion"], "tag": r["tag"], "Z": int(r["Z"]), "N": int(r["N"]),
                         "charge": int(r["charge"]), "sequence": r["sequence"],
                         "rms": float(r.get(f"rms_{mode}") or "nan"), "lam": lam})
    return rows


def features(Z: int, N: int, orb: str, core_n: int) -> list[float]:
    n, l = parse_orbital(orb)
    return [Z, N, n, l, 1.0 if n <= core_n else 0.0]


def core_n(ion) -> int:
    """Highest principal quantum number in the closed core (0 if no core)."""
    return max((n for (n, l) in ion.core), default=0)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
class LambdaGP:
    """One GP over (Z, N, n, l, core) -> λ.  Ions with RMS far above the median are dropped
    from training (a badly converged ion says little about the right λ).

    Hyperparameters are optimised once (`fit`); `refit(rows)` re-conditions on a different
    training set with the same hyperparameters, which is what the leave-one-ion-out loop needs.
    """

    def __init__(self):
        self.lo = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        self.hi = np.array([30.0, 30.0, 5.0, 3.0, 1.0])
        self.theta = None

    def _x(self, X):
        return (np.asarray(X, float) - self.lo) / (self.hi - self.lo)

    @staticmethod
    def _table(rows):
        X, y, w = [], [], []
        for r in rows:
            ion = build_ion(r["ion"])
            cn = core_n(ion)
            for orb, v in r["lam"].items():
                X.append(features(r["Z"], r["N"], orb, cn)); y.append(v)
                w.append(1.0 / (min(r["rms"], 1.0) + 0.01) if np.isfinite(r["rms"]) else 1.0)
        X, y, w = np.array(X), np.array(y), np.array(w)
        keep = w >= 0.1 * np.median(w)
        return X[keep], y[keep]

    def fit(self, rows: list[dict]):
        X, y = self._table(rows)
        self.mu, self.sd = y.mean(), y.std() + 1e-9
        self.gp = MultiOutputGP(length_scale=0.5, noise=0.1, n_restarts=1).fit(self._x(X), (y - self.mu) / self.sd)
        self.theta = self.gp.theta
        self.n_train = len(y)
        return self

    def refit(self, rows: list[dict]):
        X, y = self._table(rows)
        self.gp = MultiOutputGP().fit_fixed(self._x(X), (y - self.mu) / self.sd, self.theta)
        self.n_train = len(y)
        return self

    def predict(self, Z: int, N: int, orbs: list[str], cn: int):
        X = self._x([features(Z, N, o, cn) for o in orbs])
        mu, sd = self.gp.predict(X, return_std=True)
        return mu[:, 0] * self.sd + self.mu, sd[:, 0] * self.sd


class LambdaSeq:
    """Per (sequence N, orbital): weighted polynomial λ(Z); degree 2 with ≥4 ions, 1 with ≥2."""

    def fit(self, rows: list[dict]):
        self.fits = {}
        bykey: dict[tuple[int, str], list] = {}
        for r in rows:
            for orb, v in r["lam"].items():
                w = 1.0 / (min(r["rms"], 1.0) + 0.01) if np.isfinite(r["rms"]) else 1.0
                bykey.setdefault((r["N"], orb), []).append((r["Z"], v, w))
        for key, pts in bykey.items():
            Z, v, w = (np.array(a, float) for a in zip(*pts))
            deg = 2 if len(pts) >= 4 else (1 if len(pts) >= 2 else 0)
            self.fits[key] = (np.polyfit(Z, v, deg, w=np.sqrt(w)), len(pts))
        return self

    def predict_one(self, Z: int, N: int, orb: str):
        f = self.fits.get((N, orb))
        if f is None:
            return None
        return float(np.polyval(f[0], Z))


def predict_lambdas(rows: list[dict], spectrum: str, orbs: list[str], model: str = "gp", gp: LambdaGP | None = None):
    """λ for `spectrum` in the order `orbs`, from ions in `rows` (which must not contain it for LOIO).
    Pass a hyperparameter-fitted `gp` to re-condition on `rows` without re-optimising."""
    ion = build_ion(spectrum)
    cn = core_n(ion)
    gp = gp.refit(rows) if gp is not None and gp.theta is not None else LambdaGP().fit(rows)
    mu, sd = gp.predict(ion.Z, ion.n_electrons, orbs, cn)
    out = dict(zip(orbs, mu)); unc = dict(zip(orbs, sd)); src = {o: "gp" for o in orbs}
    if model == "seq":
        seq = LambdaSeq().fit(rows)
        for o in orbs:
            v = seq.predict_one(ion.Z, ion.n_electrons, o)
            if v is not None:
                out[o], src[o] = v, "seq"
    # keep inside the physics box the tuning used
    for o in orbs:
        n, l = parse_orbital(o)
        lo, hi = (0.8, 1.6) if n <= cn else (0.7, 2.0)
        out[o] = float(np.clip(out[o], lo, hi))
    return out, unc, src


# ----------------------------------------------------------------------------
# leave-one-ion-out
# ----------------------------------------------------------------------------
def loio(rows: list[dict], model: str = "gp") -> list[dict]:
    res = []
    gp = LambdaGP().fit(rows)       # hyperparameters once; each fold re-conditions without the ion
    for i, r in enumerate(rows):
        train = rows[:i] + rows[i + 1:]
        orbs = list(r["lam"].keys())
        pred, unc, src = predict_lambdas(train, r["ion"], orbs, model, gp=gp)
        err = np.array([pred[o] - r["lam"][o] for o in orbs])
        res.append({"ion": r["ion"], "tag": r["tag"], "Z": r["Z"], "N": r["N"], "sequence": r["sequence"],
                    "rms_tuned": r["rms"], "orbitals": orbs, "lam_tuned": [r["lam"][o] for o in orbs],
                    "lam_pred": [pred[o] for o in orbs], "unc": [unc[o] for o in orbs],
                    "max_abs_err": float(np.abs(err).max()), "rms_err": float(np.sqrt(np.mean(err ** 2)))})
    return res


def write_predictions(res: list[dict], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ion", "tag", "Z", "N", "sequence", "rms_tuned", "lambda_rms_err", "lambda_max_err",
                    "lam_tuned", "lam_pred"])
        for r in res:
            w.writerow([r["ion"], r["tag"], r["Z"], r["N"], r["sequence"], f"{r['rms_tuned']:.5f}",
                        f"{r['rms_err']:.4f}", f"{r['max_abs_err']:.4f}",
                        " ".join(f"{o}={v:.4f}" for o, v in zip(r["orbitals"], r["lam_tuned"])),
                        " ".join(f"{o}={v:.4f}" for o, v in zip(r["orbitals"], r["lam_pred"]))])


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def cmd_fit(a):
    rows = load_ions(a.dataset, a.mode)
    res = loio(rows, a.model)
    print(f"{len(rows)} tuned ions, model={a.model}, leave-one-ion-out:")
    print(f"{'ion':<10}{'seq':>5}{'Z':>4}{'RMS tuned':>11}{'λ err rms':>11}{'λ err max':>11}")
    for r in sorted(res, key=lambda r: (r['N'], r['Z'])):
        print(f"{r['ion']:<10}{r['sequence']:>5}{r['Z']:>4}{r['rms_tuned']:>11.5f}{r['rms_err']:>11.4f}"
              f"{r['max_abs_err']:>11.4f}")
    out = os.path.join(a.dataset, f"predictions_{a.model}.csv")
    write_predictions(res, out)
    print("->", out)


def cmd_predict(a):
    from .configs import build_deck
    from .deck import lambda_names
    from . import nist
    rows = [r for r in load_ions(a.dataset, a.mode) if r["ion"] != build_ion(a.ion).spectroscopic]
    ion = build_ion(a.ion)
    nl = nist.load_levels(a.data, ion.spectroscopic, fetch_if_missing=False)
    plan = build_deck(ion, nl, n_max=a.n_max, l_max=a.l_max)
    orbs = lambda_names(plan.configs, "orbital")
    pred, unc, src = predict_lambdas(rows, a.ion, orbs, a.model)
    print(f"{ion.spectroscopic}: predicted per-orbital λ from {len(rows)} tuned ions ({a.model})")
    for o in orbs:
        print(f"  {o:<4} {pred[o]:.4f} ± {unc[o]:.4f}  [{src[o]}]")
    print("--init-lambdas " + ",".join(f"{pred[o]:.4f}" for o in orbs))


def cmd_verify(a):
    """One structx run per tuned ion at its LOIO-predicted λ; compare to the tuned RMS."""
    from . import match as M, nist
    from .deck import write_ssin
    from .runner import run_batch
    rows = load_ions(a.dataset, a.mode)
    res = loio(rows, a.model)
    decks, names, meta = [], [], []
    for r in res:
        iondir = os.path.join(a.workdir, r["tag"])
        try:
            with open(os.path.join(iondir, "deck.json")) as f:
                d = json.load(f)
        except FileNotFoundError:
            continue
        ion = build_ion(d["ion"])
        cfgs = [{tuple(o): n for o, n in c} for c in d["configs"]]
        decks.append(write_ssin(cfgs, ion.Z, r["lam_pred"], mode="orbital", mpnch=0))
        names.append(r["tag"]); meta.append((r, ion, cfgs, d))
    print(f"verify: {len(decks)} runs at LOIO-predicted λ ({a.model})")
    results = run_batch(decks, a.scratch or os.path.join(a.workdir, "_verify"), os.path.abspath(a.binary),
                        n_workers=a.workers, names=names, stop_after_levels=True, timeout=a.timeout)
    lines = [f"{'ion':<10}{'seq':>5}{'Z':>4}{'RMS tuned':>11}{'RMS pred λ':>12}{'ratio':>8}{'levels':>8}{'RMS pred Ry':>12}",
             f"# RMS tuned / RMS pred λ: {a.objective} objective (rel = RMS relative error, abs = Ry); last column: plain Ry RMS"]
    out_rows = []
    for (r, ion, cfgs, d), rr in zip(meta, results):
        if not rr.ok:
            lines.append(f"{r['ion']:<10}{'FAIL':>40}"); continue
        nl = nist.load_levels(a.data, ion.spectroscopic, fetch_if_missing=False)
        targets = M.nist_targets(nl, ion, cfgs)
        m = M.match_levels(rr.levels, nl, ion, cfgs, rr.configs, targets=targets, match_key=a.match_key)
        # same metric as the tuning objective (rel by default), plus the plain Ry RMS for reference
        weights = M.target_weights({k: nl[k].E for k in targets}, a.objective)
        rp = M.rms(m, weights)
        rp_abs = M.rms(m)
        ratio = rp / r["rms_tuned"] if r["rms_tuned"] > 0 else float("nan")
        lines.append(f"{r['ion']:<10}{r['sequence']:>5}{r['Z']:>4}{r['rms_tuned']:>11.5f}{rp:>12.5f}"
                     f"{ratio:>8.2f}{len(m):>8}{rp_abs:>12.5f}")
        out_rows.append({"ion": r["ion"], "tag": r["tag"], "Z": r["Z"], "N": r["N"], "sequence": r["sequence"],
                         "rms_tuned": r["rms_tuned"], "rms_pred": rp, "ratio": ratio, "rms_pred_abs": rp_abs,
                         "n_matched": len(m),
                         "lam_pred": " ".join(f"{o}={v:.4f}" for o, v in zip(r["orbitals"], r["lam_pred"]))})
    rep = "\n".join(lines)
    print(rep)
    with open(os.path.join(a.dataset, f"verify_{a.model}.csv"), "w", newline="") as f:
        if out_rows:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    with open(os.path.join(a.dataset, f"verify_{a.model}.txt"), "w") as f:
        f.write(rep + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--mode", default="orbital")
    p.add_argument("--model", default="gp", choices=["gp", "seq"])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    sp = sub.add_parser("predict"); sp.add_argument("ion")
    sp.add_argument("--data", default="data"); sp.add_argument("--n-max", type=int, default=4)
    sp.add_argument("--l-max", type=int, default=3)
    sv = sub.add_parser("verify")
    sv.add_argument("--workdir", default="work"); sv.add_argument("--data", default="data")
    sv.add_argument("--binary", default="./structx"); sv.add_argument("--workers", type=int, default=8)
    sv.add_argument("--scratch", default=None); sv.add_argument("--timeout", type=int, default=1800)
    sv.add_argument("--match-key", default="j")
    sv.add_argument("--objective", default="rel", choices=["abs", "rel"],
                    help="metric for 'RMS pred λ' -- must match the objective the ions were tuned with")
    a = p.parse_args(argv)
    {"fit": cmd_fit, "predict": cmd_predict, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
