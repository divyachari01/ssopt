"""Multi-output surrogate of level energies E_k(λ) and the optimisation loop.

Why not BO on the scalar RMS: every structx run yields tens to hundreds of
matched level energies, each a smooth (nearly linear) function of λ over the
physical box. Modelling each E_k(λ) and computing RMS from the model uses all
of that information; modelling the scalar RMS throws 99% of it away.

Model per matched NIST target k:   E_k(λ) = a_k + b_k·λ + GP_k(λ)
(linear trend + anisotropic RBF GP on the residual, independent per output).
Targets that were not matched in a given run are simply missing for that
output (masked), so a level that appears only for some λ still contributes.

Loop:
  design  -> Sobol points in the box (+ the centre λ=1)
  fit     -> per-output GP
  propose -> argmin of surrogate RMS (exploit) + argmin of an optimistic RMS
             (explore, uses per-output σ) + a few fresh Sobol points
  verify  -> run structx on the proposals, add to the data, refit
  stop    -> when the verified RMS stops improving by more than `tol`
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from .gp import MultiOutputGP

from . import match as M
from .deck import algebra_key, lambda_names, n_lambda, set_restart, write_ssin
from .runner import run_batch, run_structx


# ----------------------------------------------------------------------------
# λ box
# ----------------------------------------------------------------------------
def default_bounds(configs: list[dict], mode: str, ion, *,
                   core_bounds=(0.9, 1.2), valence_bounds=(0.7, 2.0),
                   freeze_core: bool = False) -> list[tuple[float, float]]:
    """Physics rules: inner core orbitals tightly bounded, valence free.
    In 'l' mode every l-shell touches the core, so the core box applies to
    l-shells that contain a core orbital."""
    from .deck import orbitals_used
    orbs = orbitals_used(configs)
    core_orbs = set(ion.core)
    if mode == "orbital":
        out = []
        for o in orbs:
            if o in core_orbs:
                out.append((1.0, 1.0) if freeze_core else core_bounds)
            else:
                out.append(valence_bounds)
        return out
    n_l = n_lambda(configs, "l")
    out = []
    for l in range(n_l):
        # an l-shell λ also scales the core orbitals of that l, so the core
        # rule applies to it (the 'orbital' stage frees the valence ones)
        has_core = any(o[1] == l for o in core_orbs)
        out.append(core_bounds if has_core else valence_bounds)
    return out


# ----------------------------------------------------------------------------
# data store
# ----------------------------------------------------------------------------
@dataclass
class Sample:
    lam: list[float]
    E: dict            # nist_idx -> computed energy (Ry)
    rms: float
    n_matched: int
    elapsed: float
    ok: bool
    tag: str = ""


@dataclass
class History:
    lambda_names: list[str]
    bounds: list[tuple[float, float]]
    mode: str
    samples: list[Sample] = field(default_factory=list)
    objective: str = "abs"

    def X(self) -> np.ndarray:
        return np.array([s.lam for s in self.samples if s.ok])

    def best(self) -> Sample | None:
        ok = [s for s in self.samples if s.ok and np.isfinite(s.rms)]
        return min(ok, key=lambda s: s.rms) if ok else None

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"lambda_names": self.lambda_names, "bounds": self.bounds, "mode": self.mode,
                       "objective": self.objective,
                       "samples": [{"lam": s.lam, "E": {str(k): v for k, v in s.E.items()},
                                    "rms": s.rms, "n_matched": s.n_matched, "elapsed": s.elapsed,
                                    "ok": s.ok, "tag": s.tag} for s in self.samples]}, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "History":
        with open(path) as f:
            d = json.load(f)
        h = cls(d["lambda_names"], [tuple(b) for b in d["bounds"]], d["mode"], objective=d.get("objective", "abs"))
        for s in d["samples"]:
            h.samples.append(Sample(s["lam"], {int(k): v for k, v in s["E"].items()}, s["rms"],
                                    s["n_matched"], s["elapsed"], s["ok"], s.get("tag", "")))
        return h


# ----------------------------------------------------------------------------
# surrogate
# ----------------------------------------------------------------------------
class EnergySurrogate:
    """One multi-output GP (shared kernel) over per-target standardised residuals
    after a per-target linear trend. Missing entries (targets unmatched in a
    given run) are imputed with the linear trend so every output shares X."""

    def __init__(self, bounds, targets: list[int], E_nist: dict[int, float], min_points: int = 3,
                 weights: dict[int, float] | None = None):
        self.bounds = np.array(bounds, dtype=float)
        self.targets = list(targets)
        self.E_nist = np.array([E_nist[k] for k in self.targets])
        self.w = np.array([(weights or {}).get(k, 1.0) for k in self.targets])
        self.min_points = min_points
        self.active = np.zeros(len(self.targets), bool)

    def _scale(self, X):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(X, float) - lo) / np.where(hi > lo, hi - lo, 1.0)

    def fit(self, history: History):
        ok = [s for s in history.samples if s.ok]
        X = np.array([s.lam for s in ok])
        Xs = self._scale(X)
        n, d = Xs.shape
        m = len(self.targets)
        Y = np.full((n, m), np.nan)
        for i, s in enumerate(ok):
            for j, k in enumerate(self.targets):
                if k in s.E:
                    Y[i, j] = s.E[k]
        A = np.hstack([np.ones((n, 1)), Xs])
        self.coef = np.zeros((d + 1, m))
        self.sd = np.ones(m)
        R = np.zeros((n, m))
        for j in range(m):
            rows = np.where(np.isfinite(Y[:, j]))[0]
            if len(rows) < self.min_points:
                continue
            c, *_ = np.linalg.lstsq(A[rows], Y[rows, j], rcond=None)
            self.coef[:, j] = c
            r = Y[:, j] - A @ c
            r[~np.isfinite(r)] = 0.0          # impute with the trend
            self.sd[j] = max(np.std(r[rows]), 1e-5)
            R[:, j] = r / self.sd[j]
            self.active[j] = True
        self.coverage = np.isfinite(Y).mean(axis=0)
        self.gp = MultiOutputGP(length_scale=0.7, noise=1e-3, n_restarts=1).fit(Xs, R[:, self.active])
        self.n_train = n
        return self

    def predict(self, X, return_std: bool = False):
        X = np.atleast_2d(np.asarray(X, float))
        Xs = self._scale(X)
        A = np.hstack([np.ones((len(Xs), 1)), Xs])
        mu = A @ self.coef
        sd = np.zeros_like(mu)
        if return_std:
            r, s = self.gp.predict(Xs, return_std=True)
            sd[:, self.active] = s * self.sd[self.active]
        else:
            r = self.gp.predict(Xs)
        r = np.atleast_2d(r)
        mu[:, self.active] += r * self.sd[self.active]
        mu[:, ~self.active] = np.nan
        return (mu, sd) if return_std else mu

    def rms(self, lam, beta: float = 0.0) -> float:
        """Surrogate RMS. beta>0 gives an optimistic (lower-confidence) RMS."""
        if beta > 0:
            mu, sd = self.predict([lam], return_std=True)
            err = np.abs(mu[0] - self.E_nist)
            err = np.maximum(err - beta * sd[0], 0.0) * self.w
        else:
            mu = self.predict([lam])
            err = (mu[0] - self.E_nist) * self.w
        err = err[np.isfinite(err)]
        return float(np.sqrt(np.mean(err * err))) if len(err) else np.inf

    def minimize(self, x0s, beta: float = 0.0, opt_bounds=None):
        """Multi-start L-BFGS-B on the surrogate RMS inside opt_bounds
        (defaults to the model box; the model box itself is never changed —
        it also defines the input scaling)."""
        ob = [tuple(b) for b in (opt_bounds if opt_bounds is not None else self.bounds)]
        best = None
        for x0 in x0s:
            r = minimize(lambda x: self.rms(x, beta), np.asarray(x0, float), method="L-BFGS-B",
                         bounds=ob, options={"maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
        return best.x, best.fun

    def sensitivity(self, lam, h: float = 0.02) -> np.ndarray:
        """d(RMS)/dλ_i by central differences on the surrogate."""
        lam = np.asarray(lam, float)
        g = np.zeros(len(lam))
        for i in range(len(lam)):
            e = np.zeros(len(lam)); e[i] = h
            g[i] = (self.rms(lam + e) - self.rms(lam - e)) / (2 * h)
        return g


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
class Evaluator:
    """Turns λ vectors into Samples by running structx."""

    def __init__(self, ion, configs, nist_levels, targets, cf_lookup, *, mode, binary,
                 workroot, n_workers, timeout=1800, e_max=None, log=print, match_key="j", objective="abs",
                 closed_core=False, restart_dir=None):
        """restart_dir: where to keep the saved angular algebra (fort.10). The
        first structx run does the algebra and saves it; every later run reads
        it back and only does the radial part, which is many times faster for
        ions with many open-shell electrons. None = no restart (every run
        repeats the algebra)."""
        self.ion, self.configs, self.nist = ion, configs, nist_levels
        self.restart_dir = restart_dir
        self.restart_file = None
        self.closed_core = closed_core
        self.match_key = match_key
        self.objective = objective
        self.weights = M.target_weights({k: nist_levels[k].E for k in targets}, objective)
        self.targets, self.mode, self.binary = targets, mode, binary
        self.workroot, self.n_workers, self.timeout = workroot, n_workers, timeout
        self.e_max = e_max
        self.log = log
        self.counter = 0
        self._cf_lookup = cf_lookup

    def _run(self, decks, names):
        kw = dict(n_workers=self.n_workers, stop_after_levels=True, timeout=self.timeout)
        first = None
        if self.restart_dir and self.restart_file is None:
            first = self._save_algebra(decks[0])
        if not self.restart_file:
            return run_batch(decks, self.workroot, self.binary, names=names, **kw)
        todo = decks[1:] if first is not None else decks
        extra = [decks[0]] if first is not None else []      # one check that restart reproduces
        res = run_batch([set_restart(d, 4) for d in todo + extra], self.workroot, self.binary,
                        names=names[len(decks) - len(todo):] + [names[0] + "r"] * len(extra),
                        restart_file=self.restart_file, **kw)
        if first is not None:
            chk = res.pop()
            same = not first.ok or chk.ok and len(chk.levels) == len(first.levels) and all(
                abs(a.E - b.E) < 1e-6 for a, b in zip(chk.levels, first.levels))
            if not same:
                self.log("  restart did not reproduce the energies: running without it")
                self.restart_dir, self.restart_file = None, None
                return [first] + run_batch(decks[1:], self.workroot, self.binary, names=names[1:], **kw)
            res = [first] + res
        return res

    def _save_algebra(self, deck):
        """Run one deck with the algebra saved to fort.10 (or reuse a saved one with
        the same configuration cards). Returns that run's result, or None if reused."""
        os.makedirs(self.restart_dir, exist_ok=True)
        path = os.path.join(self.restart_dir, algebra_key(deck) + ".fort10")
        if os.path.exists(path):
            self.restart_file = path
            return None
        wd = os.path.join(self.workroot, "algebra_" + algebra_key(deck))
        t0 = time.time()
        r = run_structx(set_restart(deck, -1), wd, self.binary, stop_after_levels=True, timeout=self.timeout)
        f10 = os.path.join(wd, "fort.10")
        saved = os.path.exists(f10) and b"MSTART= 4" in open(os.path.join(wd, "ssout"), "rb").read()
        if saved:
            os.replace(f10, path)
            shutil.copyfile(os.path.join(wd, "ssout"), path + ".ssout")
            self.restart_file = path
            self.log(f"  angular algebra done once in {time.time() - t0:.0f}s and saved; "
                     f"later runs read it back")
        else:
            self.log("  could not save the angular algebra: running without restart")
            self.restart_dir = None
        return r

    def evaluate(self, lams: list[list[float]], tag: str = "") -> list[Sample]:
        decks = [write_ssin(self.configs, self.ion.Z, list(l), mode=self.mode, mpnch=0,
                            closed_core=self.closed_core) for l in lams]
        names = [f"e{self.counter + i:04d}" for i in range(len(lams))]
        self.counter += len(lams)
        t0 = time.time()
        results = self._run(decks, names)
        out = []
        for lam, r in zip(lams, results):
            if not r.ok:
                out.append(Sample(list(lam), {}, np.inf, 0, r.elapsed, False, tag))
                continue
            m = M.match_levels(r.levels, self.nist, self.ion, self.configs, r.configs,
                               targets=self.targets, match_key=self.match_key)
            E = {mm.nist_idx: mm.E_comp for mm in m}
            out.append(Sample(list(lam), E, M.rms(m, self.weights), len(m), r.elapsed, True, tag))
        self.log(f"  batch of {len(lams)} done in {time.time() - t0:.0f}s: "
                 f"RMS " + " ".join(f"{s.rms:.4f}" if s.ok else "FAIL" for s in out))
        return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def sobol(bounds, n, seed=0) -> np.ndarray:
    d = len(bounds)
    eng = qmc.Sobol(d, scramble=True, seed=seed)
    n2 = 1 << max(0, int(np.ceil(np.log2(max(n, 1)))))
    u = eng.random(n2)[:n]
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    return lo + u * (hi - lo)


def _good_samples(hist: History, factor: float = 10.0, min_frac: float = 0.75, floor: float = 0.3):
    """Samples worth modelling: far from the optimum level identities scramble
    and per-level energies stop being smooth in λ."""
    ok = [s for s in hist.samples if s.ok and np.isfinite(s.rms)]
    if not ok:
        return []
    best = min(s.rms for s in ok)
    nmax = max(s.n_matched for s in ok)
    return [s for s in ok if s.rms <= max(factor * best, floor) and s.n_matched >= min_frac * nmax]


def optimize(evaluator: Evaluator, bounds, lambda_names_, *, n_init: int = 32,
             n_rounds: int = 4, batch: int = 8, tol: float = 2e-4, seed: int = 0,
             init_radius: float = 0.2, trust_radius: float = 0.15,
             n_polish: int = 2, polish_step: float = 0.1,
             history: History | None = None, save_path: str | None = None,
             log=print) -> History:
    """Trust-region surrogate optimisation.

    * initial design: Sobol in the box  centre ± init_radius  (∩ bounds)
    * each round: fit the surrogate on the well-behaved samples, minimise the
      surrogate RMS inside  best ± ρ  (∩ bounds), verify, refit.
      ρ grows ×1.5 on success, shrinks ×0.5 on failure.
    """
    E_nist = {k: evaluator.nist[k].E for k in evaluator.targets}
    hist = history or History(list(lambda_names_), [tuple(b) for b in bounds], evaluator.mode, objective=evaluator.objective)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    d = len(bounds)

    def save():
        if save_path:
            hist.save(save_path)

    centre = np.clip(np.ones(d), lo, hi)
    if hist.best() is not None:
        centre = np.array(hist.best().lam)
    if len([s for s in hist.samples if s.ok]) < max(4, d + 2):
        box = [(max(l, c - init_radius), min(h, c + init_radius)) for l, h, c in zip(lo, hi, centre)]
        X0 = sobol(box, n_init - 1, seed)
        X0 = np.vstack([centre, X0])
        log(f"design: {len(X0)} points in {d}-D box centre±{init_radius}")
        hist.samples += evaluator.evaluate(X0.tolist(), "design")
        save()
    best = hist.best()
    log(f"best so far RMS={best.rms:.5f} λ={np.round(best.lam, 4).tolist()} ({best.n_matched} matched)")

    rho = trust_radius
    rng = np.random.default_rng(seed)
    for rnd in range(n_rounds):
        good = _good_samples(hist)
        fit_hist = History(hist.lambda_names, hist.bounds, hist.mode, good)
        sur = EnergySurrogate(bounds, evaluator.targets, E_nist, weights=evaluator.weights).fit(fit_hist)
        bx = np.array(best.lam)
        tr = [(max(l, b - rho), min(h, b + rho)) for l, h, b in zip(lo, hi, bx)]
        top = sorted(good, key=lambda s: s.rms)[:4]
        x0s = [np.clip(s.lam, [t[0] for t in tr], [t[1] for t in tr]) for s in top] + sobol(tr, 6, seed + rnd + 1).tolist()
        proposals, tags = [], []
        x_exp, f_exp = sur.minimize(x0s, beta=0.0, opt_bounds=tr)
        proposals.append(x_exp); tags.append("exploit")
        for beta in (1.0, 2.0):
            x_b, _ = sur.minimize(x0s, beta=beta, opt_bounds=tr)
            proposals.append(x_b); tags.append(f"explore{beta:g}")
        while len(proposals) < batch:
            p = np.clip(x_exp + rng.normal(0, rho / 3, d), lo, hi)
            proposals.append(p); tags.append("local")
        X = hist.X()
        span = hi - lo
        keep, ktags = [], []
        for p, t in zip(proposals, tags):
            if len(X) and np.min(np.linalg.norm((X - p) / span, axis=1)) < 2e-3:
                continue
            if any(np.linalg.norm((np.array(q) - p) / span) < 2e-3 for q in keep):
                continue
            keep.append(list(map(float, p))); ktags.append(t)
        log(f"round {rnd + 1}: ρ={rho:.3f}, surrogate ({len(good)} pts) predicts RMS={f_exp:.5f} "
            f"at λ={np.round(x_exp, 4).tolist()}; verifying {len(keep)}")
        new = evaluator.evaluate(keep, "round%d" % (rnd + 1))
        for s_, t in zip(new, ktags):
            s_.tag = t
        hist.samples += new
        save()
        nb = hist.best()
        improved = best.rms - nb.rms
        log(f"  verified best RMS={nb.rms:.5f} λ={np.round(nb.lam, 4).tolist()} "
            f"(Δ={improved:+.5f}, surrogate said {f_exp:.5f})")
        if improved > tol:
            rho = min(rho * 1.5, 0.5)
        else:
            rho = max(rho * 0.5, 0.02)
        at_edge = any(abs(nb.lam[i] - lo[i]) < 1e-6 or abs(nb.lam[i] - hi[i]) < 1e-6 for i in range(d))
        if at_edge:
            log("  note: best λ is on the physics box boundary")
        best = nb
        if improved < tol and rho <= 0.02:
            log("  converged (trust region collapsed)")
            break
    # ---- polish: coordinate scans around the best point -----------------
    # The surrogate can be over-optimistic where level identities swap with λ
    # (4f levels in Si I). A one-batch scan of every coordinate is cheap on a
    # node and does not depend on the model; the surrogate is then refit on
    # the denser data and its argmin verified once more.
    for it in range(n_polish):
        bx = np.array(best.lam)
        pts = []
        for i in range(d):
            for frac in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
                p = bx.copy()
                p[i] = np.clip(bx[i] + frac * polish_step, lo[i], hi[i])
                if abs(p[i] - bx[i]) > 1e-6:
                    pts.append(p)
        X = hist.X()
        span = hi - lo
        pts = [list(map(float, p)) for p in pts
               if not (len(X) and np.min(np.linalg.norm((X - p) / span, axis=1)) < 2e-3)]
        log(f"polish {it + 1}: scanning {len(pts)} points, ±{polish_step} per coordinate around the best")
        hist.samples += evaluator.evaluate(pts, f"polish{it + 1}")
        save()
        good = _good_samples(hist)
        sur = EnergySurrogate(bounds, evaluator.targets, E_nist, weights=evaluator.weights).fit(History(hist.lambda_names, hist.bounds, hist.mode, good))
        nb = hist.best()
        tr = [(max(l, b - polish_step), min(h, b + polish_step)) for l, h, b in zip(lo, hi, nb.lam)]
        top = sorted(good, key=lambda s: s.rms)[:6]
        x0s = [np.clip(s.lam, [t[0] for t in tr], [t[1] for t in tr]) for s in top]
        x_exp, f_exp = sur.minimize(x0s, beta=0.0, opt_bounds=tr)
        new = evaluator.evaluate([list(map(float, x_exp))], f"polish{it + 1}-argmin")
        hist.samples += new
        save()
        nb2 = hist.best()
        log(f"  polish best RMS={nb2.rms:.5f} λ={np.round(nb2.lam, 4).tolist()} "
            f"(surrogate argmin said {f_exp:.5f}, got {new[0].rms:.5f})")
        improved = best.rms - nb2.rms
        best = nb2
        if improved < tol:
            break
        polish_step *= 0.5

    good = _good_samples(hist)
    sur = EnergySurrogate(bounds, evaluator.targets, E_nist, weights=evaluator.weights).fit(History(hist.lambda_names, hist.bounds, hist.mode, good))
    g = sur.sensitivity(best.lam)
    log("sensitivity dRMS/dλ: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(lambda_names_, g)))
    return hist
