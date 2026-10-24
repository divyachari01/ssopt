"""Command line.

  ssopt fetch  "Si I"                 # NIST levels (+ lines) and NORAD files -> data/   (login node)
  ssopt deck   "Si I"                 # show the configuration list the tool would use
  ssopt tune   "Si I" [--mode l|orbital] [--n-init 32] [--rounds 4] [--workers N]
  ssopt full   "Si I" --lambdas 0.97,1.02,...   # one full run with radiative data, A-value check
  ssopt prune  "Si I" --lambdas ...  # leave-group-out test of correlation configs
  ssopt uncert "Si I" --mode orbital  # uncertainty and grade of every A-value and lifetime, held-out checks

Everything lands under <workdir>/<ion tag>/ :
  deck.json, history.json, ssin.tuned, report.txt, runs/ (scratch, cleaned)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from . import match as M
from . import nist, norad
from .configs import build_deck, prune_groups
from .deck import e1_only, lambda_names, limit_transitions, n_lambda, orbitals_used, set_restart, write_ssin
from .ions import build_ion, config_str
from .runner import run_structx, run_batch
from .surrogate import Evaluator, History, default_bounds, optimize


def _log_to(path):
    f = open(path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def _md5(path):
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except OSError:
        return "missing"


def _provenance(args, ion):
    """One line that says exactly what produced a result: code version, structx, NIST file, command."""
    import platform
    import shlex
    import subprocess
    from . import __version__
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True,
                                        timeout=10).stdout.strip()
        commit = git("rev-parse", "--short", "HEAD")
        if commit and git("status", "--porcelain", "--untracked-files=no"):
            commit += "+local changes"
    except (OSError, subprocess.SubprocessError):
        commit = ""
    npath = nist.levels_cache_path(args.data, ion.spectroscopic)
    fetched = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(npath))) if os.path.exists(npath) else "?"
    return (f"ssopt {__version__}{f' (git {commit})' if commit else ''}, structx md5 {_md5(args.binary)}, "
            f"NIST levels md5 {_md5(npath)} (fetched {fetched}), python {platform.python_version()}, "
            f"numpy {np.__version__}; command: {shlex.join(['ssopt'] + sys.argv[1:])}")


def _setup(args):
    ion = build_ion(args.ion)
    iondir = os.path.join(args.workdir, ion.tag)
    os.makedirs(iondir, exist_ok=True)
    log = _log_to(os.path.join(iondir, "log.txt"))
    nl = nist.load_levels(args.data, ion.spectroscopic, fetch_if_missing=not args.offline)
    args.provenance = _provenance(args, ion)
    log("run: " + args.provenance)
    plan = build_deck(ion, nl, n_max=args.n_max, l_max=args.l_max, max_exc=args.max_exc,
                      doubles_mode=args.doubles, e_max=args.e_max)
    if args.drop:
        drop = set(int(i) for i in args.drop.split(","))
        plan.configs = [c for i, c in enumerate(plan.configs) if i not in drop]
        plan.tags = [t for i, t in enumerate(plan.tags) if i not in drop]
    with open(os.path.join(iondir, "deck.json"), "w") as f:
        json.dump({"ion": ion.spectroscopic, "n_max": args.n_max, "l_max": args.l_max,
                   "configs": [[[list(o), n] for o, n in c.items()] for c in plan.configs],
                   "tags": plan.tags}, f)
    return ion, iondir, log, nl, plan


def _bounds(args, plan, ion, mode):
    core_b = tuple(float(x) for x in args.core_bounds.split(","))
    val_b = tuple(float(x) for x in args.valence_bounds.split(","))
    return default_bounds(plan.configs, mode, ion, core_bounds=core_b, valence_bounds=val_b,
                          freeze_core=args.freeze_core)


def cmd_fetch(args):
    ion = build_ion(args.ion)
    os.makedirs(args.data, exist_ok=True)
    p = nist.levels_cache_path(args.data, ion.spectroscopic)
    if not os.path.exists(p) or args.force:
        print("fetching NIST levels ->", p)
        nist.fetch(nist.levels_url(ion.spectroscopic), cache=p)
    print(f"  {len(nist.load_levels(args.data, ion.spectroscopic))} NIST levels")
    p = nist.lines_cache_path(args.data, ion.spectroscopic)
    if not os.path.exists(p) or args.force:
        print("fetching NIST lines ->", p)
        try:
            nist.fetch(nist.lines_url(ion.spectroscopic), cache=p)
            print(f"  {len(nist.load_lines(args.data, ion.spectroscopic))} NIST lines with A-values")
        except Exception as e:  # noqa: BLE001
            print("  lines fetch failed:", e)
    got = norad.fetch_all(args.data, ion, force=args.force)
    print("NORAD files:", got or "none available for this ion")


def cmd_deck(args):
    ion, iondir, log, nl, plan = _setup(args)
    print(f"{ion.spectroscopic}: Z={ion.Z} N={ion.n_electrons} core={config_str(ion.core)} "
          f"valence={config_str(ion.valence)}")
    print(f"{len(plan)} configurations (n<={args.n_max}, l<={args.l_max}):")
    print(plan.describe(ion))
    tg = M.nist_targets(nl, ion, plan.configs, e_max=args.e_max)
    print(f"NIST targets reachable by this deck: {len(tg)} of {len(nl)}")
    for mode in ("l", "orbital"):
        print(f"mode={mode}: {n_lambda(plan.configs, mode)} λ  {lambda_names(plan.configs, mode)}  "
              f"bounds={_bounds(args, plan, ion, mode)}")
    print("prune groups:", {k: len(v) for k, v in prune_groups(plan, ion).items()})


def cmd_tune(args):
    ion, iondir, log, nl, plan = _setup(args)
    mode = args.mode
    names = lambda_names(plan.configs, mode)
    bounds = _bounds(args, plan, ion, mode)
    targets = M.nist_targets(nl, ion, plan.configs, e_max=args.e_max)
    log(f"{ion.spectroscopic}: {len(plan)} configs, {len(targets)} NIST targets, mode={mode}, "
        f"λ={names}, bounds={bounds}")
    workroot = os.path.join(args.scratch or iondir, "runs")
    ev = Evaluator(ion, plan.configs, nl, targets, None, mode=mode, binary=os.path.abspath(args.binary),
                   workroot=workroot, n_workers=args.workers, timeout=args.timeout, log=log,
                   match_key=args.match_key, objective=args.objective, closed_core=args.closed_core,
                   restart_dir=None if args.no_restart else os.path.join(args.scratch or workroot, "algebra"))
    hpath = os.path.join(iondir, f"history_{mode}.json")
    hist = History.load(hpath) if (args.resume and os.path.exists(hpath)) else None
    if hist is not None and hist.objective != args.objective:
        log(f"history was tuned with objective={hist.objective}, now {args.objective}: starting fresh")
        hist = None
    if args.init_lambdas:
        x0 = [float(x) for x in args.init_lambdas.split(",")]
        hist = hist or History(names, bounds, mode, objective=args.objective)
        hist.samples += ev.evaluate([x0], "init")
    hist = optimize(ev, bounds, names, n_init=args.n_init, n_rounds=args.rounds, batch=args.batch,
                    tol=args.tol, seed=args.seed, init_radius=args.init_radius,
                    trust_radius=args.trust_radius, n_polish=args.polish, polish_step=args.polish_step,
                    history=hist, save_path=hpath, log=log)
    best = hist.best()
    deck = limit_transitions(write_ssin(plan.configs, ion.Z, best.lam, mode=mode, mpnch=-1,
                                        closed_core=args.closed_core), plan.tags.count("spec"))
    with open(os.path.join(iondir, f"ssin.tuned_{mode}"), "w") as f:     # runs fast with ./structx < ssin
        f.write(deck)
    unit = "Ry" if args.objective == "abs" else "(relative)"
    log(f"best λ ({mode}) = {np.round(best.lam, 4).tolist()}  RMS={best.rms:.5f} {unit} over "
        f"{best.n_matched} levels  -> {iondir}/ssin.tuned_{mode}")
    # detailed comparison at the best point
    rf = ev.restart_file
    r = run_structx(set_restart(deck, 4) if rf else deck, os.path.join(workroot, "best"),
                    os.path.abspath(args.binary), stop_after_levels=True, timeout=args.timeout, restart_file=rf)
    m = M.match_levels(r.levels, nl, ion, plan.configs, r.configs, targets=targets, match_key=args.match_key)
    rep = M.summary(m, nl, r.levels, r.configs)
    from .uncert import holdout, holdout_line
    ho = holdout_line(holdout(hist, nl, args.objective), args.objective)
    log(ho)
    with open(os.path.join(iondir, f"levels_{mode}.txt"), "w") as f:
        f.write(f"# {args.provenance}\n{rep}\n{ho}\n")
    print(rep)
    if not args.keep_runs:
        shutil.rmtree(workroot, ignore_errors=True)


def _best_lambdas(iondir, mode):
    hp = os.path.join(iondir, f"history_{mode}.json")
    if not os.path.exists(hp):
        raise SystemExit(f"no {hp} — run `tune --mode {mode}` first")
    b = History.load(hp).best()
    return [float(x) for x in b.lam]


def _lambdas_arg(args, iondir):
    if args.lambdas.strip().lower() == "best":
        return _best_lambdas(iondir, args.mode)
    return [float(x) for x in args.lambdas.split(",")]


def cmd_best(args):
    """Print the best λ (comma separated) for an ion/mode — for scripting."""
    ion = build_ion(args.ion)
    lam = _best_lambdas(os.path.join(args.workdir, ion.tag), args.mode)
    print(",".join(f"{x:.4f}" for x in lam))


def cmd_full(args):
    """One full run (radiative data on) at given λ, then the A-value check."""
    from .avalues import compare
    ion, iondir, log, nl, plan = _setup(args)
    mode = args.mode
    lam = _lambdas_arg(args, iondir)
    deck = write_ssin(plan.configs, ion.Z, lam, mode=mode, mpnch=-1, closed_core=args.closed_core)
    if not args.all_transitions:
        deck = limit_transitions(deck, plan.tags.count("spec"))
    if args.e1_only:
        deck = e1_only(deck)
    with open(os.path.join(iondir, "ssin.full"), "w") as f:
        f.write(deck)
    log(f"full run at λ={lam} ({mode}){', E1 only' if args.e1_only else ''}{', all levels' if args.all_transitions else ', transitions between NIST-configuration levels'} ...")
    r = run_structx(deck, os.path.join(iondir, "full"), os.path.abspath(args.binary),
                    stop_after_levels=False, timeout=args.timeout * 10, keep_fort7=True)
    log(f"  done in {r.elapsed:.0f}s, {len(r.levels)} levels")
    shutil.copy(r.ssout, os.path.join(iondir, "ssout.full"))
    from .export import relabel_from_nist
    relabel_from_nist(r.levels, ion, iondir, r.configs, args.data, args.match_key)   # same labels as the relabelled ssout.full
    targets = M.nist_targets(nl, ion, plan.configs, e_max=args.e_max)
    m = M.match_levels(r.levels, nl, ion, plan.configs, r.configs, targets=targets, match_key=args.match_key)
    flagged = M.ambiguous(m, r.levels, nl, ion, plan.configs, r.configs, targets, args.match_key)
    rep = M.summary(m, nl, r.levels, r.configs, flagged)
    head = f"# {args.provenance}\n# λ ({mode}) = {','.join(f'{x:.4f}' for x in lam)}\n"
    hp = os.path.join(iondir, f"history_{mode}.json")
    if os.path.exists(hp):
        from .uncert import holdout, holdout_line
        rep += "\n" + holdout_line(holdout(History.load(hp), nl, args.objective), args.objective)
    with open(os.path.join(iondir, f"levels_{mode}.txt"), "w") as f:   # same pairing and labels as avalues.txt
        f.write(head + rep + "\n")
    arep = compare(iondir, ion, r, m, nl, args.data, flagged=flagged, header=head)
    with open(os.path.join(iondir, "report.txt"), "w") as f:
        f.write(head + rep + "\n\n" + arep + "\n")
    print(rep)
    print()
    print(arep)


def cmd_uncert(args):
    """Uncertainty and grade of every E1 A-value and lifetime (see ssopt/uncert.py)."""
    from . import uncert
    ion, iondir, log, nl, plan = _setup(args)
    hp = os.path.join(iondir, f"history_{args.mode}.json")
    if not os.path.exists(hp):
        raise SystemExit(f"no {hp}: run `tune --mode {args.mode}` first")
    other = os.path.join(iondir, f"history_{'l' if args.mode == 'orbital' else 'orbital'}.json")
    extra = [History.load(other)] if os.path.exists(other) else []
    hist = History.load(hp)
    head = (f"# {args.provenance}\n# λ ({args.mode}) = "
            f"{','.join(f'{x:.4f}' for x in hist.best().lam)}\n")
    rep = uncert.run(args, ion, iondir, log, nl, plan, hist, extra)
    with open(os.path.join(iondir, "uncert.txt"), "w") as f:
        f.write(head + rep + "\n")
    print("\n".join(rep.splitlines()[:8]))
    log(f"-> {iondir}/uncert.txt")


def cmd_prune(args):
    """Leave-group-out: which correlation groups actually move the RMS?"""
    ion, iondir, log, nl, plan = _setup(args)
    mode = args.mode
    lam = _lambdas_arg(args, iondir)
    groups = prune_groups(plan, ion)
    targets = M.nist_targets(nl, ion, plan.configs, e_max=args.e_max)
    decks, names = [], []
    variants = [("baseline", [])] + [(g, idx) for g, idx in groups.items()]
    orbs = orbitals_used(plan.configs)
    for g, idx in variants:
        cfgs = [c for i, c in enumerate(plan.configs) if i not in set(idx)]
        # removing a group can remove an orbital (or the highest l): keep the λ of what is left
        if mode == "orbital":
            lam_g = [lam[orbs.index(o)] for o in orbitals_used(cfgs)]
        else:
            lam_g = lam[:n_lambda(cfgs, "l")]
        decks.append(write_ssin(cfgs, ion.Z, lam_g, mode=mode, mpnch=0, closed_core=args.closed_core))
        names.append(g.replace("^", "").replace("+", "_"))
    log(f"prune test: {len(decks)} runs ({', '.join(g for g, _ in variants)})")
    res = run_batch(decks, os.path.join(iondir, "prune"), os.path.abspath(args.binary),
                    n_workers=args.workers, names=names, stop_after_levels=True, timeout=args.timeout)
    lines = [f"{'removed group':<14}{'configs':>8}{'matched':>8}{'RMS':>10}{'ΔRMS':>10}"]
    base = None
    for (g, idx), r in zip(variants, res):
        cfgs = [c for i, c in enumerate(plan.configs) if i not in set(idx)]
        if not r.ok:
            lines.append(f"{g:<14}{len(cfgs):>8}{'FAIL':>8}")
            continue
        m = M.match_levels(r.levels, nl, ion, cfgs, r.configs, targets=targets, match_key=args.match_key,
                           term_first=False)
        rr = M.rms(m)
        if base is None:
            base = rr
        lines.append(f"{g:<14}{len(cfgs):>8}{len(m):>8}{rr:>10.5f}{rr - base:>+10.5f}")
    rep = "\n".join(lines)
    with open(os.path.join(iondir, "prune.txt"), "w") as f:
        f.write(rep + "\n")
    print(rep)
    shutil.rmtree(os.path.join(iondir, "prune"), ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ssopt")
    p.add_argument("--data", default="data", help="cache dir for NIST/NORAD files")
    p.add_argument("--workdir", default="work")
    p.add_argument("--binary", default="./structx")
    p.add_argument("--offline", action="store_true", help="never fetch (compute nodes)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("ion")
        sp.add_argument("--n-max", type=int, default=4)
        sp.add_argument("--l-max", type=int, default=3)
        sp.add_argument("--max-exc", type=int, default=2)
        sp.add_argument("--doubles", default="complex", choices=["complex", "paired", "all"],
                        help="complex (default): doubles only within the outer shell; paired: also into higher shells (slow)")
        sp.add_argument("--closed-core", action="store_true",
                        help="declare the inner shells that are full in every configuration closed on the first card")
        sp.add_argument("--e-max", type=float, default=None, help="ignore NIST levels above this (Ry)")
        sp.add_argument("--drop", default=None, help="comma-separated config indices to drop")
        sp.add_argument("--mode", default="l", choices=["l", "orbital"])
        sp.add_argument("--objective", default="rel", choices=["abs", "rel"],
                        help="rel (default): RMS percent error, so low-lying levels count as much as high ones; abs: RMS error in Ry "
                             "(favours the excited manifold over the ground-term fine structure)")
        sp.add_argument("--core-bounds", default="0.9,1.2")
        sp.add_argument("--valence-bounds", default="0.7,2.0")
        sp.add_argument("--freeze-core", action="store_true")
        sp.add_argument("--workers", type=int, default=None)
        sp.add_argument("--timeout", type=float, default=1800)
        sp.add_argument("--scratch", default=None, help="fast local dir for run scratch ($TMPDIR)")
        sp.add_argument("--match-key", default="j", choices=["j", "ls"],
                        help="'j' (default): pair NIST and computed levels within (parity, J, configuration) — "
                             "works for LS-coupled and IC-coupled ions alike, LS-label disagreements are reported "
                             "as 'term'. 'ls' additionally requires the LS term to agree.")

    s = sub.add_parser("fetch"); s.add_argument("ion"); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("deck"); common(s); s.set_defaults(fn=cmd_deck)
    s = sub.add_parser("tune"); common(s)
    s.add_argument("--n-init", type=int, default=32)
    s.add_argument("--rounds", type=int, default=4)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--tol", type=float, default=2e-4)
    s.add_argument("--init-radius", type=float, default=0.2, help="initial design box half-width around λ=1")
    s.add_argument("--trust-radius", type=float, default=0.15)
    s.add_argument("--polish", type=int, default=2, help="coordinate-scan polish iterations (0 = off)")
    s.add_argument("--polish-step", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--init-lambdas", default=None)
    s.add_argument("--keep-runs", action="store_true")
    s.add_argument("--no-restart", action="store_true",
                   help="repeat the angular algebra in every run instead of saving it once (slower, same energies)")
    s.set_defaults(fn=cmd_tune)
    s = sub.add_parser("full"); common(s)
    s.add_argument("--lambdas", required=True, help="comma-separated λ, or 'best' to read history_<mode>.json")
    s.add_argument("--e1-only", action="store_true",
                   help="skip the forbidden E2/M1 transitions (about twice as fast; the E1 data and NORAD tables "
                        "are unchanged, but lifetimes of metastable levels need E2/M1)")
    s.add_argument("--all-transitions", action="store_true",
                   help="radiative data between all levels, including those of the correlation configurations "
                        "(much slower; by default only levels whose leading configuration is a NIST one)")
    s.set_defaults(fn=cmd_full)
    s = sub.add_parser("uncert"); common(s)
    s.add_argument("--max-runs", type=int, default=16, help="at most this many λ sets in the ensemble")
    s.add_argument("--e1-only", action="store_true", help="as for full (then no lifetimes)")
    s.add_argument("--all-transitions", action="store_true", help="as for full")
    s.set_defaults(fn=cmd_uncert)
    s = sub.add_parser("prune"); common(s)
    s.add_argument("--lambdas", required=True, help="comma-separated λ, or 'best'")
    s.set_defaults(fn=cmd_prune)
    s = sub.add_parser("best"); s.add_argument("ion"); s.add_argument("--mode", default="orbital", choices=["l", "orbital"])
    s.set_defaults(fn=cmd_best)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
