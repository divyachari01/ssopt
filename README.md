# ssopt: automated λ tuning for SUPERSTRUCTURE

`ssopt` tunes the Thomas-Fermi-Dirac-Amaldi scaling parameters λ_nl of the
atomic structure code SUPERSTRUCTURE against the NIST energy levels of an ion,
with no per-ion input beyond the ion's name. It builds the configuration list
from the NIST level list, pairs computed and observed fine-structure levels,
minimises the relative RMS energy error with a surrogate-model optimiser
(about 300 structure runs per ion), runs the full radiative calculation at the
tuned λ, checks the A-values against NIST (and NORAD where available), makes
the level labels consistent with NIST where the energies decide it, and writes
level and transition tables in the NORAD file formats.

It was developed on OSC Cardinal and run on 96 ions of the N- to P-like
sequences (Z ≤ 28) plus Ne I-X. The data produced (NIST caches, tuned decks,
`ssout` files, datasets) are not in this repository; only the code is.

```
python -m ssopt.cli fetch "Ni XV"                        # login node: NIST levels + lines -> data/
sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6   # tune: per-l stage -> per-orbital stage -> prune
sbatch slurm/full.slurm "Ni XV" orbital best             # full radiative run at the tuned λ + A-value check
python -m ssopt.export "Ni XV" --workdir work --out work/ni15 --relabel-ssout   # NIST-consistent labels + NORAD tables
```

## What you need

* SUPERSTRUCTURE (Eissner, Jones & Nussbaumer 1974; Nussbaumer & Storey 1978),
  compiled as `./structx` in the repository root. The Fortran source is not
  included here; obtain it from its authors or from the Iron Project / NORAD
  distributions. `ifort -O2` and `gfortran -std=legacy -ffixed-line-length-72`
  give identical level tables.
* Python 3.10+ with numpy and scipy (nothing else: OSC's python module has no
  scikit-learn, so the Gaussian process is written in numpy).
* Internet on the node that runs `fetch` (NIST ASD). Compute nodes run offline
  from the `data/` cache.

## How it works, in plain words

A SUPERSTRUCTURE deck has one knob per orbital, λ(1s) ... λ(4f). Each structure
run takes seconds to minutes and returns tens to hundreds of level energies.
The question is which setting of the knobs makes those energies agree best with
NIST. Turning ten knobs by hand with minutes per try is hopeless, so the code
does what a careful person would do, faster: try a spread of settings, notice
the pattern, guess where the best one is, test the guess, repeat.

The important design choice is *what* it builds a model of. The obvious thing
is the score (one number per run, the RMS error). We don't model that. Every
run gives back many level energies, and each energy changes smoothly, almost
linearly, when a knob is turned. So the code models every level separately
("level 17 goes up 0.3 Ry when λ(3p) goes up 0.1, a bit more if λ(3s) also
moves") and computes the score from the predicted energies. That uses all the
information in a run instead of throwing most of it away.

### The pieces

1. **Deck** (`ssopt/configs.py`). Spectroscopic set: every configuration named
   in the NIST level list within the cap (default n ≤ 4, l ≤ 3). Correlation
   set: single excitations and paired double excitations from the ground
   configuration into the same active set. Core (1s² or 1s²2s²2p⁶) closed.
   Nothing is hard-coded per ion.
2. **Runs** (`ssopt/runner.py`). `structx` is killed the moment the level table
   is complete (marker `LIST OF TERMS WITH A WEIGHTED MEAN`); the radiative
   section costs 20 to 60 times more and is only needed once, at the end.
3. **Matching** (`ssopt/match.py`). Only NIST levels whose configuration is in
   the deck are targets. Inside each (parity, 2J) bucket the computed candidates
   are limited to the configurations NIST names in that bucket and paired
   one-to-one by nearest energy. LS term labels are deliberately not required to
   agree (see "Labels" below).
4. **Objective** (`ssopt/match.py`, `target_weights`). With `--objective rel`
   (default) every level counts by its percentage error, with a floor at 1% of
   the highest target energy. `--objective abs` is the plain RMS in Ry. The
   relative objective is what gets the ground-term fine structure right; the
   Ry objective sacrifices it for the many high levels.
5. **Surrogate optimiser** (`ssopt/surrogate.py`, `ssopt/gp.py`). Per-level
   linear trend plus a multi-output Gaussian process with one shared kernel;
   Sobol design, trust-region rounds, coordinate polish. Details below.
6. **Two stages** (`slurm/ion.slurm`). First one λ per angular momentum l
   (4 parameters, SUPERSTRUCTURE's `MSCAL` convention), then one λ per orbital
   (10 parameters) seeded from the first.
7. **Prune** (`cli.py prune`). Leave-group-out of each correlation group at the
   tuned λ: which correlation actually moves the energies. Diagnostic only.
8. **A-values** (`ssopt/avalues.py`). Read-only check of the full run against
   NIST and NORAD, with the length/velocity gauge ratio per line. Never in the
   objective.
9. **Export** (`ssopt/export.py`). NORAD-format `en.ls`, `en.fs`, `f.ls`,
   `f.fs` (level-index layout) and `f.fsm` (multiplet layout) tables, and the
   label consistency step.

### The mathematics

Let E_k(λ) be the computed energy of matched level k relative to the ground
level and E_k^obs its NIST value. The quantity minimised is

    F(λ) = sqrt( (1/K) Σ_k w_k² (E_k(λ) − E_k^obs)² )

with w_k = 1 (absolute objective, RMS in Ry) or

    w_k = 1 / max(E_k^obs, 0.01 E_max^obs)      (relative objective, default)

Every matched level is modelled as

    E_k(λ) = a_k + b_k · λ + g_k(λ),      g_k ~ GP(0, κ)

a straight line fitted by least squares plus a Gaussian-process correction with
the squared-exponential kernel

    κ(λ, λ') = α² exp[ −½ Σ_i (λ_i − λ'_i)² / ℓ_i² ] + σ² δ

whose hyperparameters (α, ℓ_i, σ) are shared by all levels and found by
maximising the summed log marginal likelihood. A small ℓ_i means "the energies
change fast along knob i"; a large one means "that knob barely matters". Levels
not matched in a given run are imputed from their linear trend so all levels
share the same design points and one Cholesky factorisation serves all of them.
The surrogate score F̂(λ) is F evaluated with the posterior means; the
optimistic variant F̂_β replaces each |Ê_k − E_k^obs| by
max(|Ê_k − E_k^obs| − β σ̂_k, 0), rewarding points where the model is unsure.

The loop:

1. Design: λ = 1 plus 47 Sobol points in the box 1 ± 0.2 (within the bounds).
2. Round (×5): fit the model on the well-behaved runs (F within 10× of the
   best); inside the trust region λ_best ± ρ propose the minimiser of F̂, the
   minimisers of F̂_1 and F̂_2, and Gaussian perturbations of the first
   (12 points); run them; refit. ρ grows ×1.5 on improvement, halves otherwise.
3. Polish: coordinate scans of ±0.1 then ±0.05 around the best point, each
   followed by a surrogate minimisation inside the scan box.

Bounds: 0.8 to 1.6 for core orbitals, 0.7 to 2.0 for valence orbitals
(`--core-bounds`, `--valence-bounds`). The runs of a batch execute in parallel
(`--workers`); with 48 cores an ion takes a few minutes to an hour per stage.

### What the tuned λ mean, and don't

On the 86-ion sweep about half of all runs land within 25% of the best RMS: the
objective is flat along most directions of λ space. Among near-optimal runs
λ(1s) wanders over ±0.3 with no effect; only λ(2p), and less so λ(2s) and
λ(3p), are pinned down. So the tuned values of most λ_nl are not properties of
the ion but of where the optimiser stopped, and they should not be read
physically. `ssopt/predict.py` fits a map (Z, N, n, l) → λ_nl across tuned ions
(GP or per-sequence polynomial) with a leave-one-ion-out test that actually
runs structx at the predicted λ; given the above, treat its predictions as a
starting point for the optimiser (`--init-lambdas`), not a replacement.

## Labels: making the ssout consistent with NIST

SUPERSTRUCTURE names each level after its largest LS component. For strongly
mixed pairs of the same configuration, parity and J the name flips while the
energies are right: in Ni XV the 3s²3p² level at 0.246 Ry comes out called
¹D₂ and the one at 0.578 Ry ³P₂, where NIST has ³P₂ at 0.249 and ¹D₂ at 0.573.
In Ar IV, Cr XVIII and Ni XXII it is the ground level (p³ ⁴S°₃/₂) that carries
the wrong name. The optimiser is not fooled (it pairs by J and energy within the
configuration) but anything that reads the ssout by label is.

`python -m ssopt.export "<ion>" --relabel-ssout` rewrites the fine-structure
level table in `work/<ion>/ssout.full` so that a level takes the NIST label
when, and only when, the energies force it. The rule, deliberately
conservative:

* For a level whose SS term X differs from its NIST partner's term Y, build the
  label-respecting alternative (SS-X with the nearest NIST-X, SS-Y with the
  nearest NIST-Y in the same configuration/parity/J bucket).
* Accept the swap only if that alternative would put the levels off by more
  than 25% of their energy and is at least 3 times worse than the swap. Two
  close terms of a light, LS-coupled ion that merely cross because of a
  few-percent deck error never get near that, and keep SS's label.
* Never relabel from a NIST level whose term is flagged uncertain (`?`).
* Levels without a NIST partner keep SS's label.

The original is kept once as `ssout.full.ss`; energies, eigenvectors, the term
table and the radiative tables are untouched (E1 rows refer to levels by index,
so they stay consistent). `python swaps.py work` lists every relabelled level
across all ions with its NIST partner for inspection. On the 96-ion set the
rule changes 10 levels in 5 ions (the five named above), all verified by hand.


## Lifetimes against experiment

What laboratories measure is the lifetime of a level, τ = 1/ΣA over every decay
channel, so the experimental test of the A-values is on lifetimes.
`python -m ssopt.lifetimes "Fe XVII" --workdir work` sums every rate SUPERSTRUCTURE
prints in the IC radiative section (E1 and the E3/M2 rows in part 1, E2 and M1 in
part 2) and lists τ per level with the channel breakdown.
`--bench benchmarks/lifetimes_exp.csv` compares with measured values; the file
ships with 139 measured values for 22 ions (storage ring, EBIT, ion trap, beam-foil,
LIF), each with the URL it was read from; rows with J = term are J-unresolved
measurements, compared with the g-weighted mean over the term. Extend it from the
literature as needed (columns: ion,
config outside the closed core in NIST style, term with parity star, J, τ, unc,
unit, method, reference). Run `--relabel-ssout` first so level names match.

## Workflow on a cluster (OSC Cardinal)

```bash
cd ~/SS/ssopt && ln -s ../structx structx
module load python/3.12
python -m ssopt.cli fetch "Ni XV"                       # login node, once per ion
sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6  # 48 cores, 4 h limit, --resume safe
sbatch slurm/full.slurm "Ni XV" orbital best            # single core, 1-6 h; runs export at the end
WORKDIR=work_rel bash slurm/sweep.sh ions_fe.txt --core-bounds 0.8,1.6   # whole list, skips tuned ions
python -m ssopt.dataset --workdir work_rel --out dataset_rel             # ions/levels/samples/avalues CSVs
sbatch slurm/verify.slurm                                                # λ predictor leave-one-ion-out
python -m ssopt.figures --dataset dataset_rel
```

Per ion, `work/<tag>/` holds `deck.json`, `history_l.json` and
`history_orbital.json` (every λ tried and every matched level energy),
`ssin.tuned_*`, `levels_*.txt` (level-by-level comparison with NIST),
`prune.txt`, `ssout.full`, `avalues.txt`, and after export the five NORAD-format
tables. Extra arguments to `ion.slurm` go to both tune stages
(`--objective abs|rel`, `--n-max`, `--core-bounds`, ...).

## SUPERSTRUCTURE facts learned the hard way

* `MSCAL = n_orbitals` means λ grouped by l (λ_s λ_p λ_d λ_f, the rest
  ignored); `MSCAL = 1000 + n` gives one λ per orbital. The `SCALE FACTORS`
  echo in `ssout` shows what was applied.
* Card 1 columns 5-8 (`QCS0 QCL0`) are read as core specifiers before being
  reinterpreted as radiative switches; setting them to −1 to skip radiative
  data hangs the radial stage. Kill on the marker instead.
* Free-format λ (`LFREE='Y'`) accepts 4 decimals.
* The J-level table has no parity column; parity comes from the configuration
  `CF`. The term table has signed S: negative = odd.
* `MAXTR` in a given build may be too small for the `fort.7` punch of a large
  deck (`MAXTR.GE. ... IS REQUIRED`); the punch is incomplete, the ssout is not.
  Only matters for downstream R-matrix use.
* `'VEL.GF'` in the radiative table is not the velocity gf: the stored
  velocity/length ratio sits at about 0.25 for every ion in this build, so the
  gauge test uses 4× the stored value.

## Layout

```
ssopt/    ions configs deck ssout nist norad match runner gp surrogate avalues
          cli dataset predict figures export
slurm/    ion.slurm full.slurm sweep.sh verify.slurm optimize.slurm
tests/    python -m pytest tests/   (needs data/nist_Si_I_levels_ry.csv: run fetch "Si I" first)
ref/      ssin.cardinal (reference deck), ssout.si1.levels (reference output, test fixture)
swaps.py  list relabelled levels with their NIST partners
benchmarks/lifetimes_exp.csv   measured lifetimes for ssopt.lifetimes --bench
ions_fe.txt, ions_norad.txt   ion lists used for the sweeps
```

## References

* W. Eissner, M. Jones, H. Nussbaumer, Comput. Phys. Commun. 8, 270 (1974)
* H. Nussbaumer, P. J. Storey, Astron. Astrophys. 64, 139 (1978)
* NIST Atomic Spectra Database, https://physics.nist.gov/asd
* NORAD-Atomic-Data, S. N. Nahar, https://norad.astronomy.osu.edu
