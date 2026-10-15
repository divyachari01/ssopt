# ssopt: automated λ tuning for SUPERSTRUCTURE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23380595.svg)](https://doi.org/10.5281/zenodo.23380595)

`ssopt` tunes the scaling parameters λ of the atomic structure code
SUPERSTRUCTURE so that its energy levels come as close as possible to NIST. You give it the name of an
ion. It does the rest.

## What it does

SUPERSTRUCTURE builds each orbital in a Thomas-Fermi-Dirac-Amaldi potential. One
scaling parameter λ per orbital sets the size of that potential. Users normally
adjust these by hand until the computed energies agree with observation. That
takes experience and hours of work. The results are also hard to reproduce.

`ssopt` replaces the hand work. For one ion it:

1. **Builds the configuration list.** Starting from the ground configuration,
   it adds single and paired double excitations into the active orbitals (up to
   n = 4, l = 3 by default), along with every configuration NIST lists in that
   range. The list is therefore larger than NIST's, and the extra
   configurations improve the energies through configuration interaction.
2. **Pairs computed levels with NIST levels.** Only the levels NIST has are used
   to judge the fit, but every other level is still calculated and kept in the
   output.
3. **Tunes λ.** It finds the λ that give the smallest relative energy error,
   using a surrogate model to guide about 300 structure runs.
4. **Runs the full calculation** at the tuned λ, including radiative data.
5. **Checks the A-values** against NIST, and against NORAD where available.
6. **Fixes level labels** so they match NIST, where the energies make it clear.
7. **Writes tables** of levels and transitions in the NORAD file formats.

Nothing is set per ion. The same settings were used for 96 ions: the N- to
P-like sequences up to Z = 28, plus Ne I to Ne X. The code was developed and run
on the Ohio Supercomputer Center's Cardinal cluster, but nothing in it is
specific to that machine. This repository holds the code only. The NIST caches, tuned
decks, `ssout` files and datasets are not included.

## How accurate is it

`ssopt` gets the energies as close to NIST as the SUPERSTRUCTURE model allows,
but it does not make the calculation exact. On a test set of 86 ions (the N- to
P-like sequences up to Z = 28), the median RMS error of the matched levels after
tuning is about 1%, and 74 of the 86 ions are within 3%. Taken level by level,
about 90% of all levels are within 3% of NIST. The Al-, Si- and P-like ions are
harder, at 1 to 4% (about 6% at worst), because an n ≤ 4 deck does not capture
all of their correlation; Ni XV, for example, comes out at 3%.

It works best for ions with a charge of 3 or more. Neutral and near-neutral
atoms are harder for the model potential, so they have the largest outliers
(Mg I is off by 24%) and their lifetimes agree less well with experiment.
Results for these atoms should be used with more caution.

## Quick start

```bash
python -m ssopt.cli fetch "Ni XV"                        # NIST levels and lines -> data/ (needs internet)
sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6   # tune lambda: per-l stage, per-orbital stage, prune
sbatch slurm/full.slurm "Ni XV" orbital best             # full radiative run at the tuned lambda, A-value check
python -m ssopt.export "Ni XV" --workdir work --out work/ni15 --relabel-ssout   # NIST labels, NORAD tables
```

## What you need

* **`structx`, the SUPERSTRUCTURE program.** `ssopt` does not do the atomic
  physics itself: it writes the input file (`ssin`), runs SUPERSTRUCTURE, and
  reads the output (`ssout`). `structx` is our name for the compiled
  SUPERSTRUCTURE program, which is not part of this repository. To get it:
  1. Get the Fortran source, `struct.f`, from the code's authors, for example
     through Prof. Sultana Nahar's group at Ohio State or the Iron Project.
  2. Compile it:
     `gfortran -O2 -std=legacy -ffixed-line-length-72 struct.f -o structx`.
     The Intel compiler (`ifort -O2 struct.f -o structx`) gives identical
     results.
  3. Put `structx` in the top folder of this repository, or link to it from
     there with `ln -s /path/to/structx structx`.

  One `structx` works for most atoms and ions, because the array sizes set by
  the `PARAMETER` statements in `struct.f` are large enough for typical decks.
  A very large deck, with many configurations or levels, can exceed them. The
  run then stops with a message such as `INCREASE MAXUC`, `INCREASE MAXDK` or
  `MAXTR .GE. ... IS REQUIRED`, and the fix is to raise the named sizes in
  `struct.f` and compile a new `structx`.
* Python 3.10 or newer with numpy and scipy. Nothing else: the Gaussian process
  is written in numpy, because OSC's Python module has no scikit-learn.
* Internet on the node that runs `fetch`. Compute nodes work offline from the
  `data/` cache.

## Where the results are

Everything for one ion goes into one folder, `work/<ion>/` (for Ni XV,
`work/ni15/`). These are the files you will most likely want:

| File | What it is |
|---|---|
| `ssin.tuned_orbital` | the SUPERSTRUCTURE input with the tuned λ. To rerun it: `./structx < ssin.tuned_orbital > ssout` |
| `levels_orbital.txt` | every computed level next to its NIST value, with the difference in Ry |
| `ssout.full` | the full SUPERSTRUCTURE output at the tuned λ. It has the energies, the mixing and all A-values |
| `avalues.txt` | the A-values compared line by line with NIST (and NORAD) |
| `en.fs`, `f.fs`, ... | level and transition tables in NORAD format, written by `export` |

The tuned λ are on the λ line of `ssin.tuned_orbital`, and SUPERSTRUCTURE also
prints them in `ssout.full` on the `SCALE FACTORS` lines.

## Words used here

* **λ (lambda)**: a scaling factor, one per orbital (1s, 2s, 2p, ...), that
  sets how strongly the model potential pulls that orbital in. λ = 1 means no
  scaling.
* **NIST**: the NIST Atomic Spectra Database, the standard list of measured
  energy levels and transition rates.
* **ssin / ssout**: SUPERSTRUCTURE's input and output files.
* **Level**: one energy state of the ion, named by its configuration, LS term
  and J, for example 3s²3p² ³P₂.
* **A-value**: how fast an excited level decays to a lower one by emitting a
  photon, in decays per second.
* **Lifetime**: how long a level lives on average, τ = 1/ΣA, which is what
  laboratories measure.
* **NORAD**: NORAD-Atomic-Data, the atomic database at Ohio State, whose file
  formats `ssopt` writes.

## How it works, in plain words

Think of a SUPERSTRUCTURE deck as a box with one knob per orbital, λ(1s) to
λ(4f). Each run of the code takes seconds to minutes and returns tens to
hundreds of level energies. We want the knob settings that make those energies
agree best with NIST.

Turning ten knobs by hand, with minutes per try, is hopeless. So the code does
what a careful person would do, only faster. It tries a spread of settings. It
notices the pattern. It guesses where the best setting is. It tests the guess.
Then it repeats.

The important design choice is what the code builds a model of. The obvious
choice is the score: one number per run, the RMS error. We do not model that.
Every run returns many level energies, and each energy changes smoothly, almost
linearly, when a knob is turned. So the code models every level on its own.
"Level 17 goes up 0.3 Ry when λ(3p) goes up 0.1, a bit more if λ(3s) also
moves." The score is then computed from the predicted energies. This uses all
the information in a run instead of throwing most of it away.

## The pieces

**Deck** (`ssopt/configs.py`). Two sets of configurations are merged. The
spectroscopic set is every configuration named in the NIST level list within
the cap (default n ≤ 4, l ≤ 3). The correlation set is the single excitations
and the paired double excitations from the ground configuration into the same
active orbitals. The core (1s² or 1s²2s²2p⁶) stays closed.

**Runs** (`ssopt/runner.py`). During tuning, `structx` is stopped as soon as the
level table is written. The radiative part costs 20 to 60 times more and is
needed only once, at the end.

**Matching** (`ssopt/match.py`). Only NIST levels whose configuration is in the
deck are targets. Within each (parity, 2J) group, computed levels are limited
to the configurations NIST names there and paired one-to-one by nearest energy.
LS term labels are not required to agree. See "Labels" below for why.

**Objective** (`ssopt/match.py`). With `--objective rel`, the default, every
level counts by its percentage error. A floor at 1% of the highest target energy
keeps the ground-term fine structure from dominating. `--objective abs` is the
plain RMS in Ry. The relative objective is what gets the ground-term fine
structure right. The Ry objective trades it away for the many high levels.

**Surrogate optimiser** (`ssopt/surrogate.py`, `ssopt/gp.py`). A linear trend
per level plus one multi-output Gaussian process with a shared kernel. Sobol
design, trust-region rounds, coordinate polish. The mathematics are below.

**Two stages** (`slurm/ion.slurm`). First one λ per angular momentum l, which is
four parameters and SUPERSTRUCTURE's `MSCAL` convention. Then one λ per orbital,
ten parameters, starting from the first result.

**Prune** (`cli.py prune`). Each correlation group is removed in turn at the
tuned λ to see which correlation actually moves the energies. Diagnostic only.

**A-values** (`ssopt/avalues.py`). A read-only check of the full run against
NIST and NORAD, with the length/velocity gauge ratio for every line. A-values
never enter the objective.

**Export** (`ssopt/export.py`). NORAD-format tables `en.ls`, `en.fs`, `f.ls`,
`f.fs` (level-index layout) and `f.fsm` (multiplet layout), plus the label
consistency step.

## The mathematics

Let E_k(λ) be the computed energy of matched level k relative to the ground
level, and E_k^obs its NIST value. The quantity minimised is

    F(λ) = sqrt( (1/K) Σ_k w_k² (E_k(λ) − E_k^obs)² )

with w_k = 1 for the absolute objective (RMS in Ry), or

    w_k = 1 / max(E_k^obs, 0.01 E_max^obs)

for the relative objective, the default.

Every matched level is modelled as

    E_k(λ) = a_k + b_k · λ + g_k(λ),      g_k ~ GP(0, κ)

That is a straight line fitted by least squares, plus a Gaussian-process
correction with the squared-exponential kernel

    κ(λ, λ') = α² exp[ −½ Σ_i (λ_i − λ'_i)² / ℓ_i² ] + σ² δ

The hyperparameters (α, ℓ_i, σ) are shared by all levels. They are found by
maximising the summed log marginal likelihood. A small ℓ_i means the energies
change fast along knob i. A large ℓ_i means that knob barely matters. Levels
not matched in a given run are filled in from their linear trend, so all levels
share the same design points and one Cholesky factorisation serves all of them.

The surrogate score F̂(λ) is F evaluated with the posterior means. The optimistic
variant F̂_β replaces each |Ê_k − E_k^obs| by max(|Ê_k − E_k^obs| − β σ̂_k, 0).
It rewards points where the model is unsure.

The loop has three parts.

1. Design. Run λ = 1 and 47 Sobol points in the box 1 ± 0.2, within the bounds.
2. Rounds, five of them. Fit the model on the well-behaved runs (F within 10× of
   the best). Inside the trust region λ_best ± ρ, propose the minimiser of F̂,
   the minimisers of F̂_1 and F̂_2, and Gaussian perturbations of the first,
   12 points in all. Run them and refit. ρ grows by 1.5 on improvement and
   halves otherwise.
3. Polish. Coordinate scans of ±0.1, then ±0.05, around the best point, each
   followed by a surrogate minimisation inside the scan box.

The bounds are 0.8 to 1.6 for core orbitals and 0.7 to 2.0 for valence orbitals
(`--core-bounds`, `--valence-bounds`). The runs of a batch execute in parallel
(`--workers`). With 48 cores an ion takes a few minutes to an hour per stage.

## What the tuned λ mean, and what they don't

On the 86-ion sweep, about half of all runs land within 25% of the best RMS.
The objective is flat along most directions in λ space. Among near-optimal runs,
λ(1s) wanders over ±0.3 with no effect. Only λ(2p) is well pinned down, with
λ(2s) and λ(3p) next. So the tuned value of most λ_nl is not a property of the
ion. It is a property of where the optimiser stopped, and should not be read
physically.

`ssopt/predict.py` fits a map from (Z, N, n, l) to λ_nl across tuned ions, with
a leave-one-ion-out test that actually runs structx at the predicted λ. Given
the flatness above, treat its predictions as a starting point for the optimiser
(`--init-lambdas`), not as a replacement.

## Labels: making the ssout consistent with NIST

SUPERSTRUCTURE names each level after its largest LS component. For strongly
mixed pairs with the same configuration, parity and J, that name can flip while
the energies are right. In Ni XV, the 3s²3p² level at 0.246 Ry comes out called
¹D₂ and the one at 0.578 Ry ³P₂. NIST has ³P₂ at 0.249 and ¹D₂ at 0.573. In
Ar IV, Cr XVIII and Ni XXII it is the ground level, p³ ⁴S°₃/₂, that carries the
wrong name. The optimiser is not fooled, because it pairs by J and energy. But
anything that reads the ssout by label is.

`python -m ssopt.export "<ion>" --relabel-ssout` rewrites the fine-structure
level table in `work/<ion>/ssout.full` so that a level takes the NIST label
when, and only when, the energies force it. The rule is deliberately cautious.

* Take a level whose SS term X differs from its NIST partner's term Y. Build the
  label-respecting alternative: SS-X paired with the nearest NIST-X, and SS-Y
  with the nearest NIST-Y, in the same configuration, parity and J.
* Accept the swap only if that alternative would put the levels off by more
  than 25% of their energy, and is at least three times worse than the swap.
  Two close terms of a light ion that merely cross because of a few-percent
  deck error never get near that. They keep SS's label.
* Never relabel from a NIST level whose term is flagged uncertain with `?`.
* Levels without a NIST partner keep SS's label.

The original file is kept once as `ssout.full.ss`. Energies, eigenvectors, the
term table and the radiative tables are untouched. E1 rows refer to levels by
index, so they stay consistent. `python swaps.py work` lists every relabelled
level across all ions, with its NIST partner, for inspection. On the 96-ion set
the rule changes 10 levels in 5 ions, the five named above, all checked by hand.

## Lifetimes against experiment

Laboratories measure the lifetime of a level, τ = 1/ΣA over every decay
channel. So the experimental test of the A-values is a test on lifetimes.

`python -m ssopt.lifetimes "Fe XVII" --workdir work` sums every rate that
SUPERSTRUCTURE prints in the radiative section: E1 and the E3/M2 rows in part 1,
E2 and M1 in part 2. It lists τ per level with the channel breakdown.

`--bench benchmarks/lifetimes_exp.csv` compares with measured values. The file
ships with 139 measurements for 22 ions, from storage rings, EBITs, ion traps,
beam-foil and laser experiments. Each row carries the URL it was read from. Rows
with J = term are measurements that do not resolve J. They are compared with the
g-weighted mean over the term. The columns are: ion, configuration outside the
closed core in NIST style, term with a parity star, J, τ, uncertainty, unit,
method, reference. Extend it from the literature as needed. Run
`--relabel-ssout` first, so that level names match.

## Workflow on a cluster (Slurm)

The scripts in `slurm/` work on any cluster that uses Slurm (OSC, Unity, a
university cluster and so on). They contain no account or cluster names. Set
these once in your shell, for example in `~/.bashrc`:

```bash
export SBATCH_ACCOUNT=<your allocation>          # if your cluster needs one
export SSOPT_MODULES="python"                    # modules to load in each job, if any
```

Each script is a short wrapper around `python -m ssopt.cli`, so it is easy to
adapt to another batch system or to run the same commands on a workstation.
Edit the core count, memory and time limits at the top of a script if your
nodes differ.

```bash
cd ssopt && ln -s /path/to/structx structx               # once
python -m ssopt.cli fetch "Ni XV"                         # login node, once per ion
sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6    # tune lambda; 48 cores, 4 h limit, --resume is safe
sbatch slurm/full.slurm "Ni XV" orbital best              # full radiative run, 1 to 6 h; runs export at the end
bash slurm/sweep.sh ions_fe.txt --core-bounds 0.8,1.6     # a whole list; skips tuned ions
python -m ssopt.dataset --workdir work --out dataset      # CSVs: ions, levels, samples, A-values
sbatch slurm/verify.slurm                                 # lambda predictor, leave-one-ion-out
python -m ssopt.figures --dataset dataset
```

Everything goes into `work/` and `dataset/` by default. To keep a set of runs
in another folder, put `WORKDIR=<folder>` in front of the `sbatch` and `sweep`
commands, for example `WORKDIR=my_runs sbatch slurm/ion.slurm ...`, and use the
same folder in `--workdir`. For `verify.slurm`, also set `DATASET=<folder>`.

Per ion, `work/<tag>/` holds:

* `deck.json`, the configuration list;
* `history_l.json` and `history_orbital.json`, every λ tried and every matched
  level energy;
* `ssin.tuned_*`, the tuned decks;
* `levels_*.txt`, the level-by-level comparison with NIST;
* `prune.txt`, `ssout.full`, `avalues.txt`;
* after export, the five NORAD-format tables.

Extra arguments to `ion.slurm` go to both tune stages, for example
`--objective abs`, `--n-max 5` or `--core-bounds 0.8,1.6`.

## SUPERSTRUCTURE facts learned the hard way

* `MSCAL = n_orbitals` means λ grouped by l: the input values are λ_s, λ_p,
  λ_d, λ_f and the rest are ignored. `MSCAL = 1000 + n` gives one λ per orbital.
  The `SCALE FACTORS` echo in `ssout` shows what was applied.
* Card 1 columns 5 to 8 (`QCS0 QCL0`) are read as core specifiers before being
  reinterpreted as radiative switches. Setting them to −1 to skip radiative
  data hangs the radial stage. Stop the run on the marker instead.
* Free-format λ (`LFREE='Y'`) accepts four decimals.
* The J-level table has no parity column. Parity comes from the configuration
  `CF`. The term table has signed S, with negative meaning odd.
* `MAXTR` in a given build may be too small for the `fort.7` punch of a large
  deck (`MAXTR.GE. ... IS REQUIRED`). The punch is then incomplete, but the
  ssout is not. This matters only for downstream R-matrix use.
* `'VEL.GF'` in the radiative table is not the velocity gf. The stored
  velocity/length ratio sits near 0.25 for every ion in this build, so the gauge
  test uses four times the stored value.

## Layout

```
ssopt/       ions configs deck ssout nist norad match runner gp surrogate
             avalues cli dataset predict figures export lifetimes
slurm/       ion.slurm full.slurm sweep.sh verify.slurm optimize.slurm
tests/       python -m pytest tests/   (needs data/nist_Si_I_levels_ry.csv: run fetch "Si I" first)
ref/         ssin.cardinal (reference deck), ssout.si1.levels (reference output, test fixture)
benchmarks/  lifetimes_exp.csv, measured lifetimes for ssopt.lifetimes --bench
swaps.py     lists relabelled levels with their NIST partners
ions_fe.txt, ions_norad.txt   ion lists used for the sweeps
```

## Citing

If you use this code or the tuned data, please cite it. GitHub's "Cite this
repository" button gives the reference. Until the method paper is out, cite the
software:

> D. Chari, S. N. Nahar and A. K. Pradhan, *ssopt: automated tuning of
> Thomas-Fermi scaling parameters for SUPERSTRUCTURE* (2026), Zenodo,
> https://doi.org/10.5281/zenodo.23380595

This DOI always points to the latest version. Each release also has its own
DOI, listed on the Zenodo page, for citing the exact version you used.

Contributors: Alain Ward (Department of Physics, The Ohio State University).

Please also cite SUPERSTRUCTURE itself: Eissner, Jones & Nussbaumer (1974) and
Nussbaumer & Storey (1978). For the forbidden transitions (E2, M1, M2, E3),
which `ssopt.lifetimes` uses, also cite Nahar, Eissner, Chen & Pradhan (2003).
If you use NORAD data, the NORAD comparisons or the NORAD-format tables, cite
NORAD as its page asks, with S. N. Nahar, Atoms 8, 68 (2020) and S. N. Nahar &
G. Hinojosa-Aguirre, Atoms 12, 22 (2024), and acknowledge NORAD-Atomic-Data
(https://norad.astronomy.osu.edu).

## License

The method and the use of SUPERSTRUCTURE follow the atomic-structure work of
Profs. Sultana N. Nahar and Anil K. Pradhan's group at The Ohio State
University. Copyright 2026 Divya Chari, Sultana N. Nahar and Anil K. Pradhan,
The Ohio State University.

The code is released under the Creative Commons Attribution-NonCommercial 4.0
International License (CC BY-NC 4.0, see `LICENSE`), the same terms as
NORAD-Atomic-Data. You may use, share and adapt it for non-commercial purposes,
with credit and citation. For commercial use, contact the authors.

## Related work

Tuning a structure code's free parameters against NIST with a learned surrogate
has been done for other codes. Ferreira da Silva et al. (2025) tune the Flexible
Atomic Code's mean-configuration occupancies for lanthanide and actinide ions by
Bayesian optimisation of the RMS energy error. Ma et al. (2024) correct the Cowan
code's Slater and spin-orbit parameters for Yb I with ElasticNet and XGBoost
models. AUTOSTRUCTURE can minimise an energy functional with respect to its λ
internally (Badnell 2011); that is variational, not a fit to observation.

`ssopt` differs in four ways. It works on SUPERSTRUCTURE's λ_nl. It models every
level energy rather than the scalar error. It builds the configuration list and
the level pairing from the NIST list with no per-ion input. And it tests the
result against measured lifetimes.

* R. Ferreira da Silva, A. Flörs, L. Leitão, J. P. Marques, J. M. Sampaio,
  G. Martínez-Pinedo, Systematic Bayesian Optimization for Atomic Structure
  Calculations of Heavy Elements, arXiv:2502.13250 (2025),
  https://arxiv.org/abs/2502.13250
* K. Ma, C. Yang, J. Zhang, Y. Li, G. Jiang, J. Chai, Machine Learning-Assisted
  Hartree-Fock Approach for Energy Level Calculations in the Neutral Ytterbium
  Atom, Entropy 26, 962 (2024), https://doi.org/10.3390/e26110962
* N. R. Badnell, A Breit-Pauli distorted wave implementation for AUTOSTRUCTURE,
  Comput. Phys. Commun. 182, 1528 (2011)

## References

SUPERSTRUCTURE

* W. Eissner, M. Jones, H. Nussbaumer, Techniques for the calculation of atomic
  structures and radiative data including relativistic corrections, Comput.
  Phys. Commun. 8, 270 (1974). The original program.
* H. Nussbaumer, P. J. Storey, The C III transition probabilities, Astron.
  Astrophys. 64, 139 (1978). Semi-empirical term-energy corrections.
* S. N. Nahar, W. Eissner, G. X. Chen, A. K. Pradhan, Atomic data from the Iron
  Project LIII. Relativistic allowed and forbidden transition probabilities for
  Fe XVII, Astron. Astrophys. 408, 789 (2003). Early versions of the program
  did not compute forbidden transitions correctly, so W. Eissner fixed them and
  added more transition types (E2, M1, M2, E3). This paper is the first
  publication of that version, which is the one `ssopt` reads.

Data

* NIST Atomic Spectra Database, A. Kramida, Yu. Ralchenko, J. Reader and NIST
  ASD Team, https://physics.nist.gov/asd
* S. N. Nahar, NORAD-Atomic-Data: a database for spectroscopy and many other
  applications, Atoms 8, 68 (2020); S. N. Nahar, G. Hinojosa-Aguirre, Atoms 12,
  22 (2024); https://norad.astronomy.osu.edu
* Measured lifetimes: one reference per row in `benchmarks/lifetimes_exp.csv`.

Related methods: Ferreira da Silva et al. (2025), Ma et al. (2024) and Badnell
(2011), listed under "Related work" above.
