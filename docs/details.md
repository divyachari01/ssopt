# ssopt: method details

This page collects the details behind the short description in the README.

## The objective

Let E_k(λ) be the computed energy of matched level k relative to the ground
level, and E_k^obs its NIST value. The quantity minimised is

    F(λ) = sqrt( (1/K) Σ_k w_k² (E_k(λ) − E_k^obs)² )

With `--objective rel` (the default) every level counts by its relative error,

    w_k = 1 / max(E_k^obs, 0.01 E_max^obs),

where the floor at 1% of the highest target energy keeps the ground-term fine
structure from dominating. With `--objective abs`, w_k = 1 and F is the plain
RMS in Ry, which trades the ground-term fine structure away for the many high
levels.

## The surrogate model

Every matched level is modelled as

    E_k(λ) = a_k + b_k · λ + g_k(λ),      g_k ~ GP(0, κ)

that is, a straight line fitted by least squares plus a Gaussian-process
correction with the squared-exponential kernel

    κ(λ, λ') = α² exp[ −½ Σ_i (λ_i − λ'_i)² / ℓ_i² ] + σ² δ

The hyperparameters (α, ℓ_i, σ) are shared by all levels and found by
maximising the summed log marginal likelihood. Levels not matched in a given
run are filled in from their linear trend, so one Cholesky factorisation serves
all levels. The surrogate score F̂(λ) is F evaluated with the predicted
energies. Its optimistic variant F̂_β replaces each |Ê_k − E_k^obs| by
max(|Ê_k − E_k^obs| − β σ̂_k, 0), which rewards points where the model is
unsure. Modelling every level, rather than the single RMS number, uses all the
information in a run.

## The search

1. Design: λ = 1 and 47 Sobol points in the box 1 ± 0.2, within the bounds.
2. Five rounds: fit the model on the well-behaved runs (F within 10 times the
   best), then propose the minimisers of F̂, F̂_1 and F̂_2 inside the trust
   region λ_best ± ρ, plus Gaussian perturbations, 12 points in all. ρ grows by
   1.5 on improvement and halves otherwise.
3. Polish: coordinate scans of ±0.1 and then ±0.05 around the best point, each
   followed by a surrogate minimisation inside the scan box.

This is done twice: first with one λ per orbital angular momentum (s, p, d, f,
SUPERSTRUCTURE's `MSCAL = n` convention), then with one λ per orbital
(`MSCAL = 1000 + n`), starting from the first result. The bounds are 0.8 to 1.6
for core orbitals and 0.7 to 2.0 for valence orbitals (`--core-bounds`,
`--valence-bounds`). The runs of a batch execute in parallel (`--workers`).

## Speeding up the runs

* **Saved angular algebra.** The first run of a tuning stage saves
  SUPERSTRUCTURE's angular algebra in its restart file `fort.10` (restart
  switch −1 in columns 1 to 2 of the terminator card), and every later run reads
  it back (switch 4) and only does the radial part. The energies are identical
  to a run from scratch, and for S II one run goes from 60 s to 3 s.
  `--no-restart` switches this off.
* **Stopping early.** During tuning `structx` is stopped as soon as the level
  table is written, since the radiative part is not needed for the energies.
* **Transitions between NIST-type levels only.** In the full run, columns 5 to
  8 of the terminator card (LCE1, LCE2) are set to the number of NIST
  configurations, which are listed first in the deck. SUPERSTRUCTURE then
  computes transitions only between levels whose leading configuration is one of
  them, with the same rates as in a run with all transitions. In S II this is
  128 of 821 levels, and the full run takes minutes instead of about an hour.
  `--all-transitions` computes everything, and `--e1-only` (LCE2 = −2) skips the
  forbidden lines.

## Labels: making ssout consistent with NIST

SUPERSTRUCTURE names each level after its largest LS component. For strongly
mixed pairs with the same configuration, parity and J, that name can flip while
the energies are right. In Ni XV the 3s3p³ level at 4.626 Ry comes out as ¹P°₁
and the one at 4.873 Ry as ³S°₁, while NIST has ³S°₁ at 4.356 and ¹P°₁ at 4.640,
and the E1 decays of the two levels follow the NIST names.
The optimiser pairs levels by J and energy, so it is not affected, but anything
that reads ssout by label is.

`python -m ssopt.export "<ion>" --relabel-ssout` (run automatically at the end
of `full.slurm`) rewrites the fine-structure level table in `ssout.full` so that
a level takes the NIST label only when the evidence is clear:

* For a mixed pair, the E1 decays decide first: the label that makes the
  level's strongest decays obey the LS selection rules (ΔS = 0, |ΔL| ≤ 1) is
  kept.
* Otherwise the energies decide, and a swap is accepted only if keeping SS's
  labels would put the levels more than 25% off and at least three times worse
  than the swap.
* Levels whose NIST term is flagged uncertain (`?`), levels without a NIST
  partner, and levels more than 10% away from their NIST partner in energy keep
  SS's label, since a level that far off may not be the same state.

The original file, exactly as SUPERSTRUCTURE wrote it, is kept as
`ssout.full.ss`. Energies, eigenvectors and the radiative tables are untouched,
and E1 rows refer to levels by index, so they stay consistent. `python swaps.py
work` lists every relabelled level.

`levels_orbital.txt` and `avalues.txt` are written from the full run with the
same labels, and they pair each NIST level with the computed level of the same
configuration, J and term when one lies within 10% of the NIST energy. Only the
levels left over are paired by nearest energy. Pairing by energy alone can pick
the wrong level when NIST lists only one of a mixed group: in Ni XV, NIST gives
only 3s²3p4f ¹G₄ among the J = 4 levels of that configuration, and the nearest
computed level is ³F₄, whose decays are those of a triplet.

## Lifetimes against experiment

Laboratories measure the lifetime τ = 1/ΣA of a level, so the experimental test
of the A-values is a test on lifetimes. `python -m ssopt.lifetimes "Fe XVII"
--workdir work` sums every rate in the radiative section (E1, E2, M1, M2, E3)
and lists τ per level with the channel breakdown. `--bench
benchmarks/lifetimes_exp.csv` compares with 139 measured lifetimes for 22 ions
(storage rings, EBITs, ion traps, beam-foil and laser experiments), each with
the URL it was read from.

## What the tuned λ mean

About half of all runs land within 25% of the best RMS, so the error is flat
along most directions in λ. Among near-optimal runs λ(1s) wanders over ±0.3
with no effect, and only λ(2p) is well pinned down, followed by λ(2s) and
λ(3p). The tuned value of most λ is therefore a property of where the
optimiser stopped rather than of the ion. `ssopt/predict.py` fits λ across
ions, which is useful as a starting point (`--init-lambdas`) but not as a
replacement for tuning.

## Other options and tools

* `--doubles paired` adds the double excitations into higher shells (4s² to
  4f²). On 87 test ions leaving them out changed the median RMS from 1.02% to
  1.07% and made the runs 4 to 50 times faster.
* `python -m ssopt.cli prune "<ion>" --mode orbital --lambdas best` removes
  each correlation group in turn at the tuned λ, to see which ones move the
  energies. It is a diagnostic and is not part of the standard job.
* `python -m ssopt.dataset --workdir work --out dataset` writes CSV files of
  ions, levels, samples and A-values, and `python -m ssopt.figures` plots them.

Per ion, `work/<tag>/` also holds `deck.json` (the configuration list) and
`history_l.json`, `history_orbital.json` (every λ tried, with every matched
level energy).

## SUPERSTRUCTURE conventions learned the hard way

* `MSCAL = n_orbitals` means λ grouped by l (the input values are λ_s, λ_p,
  λ_d, λ_f and the rest are ignored), while `MSCAL = 1000 + n` gives one λ per
  orbital. The `SCALE FACTORS` lines in ssout show what was applied.
* Columns 1 to 8 are read from every configuration card, and some switches take
  the values from the terminator card: columns 1 to 2 hold the restart switch
  (MSTRT0) and columns 5 to 8 the radiative switches LCE1 and LCE2. Columns 5
  to 8 of the first card declare the closed core.
* Free-format λ (`LFREE='Y'`) accepts four decimals.
* The J-level table has no parity column, so parity comes from the
  configuration `CF`. The term table has signed S, with negative meaning odd.
* `MAXTR` in a given build may be too small for the `fort.7` punch of a large
  deck. The punch is then incomplete, but ssout is not; this matters only for
  R-matrix work downstream.
* `'VEL.GF'` in the radiative table is not the velocity gf: the stored
  velocity/length ratio sits near 0.25 in this build, so the gauge test uses
  four times the stored value.

## Repository layout

```
ssopt/       the Python package (deck writer, runner, matching, surrogate, export, ...)
slurm/       ion.slurm (tuning), full.slurm (full run + tables), sweep.sh (a list of ions)
docs/        this page
tests/       python -m pytest tests/   (some tests need data/nist_Si_I_levels_ry.csv: run fetch "Si I" first)
ref/         a reference deck and reference output used by the tests
benchmarks/  lifetimes_exp.csv, measured lifetimes
swaps.py     lists relabelled levels with their NIST partners
```
