# ssopt: automated λ tuning for SUPERSTRUCTURE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23380595.svg)](https://doi.org/10.5281/zenodo.23380595)

`ssopt` tunes the scaling parameters λ of the atomic structure code
SUPERSTRUCTURE so that its energy levels agree as closely as possible with NIST.
You give it the name of an ion, and it builds the configuration list, tunes λ,
runs the full calculation with radiative data, and writes level and transition
tables in the NORAD formats. Nothing is set by hand for each ion.

* [How to use it](#how-to-use-it)
* [The science](#the-science)
* [Citing, license and references](#citing)

---

## How to use it

### What you need

1. **`structx`, the compiled SUPERSTRUCTURE program.** It is not part of this
   repository. Get the Fortran source `struct.f` from the code's authors (for
   example through Prof. Sultana Nahar's group at Ohio State or the Iron
   Project) and compile it:

   ```bash
   gfortran -O2 -std=legacy -ffixed-line-length-72 struct.f -o structx
   ```

   The Intel compiler (`ifort -O2 struct.f -o structx`) gives the same results.
   Put `structx` in the top folder of this repository, or link to it with
   `ln -s /path/to/structx structx`. If a very large deck stops with a message
   such as `INCREASE MAXUC` or `MAXTR .GE. ... IS REQUIRED`, raise the named
   array sizes in `struct.f` and compile again.
2. **Python 3.10 or newer** with numpy and scipy.
3. **A cluster that uses Slurm**, for the scripts in `slurm/`. They contain no
   account or cluster names, so set these once in your shell (for example in
   `~/.bashrc`) if your cluster needs them:

   ```bash
   export SBATCH_ACCOUNT=<your allocation>
   export SSOPT_MODULES="python"        # modules to load inside each job
   ```

   The scripts are short wrappers around `python -m ssopt.cli`, so the same
   commands also run on a workstation.

### Run one ion

```bash
python -m ssopt.cli fetch "Ni XV"                                  # NIST data -> data/ (needs internet, once per ion)
sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6             # tune λ (48 cores, 1 to 7 minutes)
sbatch slurm/full.slurm "Ni XV" orbital best j --core-bounds 0.8,1.6   # full run at the tuned λ, then the NORAD tables
```

Start the full run after the tuning job has finished, or chain the two with
`--dependency=afterok:<tuning job id>`. The inner shells (1s², 2s²2p⁶) are left
open in this default run, which matches NIST best for most low-charge ions.

### Run one ion with the inner shells closed

Add `--closed-core` to both commands:

```bash
sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6 --closed-core
sbatch slurm/full.slurm "Ni XV" orbital best j --core-bounds 0.8,1.6 --closed-core
```

The full inner shells are then declared closed on the first card of the deck,
which makes every run 2 to 3 times faster. It also changes the Breit-Pauli
energies, because SUPERSTRUCTURE treats a closed core differently. On our test
ions this helped the highly charged ions and hurt some low-charge Si- and
Al-like ions, so compare both for the ions you care about.

### Run a list of ions

```bash
bash slurm/sweep.sh my_ions.txt --core-bounds 0.8,1.6               # one ion name per line, e.g. "Ni XV"
bash slurm/sweep.sh my_ions.txt --core-bounds 0.8,1.6 --closed-core # the same with the inner shells closed
```

`sweep.sh` fetches the NIST data, submits one tuning job per ion and chains
each full run to its tuning job. Ions that are already tuned are skipped. To
keep a set of runs in its own folder, put `WORKDIR=<folder>` in front, for
example `WORKDIR=run_closed bash slurm/sweep.sh ...`.

### Where the results are

Everything for one ion goes into `work/<ion>/` (for Ni XV, `work/ni15/`):

| File | What it is |
|---|---|
| `ssin.tuned_orbital` | the deck with the tuned λ; `./structx < ssin.tuned_orbital > ssout` reruns it |
| `levels_orbital.txt` | every computed level next to its NIST value, and the relative RMS error |
| `ssout.full` | the full SUPERSTRUCTURE output at the tuned λ, with level labels matched to NIST |
| `report.txt`, `avalues.txt` | the A-values compared with NIST (and NORAD where available) |
| `norad_ss/<ion>.*.txt` | level and transition tables in the NORAD formats |

### Useful options

| Option | What it does |
|---|---|
| `--closed-core` | close the full inner shells (faster, changes the Breit-Pauli energies) |
| `--doubles paired` | also include double excitations into higher shells (4s² to 4f²); slower, the old default |
| `--all-transitions` | full run: compute transitions between all levels, not only NIST-type levels (much slower) |
| `--e1-only` | full run: skip the forbidden E2/M1 lines (about twice as fast) |
| `--n-max 5` | allow orbitals up to n = 5 (default 4) |

---

## The science

### What λ is

SUPERSTRUCTURE builds each orbital in a Thomas-Fermi-Dirac-Amaldi model
potential, and one scaling parameter λ per orbital sets the size of that
potential. Users normally adjust the λ by hand until the energies agree with
observation, which takes experience and a lot of trial runs and is hard to
reproduce. `ssopt` does this search automatically.

### How it works

1. **Configurations.** The list starts from every configuration that NIST
   lists within the active orbitals (n ≤ 4, l ≤ 3 by default). It then adds the
   single excitations from the ground configuration and the double excitations
   within its outermost shell (for 3s²3p², for example, 3p⁴ and 3p²3d²). These
   extra configurations improve the energies through configuration
   interaction. Double excitations into higher shells (4s² to 4f²) are left out
   because they barely mix with the low levels but make the deck several times
   larger.
2. **Matching.** Each computed level is paired with a NIST level of the same
   configuration, parity and J, by nearest energy. Only these pairs judge the
   fit, but every level is still computed and kept.
3. **Tuning.** The code looks for the λ with the smallest relative RMS error.
   It first tunes one λ per orbital angular momentum (s, p, d, f) and then one
   λ per orbital. Rather than trying λ at random, it fits a model of how every
   level energy changes with λ (a linear trend plus a Gaussian process),
   predicts where the error is smallest, runs SUPERSTRUCTURE there and repeats.
4. **Speed.** The angular algebra of a deck does not depend on λ, so it is
   computed once per stage and read back by every later run, and only the
   radial part is repeated. The energies are identical to a run from scratch.
5. **Full run.** At the tuned λ the code computes the allowed (E1) and
   forbidden (E2, M1) transitions between the levels whose leading
   configuration is a NIST one. Levels of the correlation configurations stay
   in the level table but get no transitions, which keeps the run short.
6. **Labels and tables.** SUPERSTRUCTURE names a level after its largest LS
   component, which can disagree with NIST for strongly mixed levels. Labels are
   changed to NIST's only when the energies and the E1 decays make it clear,
   and then the NORAD-format tables are written.

### How accurate and how fast

On a test set of 86 ions (the N- to P-like sequences up to Z = 28), the median
RMS error of the matched levels is about 1%, and 74 of the 86 ions are within
3%. The Al-, Si- and P-like ions are harder, at 1 to 4%, because an n ≤ 4 deck
does not capture all of their correlation (Ni XV comes out at 3%). The method
works best for ions with a charge of 3 or more, while neutral and near-neutral
atoms have the largest outliers (Mg I is off by 24%).

On one 48-core node, tuning an ion takes 1 to 7 minutes and the full run takes
a few seconds to 6 minutes. Each ion is its own job, so 119 ions, tuned and
with full runs, finished in about 15 minutes.

The tuned λ should be read as fitting parameters rather than physical
quantities, since the error barely changes along most directions in λ.

More detail on the method, the label rule, the lifetime comparison with
experiment and SUPERSTRUCTURE input conventions is in
[`docs/details.md`](docs/details.md).

---

## Citing

If you use this code or its results, please cite the software until the
method paper is out (GitHub's "Cite this repository" button gives the
reference):

> D. Chari, S. N. Nahar and A. K. Pradhan, *ssopt: automated tuning of
> Thomas-Fermi scaling parameters for SUPERSTRUCTURE* (2026), Zenodo,
> https://doi.org/10.5281/zenodo.23380595

This DOI always points to the latest version, and each release on Zenodo also
has its own DOI. Please also cite SUPERSTRUCTURE (Eissner, Jones & Nussbaumer
1974; Nussbaumer & Storey 1978; Nahar, Eissner, Chen & Pradhan 2003 for the
forbidden transitions) and, if you use the NORAD comparisons or formats, NORAD
(Nahar 2020; Nahar & Hinojosa-Aguirre 2024).

Contributors: Alain Ward (Department of Physics, The Ohio State University).

## License

Copyright 2026 Divya Chari, Sultana N. Nahar and Anil K. Pradhan, The Ohio
State University. The method and the use of SUPERSTRUCTURE follow the
atomic-structure work of Profs. Nahar and Pradhan's group. The code is released
under the Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0, see `LICENSE`), the same terms as NORAD-Atomic-Data. For
commercial use, contact the authors.

## References

SUPERSTRUCTURE

* W. Eissner, M. Jones, H. Nussbaumer, Techniques for the calculation of atomic
  structures and radiative data including relativistic corrections, Comput.
  Phys. Commun. 8, 270 (1974).
* H. Nussbaumer, P. J. Storey, The C III transition probabilities, Astron.
  Astrophys. 64, 139 (1978).
* S. N. Nahar, W. Eissner, G. X. Chen, A. K. Pradhan, Atomic data from the Iron
  Project LIII. Relativistic allowed and forbidden transition probabilities for
  Fe XVII, Astron. Astrophys. 408, 789 (2003). First publication of the version
  with the corrected forbidden transitions (E2, M1, M2, E3), which is the one
  `ssopt` reads.

Data

* NIST Atomic Spectra Database, A. Kramida, Yu. Ralchenko, J. Reader and NIST
  ASD Team, https://physics.nist.gov/asd
* S. N. Nahar, NORAD-Atomic-Data: a database for spectroscopy and many other
  applications, Atoms 8, 68 (2020); S. N. Nahar, G. Hinojosa-Aguirre, Atoms 12,
  22 (2024); https://norad.astronomy.osu.edu

Related methods

* R. Ferreira da Silva et al., Systematic Bayesian Optimization for Atomic
  Structure Calculations of Heavy Elements, arXiv:2502.13250 (2025).
* K. Ma et al., Machine Learning-Assisted Hartree-Fock Approach for Energy
  Level Calculations in the Neutral Ytterbium Atom, Entropy 26, 962 (2024).
* N. R. Badnell, A Breit-Pauli distorted wave implementation for AUTOSTRUCTURE,
  Comput. Phys. Commun. 182, 1528 (2011).
