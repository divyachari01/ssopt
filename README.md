# ssopt: automated λ tuning for SUPERSTRUCTURE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23380595.svg)](https://doi.org/10.5281/zenodo.23380595)

`ssopt` tunes the scaling parameters λ of the atomic structure code
SUPERSTRUCTURE so that its energy levels agree as closely as possible with NIST.
You give it the name of an ion, and it builds the configuration list, tunes λ,
runs the full calculation with radiative data, and writes level and transition
tables in the NORAD formats. Nothing is set by hand for each ion.

* [How to use it](#how-to-use-it)
* [The science](#the-science)
* [Related work](#related-work)
* [Citing, license and references](#citing)

---

## How to use it

All commands below are run from inside the `ssopt` folder.

### Step 1: get the code

```bash
git clone https://github.com/divyachari01/ssopt.git
cd ssopt
```

(Or download the ZIP from the GitHub page and unzip it.)

### Step 2: put `structx` in the same folder

`structx` is the SUPERSTRUCTURE program itself. It is not included here, but
the Fortran source `struct.f` can be downloaded from the programs section of
[Prof. Sultana Nahar's teaching page](https://www.astronomy.ohio-state.edu/nahar.1/teaching.html#program),
which also has her notes on running the code. Compile it with:

```bash
gfortran -O2 -std=legacy -ffixed-line-length-72 struct.f -o structx
```

Then put `structx` **inside the `ssopt` folder**, next to this README. If it
already sits somewhere else, a link works too:

```bash
ln -s /path/to/structx structx
```

Your folder should now look like this:

```
ssopt/
    README.md
    structx        <- the SUPERSTRUCTURE program, in the same folder
    ssopt/         <- the Python code
    slurm/         <- the job scripts
    ...
```

If a very large calculation later stops with a message such as
`INCREASE MAXUC`, raise that number in `struct.f` and compile again.

### Step 3: check Python

You need Python 3.10 or newer with numpy and scipy. Check with:

```bash
python -c "import numpy, scipy; print('ok')"
```

If it does not print `ok`, load your cluster's Python module (often
`module load python`) or install the two packages with
`pip install --user numpy scipy`.

### Step 4: set up your cluster account (once)

The job scripts in `slurm/` are submitted to the cluster with `sbatch`. Most
clusters need to know which project to charge the jobs to, and which module
gives you Python. Add these two lines to the end of your `~/.bashrc`, with your
own values, and then run `source ~/.bashrc` (or log in again):

```bash
export SBATCH_ACCOUNT=<your project code>   # on OSC this looks like PAS1234
export SSOPT_MODULES="python"               # the module that loads Python with numpy and scipy
```

If your cluster does not use project codes, leave out the first line. If Python
already works without loading a module, leave out the second.

### Step 5: run one ion

Ions are named as in NIST, for example `"Ni XV"` (Ni¹⁴⁺), `"Fe XVII"` or
`"O I"`. Running one ion takes three commands, shown here for Ni XV.

1. Download the NIST data for the ion. This needs internet, so run it on the
   login node:

   ```bash
   python -m ssopt.cli fetch "Ni XV"
   ```

2. Tune λ. This submits a job that takes a few minutes:

   ```bash
   sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6
   ```

   Here `4` is the highest n of the orbitals in the deck, and
   `--core-bounds 0.8,1.6` is the range λ may take for the inner orbitals.

3. When the tuning job has finished (`squeue -u $USER` no longer lists it), run
   the full calculation with the A-values and the NORAD tables:

   ```bash
   sbatch slurm/full.slurm "Ni XV" orbital best j
   ```

   Here `orbital best j` means: use one λ per orbital, take the best λ found by
   the tuning, and pair the levels with NIST by J.

The inner shells (1s² and 2s²2p⁶) are left open by default. The results are in
`work/ni15/` (see below), and the job logs are `ssopt_ion_<job id>.log` and
`ssopt_full_<job id>.log`.

### Optional: close the inner shells

Add `--closed-core` to the tuning command and to the full-run command:

```bash
sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6 --closed-core
sbatch slurm/full.slurm "Ni XV" orbital best j --closed-core
```

The full inner shells are then declared closed in the deck, which makes every
run 2 to 3 times faster. It also changes the Breit-Pauli energies, because
SUPERSTRUCTURE treats a closed core differently. On our test ions this helped
the highly charged ions and hurt some low-charge Si- and Al-like ions, so
compare both for the ions you care about.

### Optional: run many ions at once

List the ions in a text file, one per line, and run:

```bash
bash slurm/sweep.sh my_ions.txt --core-bounds 0.8,1.6                 # inner shells open
bash slurm/sweep.sh my_ions.txt --core-bounds 0.8,1.6 --closed-core   # inner shells closed
```

This downloads the NIST data, submits the tuning jobs and starts each full run
when its tuning is done. Ions that are already tuned are
skipped. To keep a set of runs in its own folder, put `WORKDIR=<name>` in front,
for example `WORKDIR=run_closed bash slurm/sweep.sh my_ions.txt ...`.

### Where the results are

The results for one ion are in `work/<ion>/` (for Ni XV, `work/ni15/`):

| File | What it is |
|---|---|
| `levels_orbital.txt` | every computed level next to its NIST value, and the relative RMS error |
| `ssin.tuned_orbital` | the SUPERSTRUCTURE input with the tuned λ; `./structx < ssin.tuned_orbital > ssout` runs it again |
| `ssout.full` | the full SUPERSTRUCTURE output at the tuned λ, with level labels matched to NIST where SUPERSTRUCTURE's names are clearly swapped |
| `ssout.full.ss` | the same output exactly as SUPERSTRUCTURE wrote it, with its original labels |
| `report.txt`, `avalues.txt` | the A-values compared with NIST (and NORAD where available) |

The level and transition tables in the NORAD formats are in the folder
`norad_ss/` (for Ni XV, `norad_ss/ni15.*.txt`).

### Useful options

| Option | What it does |
|---|---|
| `--closed-core` | close the full inner shells (faster, changes the Breit-Pauli energies) |
| `--doubles paired` | also include double excitations into higher shells (4s² to 4f²); slower |
| `--all-transitions` | full run: transitions between all levels, not only NIST-type levels (much slower) |
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

## Related work

Other atomic codes have also been tuned against NIST with a learned model.
Ferreira da Silva et al. (2025) use Bayesian optimisation to tune the Flexible
Atomic Code for lanthanide and actinide ions. Ma et al. (2024) use
machine-learning models to correct the parameters of the Cowan code for
neutral ytterbium. AUTOSTRUCTURE can also adjust its λ
internally (Badnell 2011). It does this by minimising an energy functional, so
the result is variational rather than a fit to observation.

`ssopt` differs from these in four ways. It works on SUPERSTRUCTURE's λ for
each orbital, and it models every level energy rather than only the overall
error. It also builds the configuration list and the pairing with NIST levels
from the NIST list itself, with no input per ion. Finally, it tests the
results against measured lifetimes.

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

* R. Ferreira da Silva, A. Flörs, L. Leitão, J. P. Marques, J. M. Sampaio,
  G. Martínez-Pinedo, Systematic Bayesian Optimization for Atomic Structure
  Calculations of Heavy Elements, arXiv:2502.13250 (2025),
  https://arxiv.org/abs/2502.13250
* K. Ma, C. Yang, J. Zhang, Y. Li, G. Jiang, J. Chai, Machine Learning-Assisted
  Hartree-Fock Approach for Energy Level Calculations in the Neutral Ytterbium
  Atom, Entropy 26, 962 (2024), https://doi.org/10.3390/e26110962
* N. R. Badnell, A Breit-Pauli distorted wave implementation for AUTOSTRUCTURE,
  Comput. Phys. Commun. 182, 1528 (2011)
