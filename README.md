# ssopt: automated λ tuning for SUPERSTRUCTURE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23380595.svg)](https://doi.org/10.5281/zenodo.23380595)

`ssopt` tunes the scaling parameters λ of the atomic structure code
SUPERSTRUCTURE against the observed energy levels in NIST, and then checks the
results against data that is not used in the tuning: NIST A-values, measured
lifetimes, NORAD and other calculations. You give it the name of an ion, and it
builds the configuration list, tunes λ, runs the full calculation with radiative
data, and writes level and transition tables in the NORAD formats. Nothing is
set by hand for each ion.

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

### Step 4: tell Slurm your project code (once)

The commands in step 5 submit jobs with `sbatch`. This is the same whether you
log in with `ssh` or use a terminal in a web portal such as OSC OnDemand. Many
clusters, including OSC, need each job to say which project its hours are
charged to. Save your project code once so that every job uses it:

```bash
echo 'export SBATCH_ACCOUNT=<your project code>' >> ~/.bashrc    # on OSC this looks like PAS1234
source ~/.bashrc
```

If your cluster loads Python as a module, also save the module name, so that
each job loads it too. Use the full name with the version, as listed by
`module avail python` (on OSC Cardinal this is `python/3.12`):

```bash
echo 'export SSOPT_MODULES="python/3.12"' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: run one ion

Ions are named as in NIST, for example `"Ni XV"` (Ni¹⁴⁺), `"Fe XVII"` or
`"O I"`. Running one ion takes three commands, shown here for Ni XV.

1. Download the NIST data for the ion. This needs internet, so run it on the
   login node:

   ```bash
   python -m ssopt.cli fetch "Ni XV"
   ```

2. Tune λ. This submits a job that takes a few minutes. Pick one of these two
   commands.

   Inner shells open (the default):

   ```bash
   sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6
   ```

   Inner shells closed (faster):

   ```bash
   sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6 --closed-core
   ```

   Here `4` is the highest n of the orbitals in the deck, and
   `--core-bounds 0.8,1.6` is the range λ may take for the inner orbitals.
   `--closed-core` declares the full inner shells (for Ni XV, 1s² and 2s²2p⁶)
   closed, which makes every run 2 to 3 times faster. It also changes the
   Breit-Pauli energies. On our test ions it helped the highly charged ions and
   hurt some low-charge Si- and Al-like ions, so for the ions you care about it
   is worth trying both.

3. When the tuning job has finished (`squeue -u $USER` no longer lists it), run
   the full calculation with the A-values and the NORAD tables. Use the same
   choice as in step 2.

   Inner shells open:

   ```bash
   sbatch slurm/full.slurm "Ni XV" orbital best j
   ```

   Inner shells closed:

   ```bash
   sbatch slurm/full.slurm "Ni XV" orbital best j --closed-core
   ```

   Here `orbital best j` means: use one λ per orbital, take the best λ found by
   the tuning, and pair the levels with NIST by J.

The results are in `work/ni15/` (see below), and the job logs are
`ssopt_ion_<job id>.log` and `ssopt_full_<job id>.log`.

To try both and compare, give the second run its own folder by putting
`WORKDIR=work_closed` in front of both of its `sbatch` commands, for example
`WORKDIR=work_closed sbatch slurm/ion.slurm "Ni XV" 4 --core-bounds 0.8,1.6 --closed-core`.
Its results are then in `work_closed/ni15/`.

For a larger deck, add `--doubles paired` to both commands. It also includes the
double excitations into the n = 4 shell (4s² to 4f²). On 87 test ions this
lowered the median energy error only from 1.07% to 1.02%, while the runs took 4
to 50 times longer.

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

The tuning runs in two steps. It first tunes one λ per orbital angular momentum
(s, p, d, f), which quickly finds a good region, and then one λ per orbital,
starting from that result. Files ending in `_l` (`ssin.tuned_l`, `levels_l.txt`)
are from the first step and are kept only as a record. The final results are
the `_orbital` files and everything made from them:

| File | What it is |
|---|---|
| `levels_orbital.txt` | every computed level next to its NIST value (paired by configuration, J and term), and the relative RMS error. The last line lists the uncertain identifications: strongly mixed levels for which pairing by term and by energy give different computed levels |
| `ssin.tuned_orbital` | the SUPERSTRUCTURE input with the tuned λ; `./structx < ssin.tuned_orbital > ssout` runs it again |
| `ssout.full` | the full SUPERSTRUCTURE output at the tuned λ, with level labels matched to NIST where SUPERSTRUCTURE's names are clearly swapped |
| `ssout.full.ss` | the same output exactly as SUPERSTRUCTURE wrote it, with its original labels |
| `report.txt`, `avalues.txt` | every E1 line between the paired levels, with its A-value next to NIST's (every line NIST lists, however weak) and NORAD's where available. Use this file for A-value tables rather than pairing levels by hand. The summary numbers at the top leave out lines to uncertain levels. The L/V column compares the length and velocity forms of each line: they agree for an exact wavefunction, so a value far from 1 marks a less reliable A-value |

Each run writes one line to `log.txt` that records the ssopt version, checksums
of `structx` and of the NIST file, and the exact command, and the same line heads
`levels_orbital.txt`, `avalues.txt` and `report.txt` together with the λ used.
Two runs that disagree can be compared from these lines alone (the same version,
`structx`, NIST file and options always give the same λ).

The level and transition tables in the NORAD formats are in the folder
`norad_ss/` (for Ni XV, `norad_ss/ni15.*.txt`).
As in NORAD, only bound levels are listed. SUPERSTRUCTURE does not give the
parent (core) level, so the core term is filled in only where it is unique and
the core J is left blank.

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
2. **Matching.** During the tuning, each computed level is paired with a NIST
   level of the same configuration, parity and J by nearest energy, because
   SUPERSTRUCTURE's term names can change from one λ to the next. Only these
   pairs judge the fit, but every level is still computed and kept. The final
   tables (`levels_orbital.txt`, `avalues.txt`) pair by configuration, J and
   term first, after the labels have been checked against NIST (step 6), and
   use the nearest energy only for the levels left over.
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

**What is fitted and what is checked.** Only the λ are tuned, and only to the
NIST level energies. The code does not shift the computed energies toward the
observed ones (as term energy corrections or adjusted diagonal energies do), so
every energy, A-value, lifetime and mixing coefficient comes straight from
SUPERSTRUCTURE at the tuned λ. The A-values and lifetimes are then compared with
NIST, NORAD, measured lifetimes and other calculations. None of these
comparisons feed back into the tuning, so they are an independent check.

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

Other atomic codes have also been tuned against observed levels with a learned
model. Mendez et al. (2020) tune the atomic structure of neutral Be with a
Bayesian method (a Tree-structured Parzen Estimator), for use in electron-impact
collision calculations. Mendez (2021, PhD thesis) extends this with
Gaussian-process Bayesian optimisation, and Mendez, Abdala and Mitnik (2026)
apply it to the scaling parameters of AUTOSTRUCTURE for Be and Mg.
Ferreira da Silva et al. (2025) use Bayesian optimisation to tune the Flexible
Atomic Code for lanthanide and actinide ions, and Ma et al. (2024) use
machine-learning models to correct the parameters of the Cowan code for
neutral ytterbium. Bilous, Cheung and Safronova (2024) use a neural network to
choose the configurations for large CI calculations. AUTOSTRUCTURE can also
adjust its λ internally (Badnell 2011). It does this by minimising an energy
functional, so the result is variational rather than a fit to observation.
In SUPERSTRUCTURE itself, the λ have been adjusted by hand to fit observed
levels since Nussbaumer and Storey (1978).

For a single ion, large calculations are the benchmark. For the Si-like ions,
Jönsson et al. (2016) and Zhang et al. (2021) use MCDHF and RCI calculations
(GRASP), and Gupta and Msezane (2012) use CIV3 with the diagonal energies
adjusted to the NIST values. `ssopt` makes no such adjustment, and these papers
are used only to check its Ni XV energies, A-values and lifetimes.

The surrogate in `ssopt` models every level energy and computes the error from
the predicted energies. In machine learning this is known as target-vector
Bayesian optimisation (Uhrenholt & Jensen 2019; Astudillo & Frazier 2019;
Plock et al. 2022), and `ssopt` applies it to atomic structure. The labelling
of strongly mixed levels is a known problem in other codes as well, and
Gaigalas et al. (2017) treat it for GRASP.

`ssopt` differs from the atomic work above in four ways. It tunes
SUPERSTRUCTURE, and it builds the configuration list and the pairing with NIST
levels from the NIST list itself, with no input per ion. It has also been run
on 119 ions across whole isoelectronic sequences, and it checks the results
against NIST A-values, measured lifetimes and independent calculations, none of
which are used in the tuning.

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

Copyright 2026 Divya Chari, Alain Ward, Sultana N. Nahar and Anil K. Pradhan,
The Ohio State University. The method and the use of SUPERSTRUCTURE follow the
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
* A. M. P. Mendez, J. I. Di Filippo, S. D. López, D. M. Mitnik, Bayesian
  atomic structure calculations for collisional problems, J. Phys.: Conf. Ser.
  1412, 132027 (2020), https://doi.org/10.1088/1742-6596/1412/13/132027
* M. P. A. Mendez, Optimización de blancos atómicos y moleculares en procesos
  colisionales, PhD thesis, Universidad de Buenos Aires (2021),
  https://bibliotecadigital.exactas.uba.ar/download/tesis/tesis_n6803_Mendez.pdf
* A. Mendez, M. Abdala, D. M. Mitnik, Bayesian optimization of atomic
  structure for collisional calculations, Proceedings of the 1st International
  Online Conference on Atoms (MDPI, 2026), https://sciforum.net/paper/28531
* P. Bilous, C. Cheung, M. Safronova, A neural network approach to running
  high-precision atomic computations, Phys. Rev. A 110 (2024),
  https://arxiv.org/abs/2408.00477
* A. K. Uhrenholt, B. S. Jensen, Efficient Bayesian optimization for target
  vector estimation, Proc. AISTATS 2019, PMLR 89, 2661 (2019),
  https://proceedings.mlr.press/v89/uhrenholt19a.html
* R. Astudillo, P. I. Frazier, Bayesian optimization of composite functions,
  Proc. ICML 2019, PMLR 97, 354 (2019),
  https://proceedings.mlr.press/v97/astudillo19a.html
* M. Plock, K. Andrle, S. Burger, P.-I. Schneider, Bayesian target-vector
  optimization for efficient parameter reconstruction, Adv. Theory Simul. 5,
  2200112 (2022), https://doi.org/10.1002/adts.202200112
* G. Gaigalas, C. Froese Fischer, P. Rynkun, P. Jönsson, JJ2LSJ transformation
  and unique labeling for energy levels, Atoms 5, 6 (2017),
  https://doi.org/10.3390/atoms5010006
* N. R. Badnell, A Breit-Pauli distorted wave implementation for AUTOSTRUCTURE,
  Comput. Phys. Commun. 182, 1528 (2011)

Comparison data (Ni XV)

* G. P. Gupta, A. Z. Msezane, Fine-structure energy levels, radiative rates and
  lifetimes in Si-like nickel, Phys. Scr. 86, 015303 (2012),
  https://doi.org/10.1088/0031-8949/86/01/015303
* P. Jönsson, L. Radžiūtė, G. Gaigalas, M. R. Godefroid, J. P. Marques,
  T. Brage, C. Froese Fischer, I. P. Grant, Accurate multiconfiguration
  calculations of energy levels, lifetimes, and transition rates for the silicon
  isoelectronic sequence, Astron. Astrophys. 585, A26 (2016),
  https://doi.org/10.1051/0004-6361/201527106
* X. H. Zhang, G. Del Zanna, K. Wang et al., Benchmarking multiconfiguration
  Dirac-Hartree-Fock calculations for astrophysics: Si-like ions from Cr XI to
  Zn XVII, Astrophys. J. Suppl. Ser. 257, 56 (2021),
  https://doi.org/10.3847/1538-4365/ac2a3f
