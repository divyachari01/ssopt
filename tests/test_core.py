"""Unit tests that need no structx: deck bytes, parsers, matcher, surrogate."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # tiny matrices: many BLAS threads only fight each other on a busy login node
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from ssopt import nist, ssout as so, match as M  # noqa: E402
from ssopt.configs import build_deck, generate_ci_configs  # noqa: E402
from ssopt.deck import read_ssin, write_ssin, configs_equal, n_lambda  # noqa: E402
from ssopt.ions import build_ion, config_key, parse_config_str, parse_spectrum  # noqa: E402
from ssopt.surrogate import EnergySurrogate, History, Sample  # noqa: E402

REF_SSIN = os.path.join(ROOT, "ref", "ssin.cardinal")
NIST_SI = os.path.join(HERE, "fixtures", "nist_Si_I_levels_ry.csv")   # NIST Si I levels with n <= 4, l <= 3
REF_SSOUT = os.path.join(ROOT, "ref", "ssout.si1.levels")   # trimmed reference output


def test_spectrum_parsing():
    assert parse_spectrum("Si I") == ("Si", 0)
    assert parse_spectrum("Fe XVII") == ("Fe", 16)
    assert parse_spectrum("Ni 15") == ("Ni", 14)
    assert parse_spectrum("ar17") == ("Ar", 16)


def test_config_string_parsing():
    assert parse_config_str("3s2.3p2") == {(3, 0): 2, (3, 1): 2}
    assert parse_config_str("3s2.3p.(2P*<1/2>).4f") == {(3, 0): 2, (3, 1): 1, (4, 3): 1}
    assert parse_config_str("3s.3p2.(4P).4s") == {(3, 0): 1, (3, 1): 2, (4, 0): 1}
    assert parse_config_str("3s2.3p.nd") == {(3, 0): 2, (3, 1): 1}


def test_deck_roundtrip_is_byte_identical():
    ref = open(REF_SSIN).read()
    d = read_ssin(ref)
    assert d["nzion"] == 14 and d["mscal"] == 10 and d["mode"] == "l"
    assert len(d["configs"]) == 28
    assert d["lambdas"] == [1.1, 1.0, 1.0, 1.0]
    out = write_ssin(d["configs"], 14, d["lambdas"], mode="l", lam_decimals=1)
    # configuration cards: one configuration per card now (structx reads them as a
    # stream, energies identical); everything from the terminator card on is unchanged
    assert configs_equal(read_ssin(out)["configs"], d["configs"])
    tail = lambda t: t[t.index("\n 0  "):]
    assert tail(out) == tail(ref)
    d2 = read_ssin(write_ssin(d["configs"], 14, [1.05] * 10, mode="orbital"))
    assert d2["mode"] == "orbital" and d2["mscal"] == 1010 and len(d2["lambdas"]) == 10
    assert configs_equal(d2["configs"], d["configs"])


def test_ion_and_generator():
    ion = build_ion("Si I")
    assert ion.core == {(1, 0): 2, (2, 0): 2, (2, 1): 6}
    assert ion.valence == {(3, 0): 2, (3, 1): 2}
    g = generate_ci_configs(ion, 4, 3, 2, "paired")      # the reference deck has the 4l^2 doubles
    ref = read_ssin(open(REF_SSIN).read())["configs"]
    assert {config_key(c) for c in g} == {config_key(c) for c in ref}
    assert n_lambda(g, "l") == 4 and n_lambda(g, "orbital") == 10
    h = build_ion("Si XIV")
    assert h.n_electrons == 1 and h.core == {}


def test_nist_parse():
    lv = nist.parse_levels(open(NIST_SI).read())
    assert len(lv) > 400
    assert lv[0].E == 0.0 and lv[0].term == "3P"
    assert abs(lv[3].E - 0.05739934) < 1e-8      # 1D2
    assert lv[5].parity == -1 and lv[5].twoSp1 == 5  # 5S*


def test_nist_driven_deck_matches_reference():
    ion = build_ion("Si I")
    lv = nist.parse_levels(open(NIST_SI).read())
    plan = build_deck(ion, lv, n_max=4, l_max=3, doubles_mode="paired")   # the reference deck has the 4l^2 doubles
    ref = read_ssin(open(REF_SSIN).read())["configs"]
    assert {config_key(c) for c in plan.configs} == {config_key(c) for c in ref}
    assert plan.tags.count("spec") == 10


def test_ssout_parse_and_match():
    if not os.path.exists(REF_SSOUT):
        return
    text = so.read(REF_SSOUT)
    cf = so.parse_configs(text)
    assert len(cf) == 28
    lv = so.parse_levels(text, cf)
    assert len(lv) == 1150 and lv[0].E == 0.0
    assert lv[5].parity == -1 and lv[5].twoSp1 == 5      # 5S° from 3s3p3
    assert so.parse_scale_factors(text)[:4] == [1.1, 1.1, 1.0, 1.1]
    ion = build_ion("Si I")
    nl = nist.parse_levels(open(NIST_SI).read())
    cfgs = read_ssin(open(REF_SSIN).read())["configs"]
    m = M.match_levels(lv, nl, ion, cfgs, cf, term_first=False)    # the pairing used in tuning
    assert 60 <= len(m) <= 69
    # a swapped pair is only ever within the configurations NIST names in that bucket
    keys = {M.valence_key(c, ion.core) for c in cfgs}
    for mm in m:
        assert M.valence_key(cf[lv[mm.comp_idx].CF], ion.core) in keys
        if mm.how == "cfg":
            assert M.valence_key(nl[mm.nist_idx].cfg, ion.core) == M.valence_key(cf[lv[mm.comp_idx].CF], ion.core)
    strict = M.match_levels(lv, nl, ion, cfgs, cf, strict_config=True, term_first=False)
    assert all(x.how != "swap" for x in strict)
    assert M.rms(m) <= M.rms(strict) + 1e-9
    assert 0.02 < M.rms(m) < 0.10
    # the summary tables pair by configuration, J and term first: same NIST levels, terms kept where possible
    tf = M.match_levels(lv, nl, ion, cfgs, cf)
    assert sorted(x.nist_idx for x in tf) == sorted(x.nist_idx for x in m)
    same = lambda mm: ((lv[mm.comp_idx].twoSp1, lv[mm.comp_idx].L) == (nl[mm.nist_idx].twoSp1, nl[mm.nist_idx].L)
                       and M.valence_key(cf[lv[mm.comp_idx].CF], ion.core) == M.valence_key(nl[mm.nist_idx].cfg, ion.core))
    assert sum(map(same, tf)) > sum(map(same, m))          # 45 vs 38 pairs agree in configuration and term


def test_surrogate_recovers_quadratic_optimum():
    rng = np.random.default_rng(0)
    d, m = 3, 12
    lstar = np.array([0.9, 1.1, 1.3])
    C = rng.normal(0, 0.2, (m, d)); Q = rng.normal(0, 0.05, (m, d))
    E_nist = np.sort(rng.uniform(0, 1, m))
    X = rng.uniform(0.7, 2.0, (40, d))
    hist = History(["a", "b", "c"], [(0.7, 2.0)] * 3, "l")
    for x in X:
        dl = x - lstar
        E = E_nist + C @ dl + Q @ dl ** 2
        hist.samples.append(Sample(list(x), {k: E[k] for k in range(m)}, 0.0, m, 0.0, True))
    sur = EnergySurrogate([(0.7, 2.0)] * 3, list(range(m)), {k: E_nist[k] for k in range(m)}).fit(hist)
    x, f = sur.minimize([np.ones(3), lstar + 0.3])
    assert np.allclose(x, lstar, atol=0.02), x


def test_norad_f_fs_parse_and_nist_match():
    from ssopt import norad as R
    from ssopt.avalues import _norad_level_index
    text = open(os.path.join(HERE, "fixtures", "norad_si1_f_fs_sample.txt")).read()
    tr = R.parse_f_fs(text)
    assert len(tr) == 9
    assert R.ionization_potential(text) == 0.59992
    t = tr[1]                                  # a3Pe(J=2) -> z3Po(J=2)
    assert t.lower == "a3Pe" and t.upper == "z3Po" and t.gi == 5 and t.gf == 5
    assert abs(t.Ei - 0.00203356) < 2e-6 and abs(t.Ef - 0.36409723) < 2e-6
    assert t.lower_qn == (1, 3, 1, 4) and t.upper_qn == (-1, 3, 1, 4)
    assert tr[-1].calc_i and not tr[-1].calc_f
    nl = nist.parse_levels(open(NIST_SI).read())
    idx = _norad_level_index(tr, nl)
    assert len(idx) == 9
    lo, hi = next(k for k, v in idx.items() if v is t)
    assert nl[lo].term == "3P" and nl[lo].J == 2 and nl[hi].term == "3P*" and nl[hi].J == 2


def test_lambda_predictor_recovers_smooth_trend(tmp_path):
    """LOIO prediction on a synthetic smooth λ(Z) table should be far better than the spread."""
    import csv
    from ssopt.predict import load_ions, loio
    from ssopt.ions import build_ion, int_to_roman
    rows = []
    for N in (13, 14):
        for Z in (14, 16, 18, 20, 22, 24, 26, 28):
            if Z < N:
                continue
            ion = build_ion(f"{['Si','S','Ar','Ca','Ti','Cr','Fe','Ni'][(Z-14)//2]} {int_to_roman(Z-N+1)}")
            lam = " ".join(f"{o}={1.0 + 0.01*(Z-N)*(1 if o[-1] in 'sp' else -1):.4f}"
                           for o in ("1s", "2s", "2p", "3s", "3p", "3d"))
            rows.append({"ion": ion.spectroscopic, "tag": ion.tag, "Z": Z, "N": N, "charge": Z - N,
                         "sequence": "Al" if N == 13 else "Si", "rms_orbital": 0.02, "lambdas_orbital": lam})
    with open(tmp_path / "ions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    res = loio(load_ions(str(tmp_path)), "gp")
    assert len(res) == len(rows)
    assert max(r["rms_err"] for r in res) < 0.03      # trend spans 0.14; interpolation error must be small


def test_complex_doubles_stay_in_outer_shell():
    ion = build_ion("Si I")
    full = generate_ci_configs(ion, 4, 3, 2, "paired")
    cx = generate_ci_configs(ion, 4, 3, 2)              # default: complex
    assert len(cx) < len(full)
    ref = {o: n for o, n in ion.valence.items()}
    for c in cx:
        gained = [o for o in c if c.get(o, 0) > ref.get(o, 0) + ion.core.get(o, 0)]
        exc = sum(max(0, ion.valence.get(o, 0) - (c.get(o, 0) - ion.core.get(o, 0))) for o in ion.valence)
        if exc == 2:
            assert all(n <= 3 for (n, l) in gained)


def test_restart_switch_and_algebra_key():
    from ssopt import deck as D
    ion = build_ion("Fe XV")
    cfgs = generate_ci_configs(ion, 4, 3, 2)
    n = D.n_lambda(cfgs, "orbital")
    a = D.write_ssin(cfgs, ion.Z, [1.0] * n, mode="orbital", mpnch=0)
    b = D.write_ssin(cfgs, ion.Z, [1.2] * n, mode="orbital", mpnch=-1)
    assert D.algebra_key(a) == D.algebra_key(b)                  # λ and MPNCH do not change the algebra
    r = D.set_restart(a, 4)
    assert D.algebra_key(r) == D.algebra_key(a)
    assert D.read_ssin(r)["configs"] == D.read_ssin(a)["configs"]
    term = [ln for ln in r.split("\n") if "-1" in ln[60:72]][0]
    assert term[:2] == " 4" and D.set_restart(r, 0) == a


def test_uncertainty_pieces():
    from ssopt import uncert as U
    from ssopt.ssout import E1, Level
    # NIST letter grades
    assert U.grade(0.02) == "A+" and U.grade(0.08) == "B" and U.grade(0.6) == "E"
    assert abs(U.frac(0.1) - (np.exp(0.1) - 1)) < 1e-12
    # ensemble: only λ sets inside the 68% region of the energy fit, best first
    h = History(["s", "p"], [(0.7, 2.0)] * 2, "orbital")
    for i, r in enumerate([0.010, 0.0105, 0.011, 0.02, 0.05]):
        h.samples.append(Sample([1.0 + 0.01 * i, 1.0], {}, r, 20, 1.0, True))
    pool, fac = U.ensemble(h.samples, h.best(), n_fit=20)
    assert pool[0].rms == 0.010 and all(s.rms <= fac * 0.010 for s in pool) and len(pool) == 2   # fac = 1.056
    # levels are followed across runs by (parity, 2J, rank in energy)
    lv = [Level(1, 1, 0, 2, 1, 0, 1, 0.0, 1), Level(2, 2, 0, 2, 1, 2, 1, 0.1, -1),
          Level(3, 3, 0, 2, 0, 2, 1, 0.2, -1)]
    assert U.level_keys(lv) == {1: (1, 0, 0), 2: (-1, 2, 0), 3: (-1, 2, 1)}
    # s_LV: own |ln(L/V)| and the mean over lines within a factor 10 in S
    e1 = [E1(2, 1, 1e9, 0.5, 0, 0, 1.0, 100.0, 0.5), E1(3, 1, 1e9, 0.5, 0, 0, -2.0, 90.0, 0.25)]   # SD carries a sign in ssout
    t = U.lv_terms(e1)
    assert t[(2, 1)][0] == 0.0 and abs(t[(3, 1)][0] - np.log(2)) < 1e-12
    assert abs(t[(2, 1)][1] - np.log(2) / 2) < 1e-12
    # spread over members
    keys = U.level_keys(lv)
    mem = [U.lines_by_key([E1(2, 1, a, 0.5, 0, 0, 1.0, 100.0, 0.5)], keys) for a in (1e9, 1.1e9, 0.9e9)]
    c = U.components(e1[:1], keys, mem)
    assert c[(2, 1)][3] == 3 and 0.05 < c[(2, 1)][0] < 0.15


def test_holdout_check():
    from ssopt import uncert as U
    class L:  # minimal stand-in for a NIST level
        def __init__(self, E):
            self.E = E
    nl = [L(0.1 * (i + 1)) for i in range(10)]
    h = History(["s"], [(0.7, 2.0)], "l")
    for lam, off in ((1.0, 0.02), (1.1, 0.005), (1.2, 0.01)):
        h.samples.append(Sample([lam], {i: nl[i].E + off for i in range(10)}, off, 10, 1.0, True))
    a, b, one, n = U.holdout(h, nl, "abs")
    assert n == 10 and abs(a - 0.005) < 1e-12 and abs(b - 0.005) < 1e-12 and abs(one - 0.02) < 1e-12


def test_ci_generator_respects_full_core_shells():
    from ssopt.ions import cap
    # an open 3d below a full 3p, and an open 4f below full 4s 4p 4d: no shell may overflow
    for name, n_max in (("Fe I", 4), ("Yb II", 6)):
        cfgs = generate_ci_configs(build_ion(name), n_max, 3, 2, "complex")
        assert cfgs and all(occ <= cap(l) for c in cfgs for (n, l), occ in c.items())
    yb = generate_ci_configs(build_ion("Yb II"), 6, 3, 2, "complex")
    assert any(c.get((4, 3)) == 14 and c.get((6, 0)) == 1 for c in yb)     # 4f14 6s, the real ground


def test_core_hole_levels_are_not_fit_targets():
    # a NIST level with a hole in a core shell (2p5 over the 2p6 core of Mg I) must not count
    # as reproducible by a deck that has no such configuration
    ion = build_ion("Mg I")
    deck = [parse_config_str("3s2"), parse_config_str("3s.3p"), parse_config_str("3s.4s")]
    keys = M.deck_valence_keys(deck, ion.core)
    assert M.valence_key(parse_config_str("2p6.3s2"), ion.core) in keys      # core written out in full
    assert M.valence_key(parse_config_str("3s.4s"), ion.core) in keys
    assert M.valence_key(parse_config_str("2p5.(2P*<3/2>).3s2.4s"), ion.core) not in keys
