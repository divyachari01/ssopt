"""NIST ASD energy levels and lines — fetch (login node only) and parse.

Levels CSV (format=2) columns:
  Configuration, Term, J, Prefix, Level (Ry), Suffix, Uncertainty (Ry),
  Leading percentages, Reference
Each cell is written Excel-style as ="value".

Lines CSV (format=2) columns include:
  obs_wl_vac(A), ..., Aki(s^-1), Acc, Ei(Ry), Ek(Ry), conf_i, term_i, J_i,
  conf_k, term_k, J_k, ...

Both endpoints require de=0 or ASD silently returns nothing.
Units code: 0 = cm^-1, 1 = eV, 2 = Ry.
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass
from urllib.parse import urlencode

from .ions import parse_config_str, parity as cfg_parity

ASD_LEVELS = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
ASD_LINES = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"

_LTERM = {c: i for i, c in enumerate("SPDFGHIKLMNOQ")}


@dataclass
class NistLevel:
    config: str
    term: str
    J: float
    E: float            # Ry, ground = 0
    twoSp1: int | None
    L: int | None
    parity: int         # +1/-1, 0 if unknown
    cfg: dict
    leading_pct: str = ""

    @property
    def twoJ(self) -> int:
        return int(round(2 * self.J))

    @property
    def key(self):
        return (self.parity, self.twoSp1, self.L, self.twoJ)


@dataclass
class NistLine:
    wl: float           # vacuum Å (may be nan)
    A: float            # s^-1
    acc: str
    Ei: float           # Ry
    Ek: float           # Ry
    conf_i: str
    term_i: str
    J_i: float
    conf_k: str
    term_k: str
    J_k: float
    gf: float = float("nan")
    type: str = ""        # '' = E1, else 'M1', 'E2', 'M2', ...


def levels_url(spectrum: str, units: int = 2) -> str:
    params = {
        "de": 0, "spectrum": spectrum, "submit": "Retrieve Data", "units": units,
        "format": 2, "output": 0, "page_size": 15, "multiplet_ordered": 0,
        "conf_out": "on", "term_out": "on", "level_out": "on", "unc_out": 1,
        "j_out": "on", "lande_out": "on", "perc_out": "on", "biblio": "on", "temp": "",
    }
    return ASD_LEVELS + "?" + urlencode(params)


def lines_url(spectrum: str, units: int = 2) -> str:
    """All lines with transition probabilities, energies in `units`."""
    params = {
        "spectra": spectrum, "limits_type": 0, "low_w": "", "upp_w": "",
        "unit": 0,                       # wavelength unit: Å
        "de": 0, "format": 2, "line_out": 1,   # only lines with A values
        "en_unit": units, "output": 0, "page_size": 15,
        "show_obs_wl": 1, "show_calc_wl": 1, "unc_out": 1, "order_out": 0,
        "max_low_enrg": "", "show_av": 2, "max_upp_enrg": "", "tsb_value": 0,
        "min_str": "", "A_out": 0, "f_out": "on", "S_out": "on", "intens_out": "on",
        "max_str": "", "allowed_out": 1, "forbid_out": 1, "min_accur": "",
        "min_intens": "", "conf_out": "on", "term_out": "on", "enrg_out": "on",
        "J_out": "on", "submit": "Retrieve Data",
    }
    return ASD_LINES + "?" + urlencode(params)


def parse_term(term: str):
    """'3P*' -> (3, 1, -1);  '1D' -> (1, 2, +1);  unparsable -> (None, None, 0)."""
    t = term.strip()
    odd = -1 if t.endswith("*") else 1
    t = t.rstrip("*").strip()
    m = re.match(r"^(\d+)\[?([SPDFGHIKLMNOQ])", t.upper())
    if not m:
        return None, None, odd
    return int(m.group(1)), _LTERM[m.group(2)], odd


def _clean(cell: str) -> str:
    return cell.strip().lstrip("=").strip('"').strip()


def _rows(text: str) -> tuple[list[str], list[list[str]]]:
    head = text.lstrip()[:200].lower()
    if head.startswith(("<html", "<!doctype", "<?xml")):
        raise ValueError("NIST returned an HTML page, not data (check de=0 / spectrum)")
    for delim in (",", "\t"):
        rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if r and any(c.strip() for c in r)]
        if len(rows) < 2:
            continue
        header = [_clean(h) for h in rows[0]]
        if len(header) >= 3:
            return header, rows[1:]
    raise ValueError("could not parse NIST CSV")


def parse_levels(text: str) -> list[NistLevel]:
    header, rows = _rows(text)

    def col(name):
        for i, h in enumerate(header):
            if h.lower().startswith(name.lower()):
                return i
        return None

    ci = {k: col(k) for k in ("Configuration", "Term", "J", "Level", "Leading")}
    if ci["Level"] is None:
        raise ValueError(f"no Level column in {header}")
    out: list[NistLevel] = []
    for r in rows:
        def cell(k):
            i = ci[k]
            return _clean(r[i]) if i is not None and i < len(r) else ""
        raw = cell("Level").translate(str.maketrans("", "", "[]()+xa?"))
        try:
            E = float(raw)
        except ValueError:
            continue
        try:
            jtxt = cell("J")
            if "/" in jtxt:
                a, b = jtxt.split("/")
                J = float(a) / float(b)
            else:
                J = float(jtxt)
        except ValueError:
            continue           # skip unresolved / range J entries
        cfg_txt, term = cell("Configuration"), cell("Term")
        twoSp1, L, term_par = parse_term(term)
        cfg = parse_config_str(cfg_txt)
        par = cfg_parity(cfg) if cfg else term_par
        if cfg and twoSp1 is not None and par != term_par:
            # NIST config strings often omit the closed core; trust the term's '*'
            par = term_par
        out.append(NistLevel(cfg_txt, term, J, E, twoSp1, L, par, cfg, cell("Leading")))
    out.sort(key=lambda d: d.E)
    if out:
        e0 = out[0].E
        for d in out:
            d.E -= e0
    return out


def parse_lines(text: str) -> list[NistLine]:
    header, rows = _rows(text)
    hl = [h.lower() for h in header]

    def col(*names):
        for name in names:
            for i, h in enumerate(hl):
                if h.startswith(name.lower()):
                    return i
        return None

    c = {
        "wl": col("obs_wl_vac", "obs_wl", "ritz_wl_vac"), "A": col("Aki"), "acc": col("Acc"),
        "Ei": col("Ei"), "Ek": col("Ek"), "ci": col("conf_i"), "ti": col("term_i"),
        "Ji": col("J_i"), "ck": col("conf_k"), "tk": col("term_k"), "Jk": col("J_k"),
        "fik": col("fik"), "type": col("Type"),
    }
    if c["A"] is None or c["Ei"] is None:
        raise ValueError(f"unexpected lines header: {header}")
    out: list[NistLine] = []

    def num(r, k, default=float("nan")):
        i = c[k]
        if i is None or i >= len(r):
            return default
        s = _clean(r[i]).translate(str.maketrans("", "", "[]()+xa?"))
        try:
            return float(s)
        except ValueError:
            return default

    def J(r, k):
        i = c[k]
        s = _clean(r[i]) if i is not None and i < len(r) else ""
        try:
            if "/" in s:
                a, b = s.split("/")
                return float(a) / float(b)
            return float(s)
        except ValueError:
            return float("nan")

    def txt(r, k):
        i = c[k]
        return _clean(r[i]) if i is not None and i < len(r) else ""

    for r in rows:
        A = num(r, "A")
        if A != A:  # nan
            continue
        out.append(NistLine(num(r, "wl"), A, txt(r, "acc"), num(r, "Ei"), num(r, "Ek"),
                            txt(r, "ci"), txt(r, "ti"), J(r, "Ji"), txt(r, "ck"),
                            txt(r, "tk"), J(r, "Jk"), num(r, "fik"), txt(r, "type")))
    return out


def fetch(url: str, cache: str | None = None, timeout: int = 60) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "ssopt/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache, "w") as f:
            f.write(text)
    return text


def levels_cache_path(data_dir: str, spectrum: str) -> str:
    return os.path.join(data_dir, f"nist_{spectrum.replace(' ', '_')}_levels_ry.csv")


def lines_cache_path(data_dir: str, spectrum: str) -> str:
    return os.path.join(data_dir, f"nist_{spectrum.replace(' ', '_')}_lines_ry.csv")


def load_levels(data_dir: str, spectrum: str, fetch_if_missing: bool = True) -> list[NistLevel]:
    p = levels_cache_path(data_dir, spectrum)
    if not os.path.exists(p):
        if not fetch_if_missing:
            raise FileNotFoundError(p)
        fetch(levels_url(spectrum), cache=p)
    with open(p) as f:
        return parse_levels(f.read())


def load_lines(data_dir: str, spectrum: str, fetch_if_missing: bool = True) -> list[NistLine]:
    p = lines_cache_path(data_dir, spectrum)
    if not os.path.exists(p):
        if not fetch_if_missing:
            raise FileNotFoundError(p)
        fetch(lines_url(spectrum), cache=p)
    with open(p) as f:
        return parse_lines(f.read())
