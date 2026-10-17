"""Parsers for SUPERSTRUCTURE `ssout`.

Sections we use (in file order):
  * "CONFIGURATION CF=  n ... (N L)-COMBINATIONS"  -> CF index -> configuration
  * "SCALE FACTORS"                                 -> λ actually applied
  * "K  LV  K*CM  2*S+1  L  2J  CF  (EK-E1)/RY"     -> Breit-Pauli J-levels
  * "LIST OF TERMS WITH A WEIGHTED MEAN"            -> LS terms (S signed: <0 odd)
  * "**** INTERMEDIATE-COUPLING RADIATIVE DATA ****" -> E1 table (K KP AE1 GF ...)

The J-level table is complete once the "LIST OF TERMS WITH A WEIGHTED MEAN"
header appears; that is the marker the runner uses to stop structx early.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ions import parity as cfg_parity

LEVEL_HEADER = "2*S+1"
TERMS_MARKER = "LIST OF TERMS WITH A WEIGHTED MEAN"
RAD_MARKER = "INTERMEDIATE-COUPLING RADIATIVE DATA"
CONFIG_RE = re.compile(r"CONFIGURATION CF=\s*(\d+),.*\(N L\)-COMBINATIONS\s*(.*)$")


@dataclass
class Level:
    K: int          # index in energy order (1 = ground)
    LV: int
    E_cm: float
    twoSp1: int
    L: int
    twoJ: int
    CF: int
    E: float        # (EK-E1)/RY, ground-subtracted, Rydberg
    parity: int = 0  # +1 even / -1 odd, filled from the CF configuration

    @property
    def key(self):
        return (self.parity, self.twoSp1, self.L, self.twoJ)


@dataclass
class Term:
    I: int
    T: int
    E_cm: float
    twoSp1: int     # absolute value
    parity: int
    L: int
    CF: int
    E: float        # Ry


@dataclass
class E1:
    K: int
    KP: int
    A: float        # s^-1
    gf: float
    f_abs: float
    f_emi: float
    S: float
    wavel: float    # Å
    gf_vel: float


def parse_configs(text: str) -> dict[int, dict]:
    """CF index -> {(n,l): occ}. The (N L) list may continue on following lines.
    Shells declared closed on the first card ('CLOSED (N L)-SHELLS' in the header)
    are not repeated per configuration by structx; they are added back here."""
    out: dict[int, dict] = {}
    closed: dict = {}
    mc = re.search(r"CLOSED \(N L\)-SHELLS((?:\s+\d+\s+\d+)+)", text[:20000])
    if mc:
        v = [int(x) for x in mc.group(1).split()]
        closed = {(v[k], v[k + 1]): 2 * (2 * v[k + 1] + 1) for k in range(0, len(v) - 1, 2)}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = CONFIG_RE.search(lines[i])
        if m:
            cf = int(m.group(1))
            nums = m.group(2).split()
            j = i + 1
            # continuation lines are pure integer pairs, heavily indented
            while j < len(lines) and re.fullmatch(r"\s{20,}(\d+\s+)+\d*\s*", lines[j] + " "):
                nums += lines[j].split()
                j += 1
            cfg: dict = dict(closed)
            for k in range(0, len(nums) - 1, 2):
                o = (int(nums[k]), int(nums[k + 1]))
                cfg[o] = cfg.get(o, 0) + 1
            out[cf] = cfg
            i = j
        else:
            i += 1
    return out


def parse_scale_factors(text: str) -> list[float]:
    m = re.search(r"SCALE FACTORS((?:\s+[\d.]+)+(?:\n(?:\s+[\d.]+)+)*)", text)
    if not m:
        return []
    return [float(t) for t in m.group(1).split()]


def levels_complete(text: str) -> bool:
    """True once the J-level table has been fully written."""
    return TERMS_MARKER in text


def parse_levels(text: str, configs: dict[int, dict] | None = None) -> list[Level]:
    if configs is None:
        configs = parse_configs(text)
    levels: list[Level] = []
    grab = False
    for line in text.splitlines():
        if LEVEL_HEADER in line and "K*CM" in line:
            grab = True
            levels = []          # keep the LAST table (the BP one)
            continue
        if not grab:
            continue
        if TERMS_MARKER in line:
            break
        p = line.split()
        if len(p) < 8:
            if levels and not p:
                continue
            if levels and len(p) < 8:
                break
            continue
        try:
            K, LV = int(p[0]), int(p[1])
            E_cm = float(p[2].rstrip("."))
            twoSp1, L, twoJ, CF = int(p[3]), int(p[4]), int(p[5]), int(p[6])
            E = float(p[7])
        except ValueError:
            if levels:
                break
            continue
        par = cfg_parity(configs[CF]) if CF in configs else 0
        levels.append(Level(K, LV, E_cm, twoSp1, L, twoJ, CF, E, par))
    return levels


def parse_terms(text: str) -> list[Term]:
    terms: list[Term] = []
    grab = False
    for line in text.splitlines():
        if TERMS_MARKER in line:
            grab = True
            continue
        if not grab:
            continue
        if RAD_MARKER in line:
            break
        p = line.split()
        if len(p) < 7 or not p[0].isdigit():
            if terms and p and not p[0].isdigit():
                break
            continue
        try:
            I, T = int(p[0]), int(p[1])
            E_cm = float(p[2].rstrip("."))
            S, L, CF = int(p[3]), int(p[4]), int(p[5])
            E = float(p[6])
        except ValueError:
            continue
        terms.append(Term(I, T, E_cm, abs(S), -1 if S < 0 else 1, L, CF, E))
    return terms


_E1_ROW = re.compile(
    r"^\s*\d+\s+(\d+)\s+(\d+)\s+([\d.]+E[+-]\d+)\s+([\d.]+E[+-]\d+)\s+"
    r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+E[+-]\d+)")


def parse_e1(text: str) -> list[E1]:
    """Electric-dipole rows of the IC radiative table (skips AM2/E3 lines)."""
    out: list[E1] = []
    start = text.find(RAD_MARKER)
    if start < 0:
        return out
    for line in text[start:].splitlines():
        m = _E1_ROW.match(line)
        if not m:
            continue
        K, KP = int(m.group(1)), int(m.group(2))
        out.append(E1(K, KP, float(m.group(3)), float(m.group(4)), float(m.group(5)),
                      float(m.group(6)), float(m.group(7)), float(m.group(8)),
                      float(m.group(9))))
    return out


def read(path: str) -> str:
    with open(path, errors="replace") as f:
        return f.read()
