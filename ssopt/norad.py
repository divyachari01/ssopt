"""NORAD-Atomic-Data (Nahar, OSU) — fetch and parse energy / f-value files.

File naming (https://www.astronomy.ohio-state.edu/nahar.1/nahar_radiativeatomicdata/<tag>/):
  <tag>.en.ls.txt   LS term energies (binding energies, Ry, relative to the
                    core ground state; negative or listed as |E|)
  <tag>.en.fs.txt   fine-structure levels (where available)
  <tag>.f.ls.txt    LS multiplet oscillator strengths
  <tag>.f.fs.txt    fine-structure transitions: label, Ei, Ef, dE, gi, gf, f, S, A
  <tag>.fsa.txt     H-like / He-like: fine structure energies + A values in one file
  <tag>.f.e1.txt    E1 transitions (some ions)

Not every ion is on NORAD (Si I, Si XIV, Ar XVII, Fe XVII are; Ni XV, Ni XVI
are not) — fetch_all() just reports what it found.
"""
from __future__ import annotations

import os
import re
import urllib.error
from dataclasses import dataclass

from .ions import Ion

BASE = "https://www.astronomy.ohio-state.edu/nahar.1/nahar_radiativeatomicdata"
SUFFIXES = ["en.ls.txt", "en.fs.txt", "f.ls.txt", "f.fs.txt", "f.e1.txt", "fsa.txt", "f.forbid.txt"]

_LTERM = {c: i for i, c in enumerate("SPDFGHIKLMNOQ")}


def url(ion: Ion, suffix: str) -> str:
    return f"{BASE}/{ion.tag}/{ion.tag}.{suffix}"


def cache_path(data_dir: str, ion: Ion, suffix: str) -> str:
    return os.path.join(data_dir, "norad", f"{ion.tag}.{suffix}")


def fetch_all(data_dir: str, ion: Ion, force: bool = False, timeout: int = 60) -> list[str]:
    import urllib.request
    got = []
    for suf in SUFFIXES:
        p = cache_path(data_dir, ion, suf)
        if os.path.exists(p) and not force:
            got.append(p)
            continue
        req = urllib.request.Request(url(ion, suf), headers={"User-Agent": "ssopt/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                continue
            raise
        if text.lstrip()[:15].lower().startswith("<!doctype") or "<html" in text[:300].lower():
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        got.append(p)
    return got


RY_CM = 109737.31568   # cm^-1 per Rydberg (R_inf)


@dataclass
class NoradTransition:
    """One fine-structure transition from an f.fs file.

    The file lists an LS-multiplet header  'a3Pe->z3Po  Ei Ef dE gi gf f S A'
    (Ei/Ef = |binding| in Ry) followed by one sub-row per fine-structure pair
    'Ei(cm-1) Ef(cm-1) dE(Ry) gi gf f S A'. Observed (NIST) energies are used
    in the sub-rows whenever available, so E_cm/RY_CM reproduces the NIST
    level energy; a '*' marks a calculated energy."""
    lower: str          # LS label e.g. 'a3Pe'
    upper: str          # e.g. 'z3Po'
    Ei: float           # excitation energy of the lower level (Ry, ground = 0)
    Ef: float           # of the upper level
    gi: int
    gf: int
    f: float
    S: float
    A: float            # s^-1
    calc_i: bool = False   # energy marked calculated
    calc_f: bool = False

    @staticmethod
    def _qn(label: str):
        m = re.match(r"^[a-zA-Z]?(\d)([SPDFGHIKL])([eo])?$", label.strip())
        if not m:
            return None, None, 0
        par = -1 if m.group(3) == "o" else 1
        return int(m.group(1)), _LTERM[m.group(2)], par

    @property
    def lower_qn(self):
        s, l, p = self._qn(self.lower)
        return (p, s, l, self.gi - 1)      # 2J = g - 1

    @property
    def upper_qn(self):
        s, l, p = self._qn(self.upper)
        return (p, s, l, self.gf - 1)


_HDR = re.compile(r"^\s*([a-zA-Z]?\d[SPDFGHIKL][eo]?)\s*->\s*([a-zA-Z]?\d[SPDFGHIKL][eo]?)\s")
_SUB = re.compile(
    r"^\s*\*?\s*([\d.]+)(\*?)\s+([\d.]+)(\*?)\s+([-\d.E+]+)\s+(\d+)\s+(\d+)\s+"
    r"([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)")


def parse_f_fs(text: str) -> list[NoradTransition]:
    out = []
    lower = upper = None
    for line in text.splitlines():
        h = _HDR.match(line)
        if h:
            lower, upper = h.group(1), h.group(2)
            continue
        if lower is None:
            continue
        m = _SUB.match(line)
        if not m:
            continue
        try:
            Ei = float(m.group(1)) / RY_CM
            Ef = float(m.group(3)) / RY_CM
            out.append(NoradTransition(lower, upper, Ei, Ef, int(m.group(6)), int(m.group(7)),
                                       float(m.group(8)), float(m.group(9)), float(m.group(10)),
                                       bool(m.group(2)), bool(m.group(4))))
        except ValueError:
            continue
    return out


def ionization_potential(text: str) -> float | None:
    m = re.search(r"I\.?P\.?\s*\(?Ry\)?\s*=\s*(-?[\d.]+)", text, re.I)
    if not m:
        m = re.search(r"Ionization Potential[^\d-]*(-?[\d.]+)", text, re.I)
    return float(m.group(1)) if m else None


def load_transitions(data_dir: str, ion: Ion):
    """(transitions, ionization potential) or (None, None) if not cached."""
    for suf in ("f.fs.txt", "fsa.txt", "f.e1.txt"):
        p = cache_path(data_dir, ion, suf)
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                text = f.read()
            tr = parse_f_fs(text)
            if tr:
                return tr, ionization_potential(text)
    return None, None
