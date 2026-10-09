"""Element data, spectroscopic-notation parsing, ground configurations.

A configuration is a dict {(n, l): occupancy} with occupancy > 0 only.
Orbital labels follow the SUPERSTRUCTURE / Eissner convention:
    1=1s 2=2s 3=2p 4=3s 5=3p 6=3d 7=4s 8=4p 9=4d A=4f B=5s ... L=6h
"""
from __future__ import annotations

from dataclasses import dataclass

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os "
    "Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U"
).split()
Z_OF = {s: i + 1 for i, s in enumerate(SYMBOLS)}

L_LETTER = "spdfghiklmnoq"
L_OF = {c: i for i, c in enumerate(L_LETTER)}

# Eissner orbital order and single-character labels.
EISSNER_ORDER = [(n, l) for n in range(1, 9) for l in range(n)]
EISSNER_LABELS = "123456789" + "".join(chr(c) for c in range(ord("A"), ord("Z") + 1))
LABEL_OF = {o: EISSNER_LABELS[i] for i, o in enumerate(EISSNER_ORDER) if i < len(EISSNER_LABELS)}
ORBITAL_OF_LABEL = {v: k for k, v in LABEL_OF.items()}

# Madelung filling order
AUFBAU = [
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1),
    (5, 0), (4, 2), (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0),
    (5, 3), (6, 2), (7, 1),
]

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def cap(l: int) -> int:
    """Maximum occupancy of an nl subshell."""
    return 2 * (2 * l + 1)


def orb_str(n: int, l: int, occ: int | None = None) -> str:
    s = f"{n}{L_LETTER[l]}"
    return f"{s}{occ}" if occ is not None else s


def parse_orbital(s: str) -> tuple[int, int]:
    """'3p' -> (3, 1); '4f' -> (4, 3)."""
    s = s.strip()
    return int(s[:-1]), L_OF[s[-1].lower()]


def roman_to_int(s: str) -> int:
    s = s.upper().strip()
    if not s or any(c not in _ROMAN for c in s):
        raise ValueError(f"not a Roman numeral: {s!r}")
    total, prev = 0, 0
    for c in reversed(s):
        v = _ROMAN[c]
        total += v if v >= prev else -v
        prev = v
    return total


def int_to_roman(num: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, r in vals:
        while num >= v:
            out += r
            num -= v
    return out


def parse_spectrum(s: str) -> tuple[str, int]:
    """'Si I' -> ('Si', 0);  'Fe XVII' -> ('Fe', 16);  'Ni 15' -> ('Ni', 14);
    'Si' -> ('Si', 0).  Returns (symbol, ionic charge)."""
    import re
    m = re.fullmatch(r"\s*([A-Za-z]{1,2})(\d+)\s*", s)
    if m:                                   # 'ar17', 'si1' tag style
        s = f"{m.group(1)} {m.group(2)}"
    tokens = s.replace("+", " ").split()
    if not tokens:
        raise ValueError("empty spectrum")
    head = tokens[0]
    symbol = SYMBOLS[int(head) - 1] if head.isdigit() else head.capitalize()
    if symbol not in Z_OF:
        raise ValueError(f"unknown element {head!r}")
    if len(tokens) == 1:
        return symbol, 0
    tail = tokens[1]
    stage = int(tail) if tail.isdigit() else roman_to_int(tail)
    return symbol, stage - 1


def spectroscopic(symbol: str, charge: int) -> str:
    return f"{symbol} {int_to_roman(charge + 1)}"


def ground_configuration(n_electrons: int) -> dict:
    """Aufbau filling. Anomalous cases (Cr, Cu, ...) are deliberately not
    applied: for a CI reference the Aufbau configuration is what we want."""
    cfg, left = {}, n_electrons
    for (n, l) in AUFBAU:
        if left <= 0:
            break
        take = min(cap(l), left)
        cfg[(n, l)] = take
        left -= take
    if left > 0:
        raise ValueError("extend AUFBAU")
    return cfg


def config_str(cfg: dict) -> str:
    return " ".join(orb_str(n, l, occ) for (n, l), occ in sorted(cfg.items()) if occ)


def parity(cfg: dict) -> int:
    """+1 even, -1 odd."""
    return -1 if sum(l * occ for (n, l), occ in cfg.items()) % 2 else 1


def parse_config_str(s: str) -> dict:
    """'3s2 3p2', '3s2.3p2', '3s2.3p.4s', '2s2.2p6.3s.3p' -> {(n,l): occ}.
    Handles NIST parent-term parentheses like '3s2.3p.(2P*).4s' by dropping
    anything in parentheses. Orbital letters are lowercase; anything else is
    not an orbital."""
    import re
    s = re.sub(r"\([^)]*\)", " ", s)
    cfg = {}
    for tok in re.split(r"[\s.]+", s):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)([spdfghiklmnoq])(\d*)", tok)
        if not m:
            continue
        n, l, occ = int(m.group(1)), L_OF[m.group(2)], int(m.group(3) or 1)
        cfg[(n, l)] = cfg.get((n, l), 0) + occ
    return cfg


def config_key(cfg: dict) -> tuple:
    return tuple(sorted((o, occ) for o, occ in cfg.items() if occ))


@dataclass
class Ion:
    symbol: str
    Z: int
    charge: int
    n_electrons: int
    ground: dict      # full ground configuration
    core: dict        # closed shells frozen in every configuration
    valence: dict     # active electrons

    @property
    def spectroscopic(self) -> str:
        return spectroscopic(self.symbol, self.charge)

    @property
    def n_valence(self) -> int:
        return sum(self.valence.values())

    @property
    def tag(self) -> str:
        """Filesystem-friendly: 'si1', 'ni15'."""
        return f"{self.symbol.lower()}{self.charge + 1}"


def build_ion(spectrum: str, valence_n_min: int | None = None) -> Ion:
    """Build an Ion from 'Si I', 'Ni XV', ...

    Valence = every subshell in the outermost occupied principal shell plus
    any open inner subshell. Everything else is core."""
    symbol, charge = parse_spectrum(spectrum)
    Z = Z_OF[symbol]
    n_elec = Z - charge
    if n_elec < 1:
        raise ValueError("charge too high")
    ground = ground_configuration(n_elec)
    max_n = max(n for (n, l) in ground)
    if valence_n_min is None:
        valence_n_min = max_n
    valence = {o: occ for o, occ in ground.items() if o[0] >= valence_n_min}
    for o, occ in ground.items():
        if occ < cap(o[1]):
            valence[o] = occ
    core = {o: occ for o, occ in ground.items() if o not in valence}
    return Ion(symbol, Z, charge, n_elec, ground, core, valence)
