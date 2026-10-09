"""SUPERSTRUCTURE `ssin` deck writer / reader (Eissner packed format).

Card layout (verified byte-for-byte against a working Si I deck):

  card 1..n   FORMAT(4I2,1X,21(I2,A1),2A4)
              cols 1-8  MSTRT0 IFREE QCS0 QCL0   (only on the first card)
              cols 10.. stream of (occ, label) pairs, 11 per card.
              occ+50 means "same configuration as the pair to the left".
  terminator  ' 0' + 11 blanks + '0'(col 14) + 55 blanks + '-1'(cols 70-71)
              + 9-char label (cols 72-80).  QBMS(21)=-1 sets KUTSS=-1.
  control     FORMAT(14I5,I1,A1):
              NZION I MSCAL NGRP IMAXIT MPRIN0 ITOL MPNCH KUTCA MRED MQ
              INTRAN LSLIM1 LSLIM2  MECO  LFREE
  lambdas     free format (LFREE='Y'), 9 per line
  trailer     '0 0 0 0' / ' 0' / '' / ''
  legend      two lines listing the Eissner labels (not read by structx)

MSCAL semantics (from struct.f, SR.MINIM comments):
  MSCAL = nlam        -> λ grouped BY l: input values are λ_s λ_p λ_d λ_f ...
                         (IEQUAL(K) defaults to l+1). Extra values ignored.
  MSCAL = 1000 + nlam -> one λ per orbital, in Eissner order of the orbitals
                         actually used.
"""
from __future__ import annotations

import re

from .ions import EISSNER_ORDER, LABEL_OF, ORBITAL_OF_LABEL, config_key, L_LETTER

PAIRS_PER_CARD = 11
LAM_PER_LINE = 9


def orbitals_used(configs: list[dict]) -> list[tuple[int, int]]:
    """Orbitals occupied in any configuration, Eissner order."""
    used = {o for c in configs for o, occ in c.items() if occ > 0}
    return [o for o in EISSNER_ORDER if o in used]


def n_lambda(configs: list[dict], mode: str) -> int:
    orbs = orbitals_used(configs)
    if mode == "orbital":
        return len(orbs)
    if mode == "l":
        return max(l for (n, l) in orbs) + 1
    raise ValueError(mode)


def lambda_names(configs: list[dict], mode: str) -> list[str]:
    orbs = orbitals_used(configs)
    if mode == "orbital":
        return [f"{n}{L_LETTER[l]}" for (n, l) in orbs]
    return [L_LETTER[l] for l in range(max(l for (n, l) in orbs) + 1)]


def pack(configs: list[dict]) -> list[tuple[int, str]]:
    orbs = orbitals_used(configs)
    pairs = []
    for c in configs:
        first = True
        for o in orbs:
            occ = c.get(o, 0)
            if occ == 0:
                continue
            pairs.append((occ if first else occ + 50, LABEL_OF[o]))
            first = False
    return pairs


def write_ssin(configs: list[dict], nzion: int, lambdas: list[float],
               *, mode: str = "l", mpnch: int = -1, mprin0: int = -1,
               mq: int = 7, label: str = "Xfe21kshl",
               mode_control=(-2, -1, 0, 0), lam_decimals: int = 4,
               basis_n_max: int = 6, out_path: str | None = None) -> str:
    """Emit an ssin deck.

    mode='l'       : MSCAL = n_l  — λ per l-shell, len(lambdas) == n_l
    mode='orbital' : MSCAL = 1000 + n_orb — λ per orbital, len == n_orb
    """
    orbs = orbitals_used(configs)
    if not orbs:
        raise ValueError("no occupied orbitals")
    nlam = n_lambda(configs, mode)
    if len(lambdas) != nlam:
        raise ValueError(f"mode={mode!r} needs {nlam} lambdas, got {len(lambdas)}")
    # In 'l' mode SUPERSTRUCTURE reads NSCAL = n_orbitals values and maps
    # orbital K -> value IEQUAL(K) = l+1, so only the first n_l matter. We
    # keep MSCAL = n_orbitals (as in the reference deck) and pad with 1.0.
    mscal = len(orbs) if mode == "l" else 1000 + nlam
    lam_out = list(lambdas) + [1.0] * (len(orbs) - len(lambdas))

    pairs = pack(configs)
    lines = []
    for i in range(0, len(pairs), PAIRS_PER_CARD):
        chunk = pairs[i:i + PAIRS_PER_CARD]
        hdr = "".join(f"{v:2d}" for v in mode_control) if i == 0 else " " * 8
        lines.append(hdr + " " + "".join(f"{occ:2d}{ch}" for occ, ch in chunk))
    lines.append(" 0" + " " * 11 + "0" + " " * 55 + "-1" + f"{label:<9}")
    ctrl = [nzion, 0, mscal, 0, 0, mprin0, 0, mpnch, 0, 0, mq, 0, 0, 0]
    lines.append("".join(f"{v:5d}" for v in ctrl) + "0Y")
    for i in range(0, len(lam_out), LAM_PER_LINE):
        lines.append("".join(f"{v:<8.{lam_decimals}f}" for v in lam_out[i:i + LAM_PER_LINE]).rstrip())
    lines += ["0 0 0 0", " 0", "", ""]
    basis = [(n, l) for n in range(1, basis_n_max + 1) for l in range(n)]
    lines.append(" " + "  ".join(LABEL_OF[o] for o in basis))
    lines.append(" ".join(f"{n}{L_LETTER[l]}" for n, l in basis))
    text = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    return text


def read_ssin(text: str) -> dict:
    """Parse a deck back into {configs, nzion, mscal, mode, lambdas, mpnch, label}."""
    lines = text.splitlines()
    pairs: list[tuple[int, str]] = []
    i = 0
    hdr = None
    while i < len(lines):
        ln = lines[i]
        if ln[:2].strip() == "0" and "-1" in ln[60:72]:
            label = ln[71:80].strip()
            break
        if hdr is None:
            hdr = tuple(int(ln[k:k + 2]) for k in range(0, 8, 2))
        body = ln[9:]
        for k in range(0, len(body) - 2, 3):
            occ_s, lab = body[k:k + 2], body[k + 2]
            if not occ_s.strip():
                break
            pairs.append((int(occ_s), lab))
        i += 1
    else:
        raise ValueError("no terminator card")
    configs: list[dict] = []
    for occ, lab in pairs:
        o = ORBITAL_OF_LABEL[lab]
        if occ < 50:
            configs.append({})
        configs[-1][o] = configs[-1].get(o, 0) + (occ if occ < 50 else occ - 50)
    ctrl_line = lines[i + 1]
    ctrl = [int(ctrl_line[k:k + 5]) for k in range(0, 70, 5)]
    nzion, mscal, mpnch = ctrl[0], ctrl[2], ctrl[7]
    lam_tokens: list[float] = []
    j = i + 2
    while j < len(lines) and re.match(r"^\s*-?\d+\.\d", lines[j]):
        lam_tokens += [float(t) for t in lines[j].split()]
        j += 1
    mode = "orbital" if mscal >= 1000 else "l"
    nlam = mscal % 1000 if mode == "orbital" else n_lambda(configs, "l")
    return {"configs": configs, "nzion": nzion, "mscal": mscal, "mode": mode,
            "lambdas": lam_tokens[:nlam], "lambdas_raw": lam_tokens,
            "mpnch": mpnch, "label": label, "mode_control": hdr}


def configs_equal(a: list[dict], b: list[dict]) -> bool:
    return [config_key(c) for c in a] == [config_key(c) for c in b]
