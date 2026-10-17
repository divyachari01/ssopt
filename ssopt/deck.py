"""SUPERSTRUCTURE `ssin` deck writer / reader (Eissner packed format).

Card layout (verified byte-for-byte against a working Si I deck):

  card 1..n   FORMAT(4I2,1X,21(I2,A1),2A4)
              cols 1-8  MSTRT0 IFREE QCS0 QCL0   (only on the first card)
              cols 10.. stream of (occ, label) pairs; we write one configuration
                        per card (structx reads the cards as one stream).
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


def pack(configs: list[dict], orbs: list | None = None) -> list[tuple[int, str]]:
    orbs = orbs or orbitals_used(configs)
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
               basis_n_max: int = 6, closed_core: bool = False,
               out_path: str | None = None) -> str:
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

    # One configuration per card, so the deck reads like the configuration list
    # (structx reads the cards as one stream, so the energies are identical to
    # packing 11 pairs per card). A configuration with more than 11 occupied
    # subshells would continue on the next card.
    # closed_core: the leading orbitals (Eissner order) that are full in every
    # configuration are declared closed on the first card (cols 5-8 = first and
    # last of them) and left out of each configuration.
    ncore = 0
    if closed_core:
        for (n, l) in orbs:
            if all(c.get((n, l), 0) == 2 * (2 * l + 1) for c in configs):
                ncore += 1
            else:
                break
    if ncore:
        mode_control = (mode_control[0], mode_control[1], 1, ncore)
    lines = []
    for c in configs:
        cp = pack([c], orbs[ncore:])
        for i in range(0, len(cp), PAIRS_PER_CARD):
            hdr = "".join(f"{v:2d}" for v in mode_control) if not lines else " " * 8
            lines.append(hdr + " " + "".join(f"{occ:2d}{ch}" for occ, ch in cp[i:i + PAIRS_PER_CARD]))
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


def _is_terminator(ln: str) -> bool:
    # no configuration on the card (cols 10-11) and KUTSS=-1 in cols 70-71;
    # cols 1-2 hold the restart switch, normally 0
    return ln[9:11].strip() in ("", "0") and "-1" in ln[60:72] and ln[:2].strip() in ("0", "4", "-1")


def _terminator_index(lines: list[str]) -> int:
    for i, ln in enumerate(lines):
        if _is_terminator(ln):
            return i
    raise ValueError("no terminator card")


def set_restart(text: str, code: int) -> str:
    """Set SUPERSTRUCTURE's restart switch (MSTRT0, cols 1-2 of the configuration
    terminator card).
      -1: do the angular algebra and save it on fort.10
       4: read the algebra back from fort.10 and go straight to the radial part
       0: no restart file (what write_ssin writes)
    The algebra depends only on the configuration cards, not on the λ, so one
    saved fort.10 serves every λ of a tuning run. The energies are identical."""
    lines = text.split("\n")
    i = _terminator_index(lines)
    lines[i] = f"{code:2d}" + lines[i][2:]
    return "\n".join(lines)


def limit_transitions(text: str, n_configs: int) -> str:
    """Radiative data only between levels whose leading configuration is one of
    the first n_configs of the deck (LCE1 = LCE2 = n_configs, cols 5-8 of the
    terminator card). With the NIST configurations listed first, this keeps
    every transition between NIST-type levels, with unchanged rates, and skips
    those involving levels of the correlation configurations."""
    lines = text.split("\n")
    i = _terminator_index(lines)
    ln = lines[i]
    lines[i] = ln[:4] + f"{n_configs:2d}{n_configs:2d}" + ln[8:]
    return "\n".join(lines)


def e1_only(text: str) -> str:
    """Skip the forbidden (parity-conserving E2/M1) transitions in the
    intermediate-coupling radiative data: LCE2 = -2 in cols 7-8 of the
    terminator card. The E1 table is unchanged; the full run takes about half
    the time."""
    lines = text.split("\n")
    i = _terminator_index(lines)
    ln = lines[i]
    lines[i] = ln[:6] + "-2" + ln[8:]
    return "\n".join(lines)


def algebra_key(text: str) -> str:
    """Hash of the configuration cards (everything up to the terminator), i.e.
    of what the saved angular algebra depends on."""
    import hashlib
    lines = text.split("\n")
    i = _terminator_index(lines)
    body = "\n".join(lines[:i] + ["  " + lines[i][2:]])
    return hashlib.md5(body.encode()).hexdigest()[:12]


def read_ssin(text: str) -> dict:
    """Parse a deck back into {configs, nzion, mscal, mode, lambdas, mpnch, label}."""
    lines = text.splitlines()
    pairs: list[tuple[int, str]] = []
    i = 0
    hdr = None
    while i < len(lines):
        ln = lines[i]
        if _is_terminator(ln):
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
