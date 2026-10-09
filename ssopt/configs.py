"""CI configuration selection.

Two sources, merged:
  * spectroscopic set — every configuration label that appears in the NIST
    level list for the ion, restricted to the active-orbital cap (n_max,
    l_max). These are the configurations the deck MUST contain to reproduce
    the NIST spectrum at all.
  * correlation set — singles + physically-filtered ("paired") doubles from
    the valence reference into the same active set. These lower the energies
    of the spectroscopic terms through CI mixing.

Nothing is hardcoded per ion: the cap and the NIST list decide.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ions import Ion, cap, config_key, config_str
from .ions import EISSNER_ORDER, parity


def excitation_level(cfg: dict, ref: dict) -> int:
    orbs = set(cfg) | set(ref)
    return sum(max(0, ref.get(o, 0) - cfg.get(o, 0)) for o in orbs)


def active_orbitals(ion: Ion, n_max: int, l_max: int) -> list[tuple[int, int]]:
    orbs = set(ion.valence)
    base_n = min(n for (n, l) in ion.valence)
    for n in range(base_n, n_max + 1):
        for l in range(0, min(l_max, n - 1) + 1):
            orbs.add((n, l))
    return [o for o in EISSNER_ORDER if o in orbs]


def _distribute(orbs, n_left, idx, current, out):
    if idx == len(orbs):
        if n_left == 0:
            out.append(dict(current))
        return
    n, l = orbs[idx]
    for occ in range(0, min(cap(l), n_left) + 1):
        current[(n, l)] = occ
        _distribute(orbs, n_left - occ, idx + 1, current, out)
        current.pop((n, l), None)


def full_config(ion: Ion, valence: dict) -> dict:
    full = dict(ion.core)
    for o, occ in valence.items():
        if occ:
            full[o] = full.get(o, 0) + occ
    return full


def generate_ci_configs(ion: Ion, n_max: int = 4, l_max: int = 3, max_exc: int = 2,
                        doubles_mode: str = "paired") -> list[dict]:
    """Singles (+ doubles) of the valence electrons into the active set.
    doubles_mode='paired': both promoted electrons land in ONE subshell."""
    orbs = active_orbitals(ion, n_max, l_max)
    ref = {o: ion.valence.get(o, 0) for o in orbs}
    raw: list[dict] = []
    _distribute(orbs, ion.n_valence, 0, {}, raw)
    out, seen = [], set()
    for dist in raw:
        exc = excitation_level(dist, ref)
        if exc > max_exc:
            continue
        if doubles_mode == "paired" and exc == 2:
            gained = sum(1 for o in dist if dist.get(o, 0) > ref.get(o, 0))
            if gained > 1:
                continue
        full = full_config(ion, dist)
        k = config_key(full)
        if k in seen:
            continue
        seen.add(k)
        out.append(full)
    return out


def spectroscopic_from_nist(ion: Ion, nist_levels, n_max: int = 4, l_max: int = 3,
                            e_max: float | None = None) -> list[dict]:
    """Configurations named in NIST, within the active cap, as full configs."""
    out, seen = [], set()
    for d in nist_levels:
        if not d.cfg:
            continue
        if e_max is not None and d.E > e_max:
            continue
        val = {o: occ for o, occ in d.cfg.items() if o not in ion.core or occ > ion.core.get(o, 0)}
        # NIST omits the core; if it lists core shells, subtract them
        val = {o: occ - ion.core.get(o, 0) for o, occ in val.items() if occ - ion.core.get(o, 0) > 0}
        if sum(val.values()) != ion.n_valence:
            continue
        if any(n > n_max or l > l_max for (n, l) in val):
            continue
        full = full_config(ion, val)
        k = config_key(full)
        if k in seen:
            continue
        seen.add(k)
        out.append(full)
    return out


@dataclass
class DeckPlan:
    configs: list[dict]
    tags: list[str]              # 'spec' or 'corr'
    n_max: int
    l_max: int

    def __len__(self):
        return len(self.configs)

    def describe(self, ion: Ion) -> str:
        ref = full_config(ion, ion.valence)
        lines = []
        for c, t in zip(self.configs, self.tags):
            v = {o: n for o, n in c.items() if o not in ion.core}
            lines.append(f"  {config_str(v):<22} {t:<5} exc={excitation_level(c, ref)} [{'o' if parity(c) < 0 else 'e'}]")
        return "\n".join(lines)


def build_deck(ion: Ion, nist_levels=None, *, n_max: int = 4, l_max: int = 3,
               max_exc: int = 2, doubles_mode: str = "paired",
               e_max: float | None = None) -> DeckPlan:
    """Spectroscopic (NIST-derived) configs first, then correlation configs."""
    spec = spectroscopic_from_nist(ion, nist_levels or [], n_max, l_max, e_max)
    corr = generate_ci_configs(ion, n_max, l_max, max_exc, doubles_mode)
    ref = full_config(ion, ion.valence)
    seen = {config_key(c) for c in spec}
    ground_key = config_key(ref)
    if ground_key not in seen:
        spec.insert(0, ref)
        seen.add(ground_key)
    extra = [c for c in corr if config_key(c) not in seen]

    def sort_key(c):
        return (excitation_level(c, ref), tuple(sorted(o for o, occ in c.items() if occ)))

    spec.sort(key=sort_key)
    extra.sort(key=sort_key)
    return DeckPlan(spec + extra, ["spec"] * len(spec) + ["corr"] * len(extra), n_max, l_max)


def prune_groups(plan: DeckPlan, ion: Ion) -> dict[str, list[int]]:
    """Correlation configs grouped for leave-group-out testing, keyed by the
    orbital(s) they promote into, e.g. '3d^2', '4p'."""
    ref = full_config(ion, ion.valence)
    groups: dict[str, list[int]] = {}
    for i, (c, t) in enumerate(zip(plan.configs, plan.tags)):
        if t != "corr":
            continue
        gained = [(o, c.get(o, 0) - ref.get(o, 0)) for o in c if c.get(o, 0) > ref.get(o, 0)]
        name = "+".join(f"{n}{'spdfghi'[l]}{'^%d' % k if k > 1 else ''}" for (n, l), k in sorted(gained))
        groups.setdefault(name, []).append(i)
    return groups
