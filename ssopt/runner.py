"""Run `structx` — optionally stopping as soon as the J-level table is done.

The radiative section of SUPERSTRUCTURE (everything after the term table)
costs ~5x the energies and is useless for λ tuning, so `stop_after_levels=True`
kills the process the moment the "LIST OF TERMS WITH A WEIGHTED MEAN" marker
appears in ssout. A hard timeout guards against hangs on pathological λ.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import ssout as so

MARKER = so.TERMS_MARKER.encode()


@dataclass
class RunResult:
    workdir: str
    ssout: str
    elapsed: float
    ok: bool
    stopped_early: bool = False
    timed_out: bool = False
    returncode: int | None = None
    levels: list = field(default_factory=list)
    configs: dict = field(default_factory=dict)
    scale_factors: list = field(default_factory=list)
    note: str = ""


def _prepare(workdir: str, binary: str, deck_text: str) -> tuple[str, str]:
    os.makedirs(workdir, exist_ok=True)
    binary = os.path.abspath(binary)
    local = os.path.join(workdir, "structx")
    if os.path.islink(local) and os.readlink(local) != binary:
        os.unlink(local)                      # stale/broken link from an earlier run
    if not os.path.lexists(local):
        try:
            os.symlink(binary, local)
        except FileExistsError:
            pass
    ssin = os.path.join(workdir, "ssin")
    with open(ssin, "w") as f:
        f.write(deck_text)
    return local, ssin


def run_structx(deck_text: str, workdir: str, binary: str, *,
                stop_after_levels: bool = True, timeout: float | None = 1800,
                poll: float = 0.25, keep_fort7: bool = False,
                restart_file: str | None = None) -> RunResult:
    """restart_file: a saved fort.10 (angular algebra) to read back; the deck must
    then carry restart code 4 (deck.set_restart)."""
    local, ssin = _prepare(workdir, binary, deck_text)
    f10 = os.path.join(workdir, "fort.10")
    if os.path.lexists(f10):
        os.remove(f10)
    if restart_file:
        os.symlink(os.path.abspath(restart_file), f10)
    out_path = os.path.join(workdir, "ssout")
    t0 = time.time()
    with open(ssin) as fin, open(out_path, "wb") as fout:
        proc = subprocess.Popen(["./structx"], stdin=fin, stdout=fout,
                                stderr=subprocess.DEVNULL, cwd=workdir)
        stopped = timed_out = False
        offset = 0
        carry = b""
        while True:
            rc = proc.poll()
            if stop_after_levels:
                try:
                    with open(out_path, "rb") as fr:
                        fr.seek(offset)
                        chunk = fr.read()
                except FileNotFoundError:
                    chunk = b""
                if chunk:
                    offset += len(chunk)
                    buf = carry + chunk
                    if MARKER in buf:
                        proc.kill()
                        stopped = True
                        proc.wait()
                        break
                    carry = buf[-len(MARKER):]
            if rc is not None:
                break
            if timeout and time.time() - t0 > timeout:
                proc.kill()
                proc.wait()
                timed_out = True
                break
            time.sleep(poll)
    elapsed = time.time() - t0
    if not keep_fort7:
        for junk in ("fort.7",):
            p = os.path.join(workdir, junk)
            if os.path.exists(p):
                os.remove(p)
    if restart_file and os.path.islink(f10):
        os.remove(f10)
    res = RunResult(workdir, out_path, elapsed, ok=False, stopped_early=stopped,
                    timed_out=timed_out, returncode=proc.returncode)
    if timed_out:
        res.note = "timeout"
        return res
    text = so.read(out_path)
    res.configs = so.parse_configs(text)
    if not res.configs and restart_file and os.path.exists(restart_file + ".ssout"):
        # a restarted run skips the algebra and so does not print the configuration
        # list; read it from the output of the run that saved the algebra
        res.configs = so.parse_configs(so.read(restart_file + ".ssout"))
    res.levels = so.parse_levels(text, res.configs)
    res.scale_factors = so.parse_scale_factors(text)
    res.ok = len(res.levels) > 1
    if not res.ok:
        res.note = "no level table"
    return res


def run_batch(decks: list[str], root: str, binary: str, *, n_workers: int | None = None,
              names: list[str] | None = None, **kw) -> list[RunResult]:
    """Run many decks concurrently, one subdirectory each."""
    n = len(decks)
    names = names or [f"run{i:04d}" for i in range(n)]
    n_workers = n_workers or max(1, os.cpu_count() or 1)

    def job(i):
        return run_structx(decks[i], os.path.join(root, names[i]), binary, **kw)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(job, range(n)))


def clean(root: str):
    shutil.rmtree(root, ignore_errors=True)
