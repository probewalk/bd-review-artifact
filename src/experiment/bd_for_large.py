from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# =========================
# Optional: Numba (single-threaded JIT)
# =========================
try:
    import numba as nb  # type: ignore
    _HAS_NUMBA = True
except Exception:
    nb = None
    _HAS_NUMBA = False

# ===== NEW: requirements for parallelism =====
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==================================================================================
# 0) Thread/BLAS limits (optional)
# ==================================================================================

def set_env_threads(n: Optional[int]) -> None:
    if not n:
        return
    n = int(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    # Note: The main probewalk is Python/Numba single-threaded and doesn't rely on these libs; this is to avoid hidden parallelism affecting timings.

# ==================================================================================
# I. On-disk CSR (build + map)
# ==================================================================================

BIN_VERSION = 1

@dataclass
class CSRPaths:
    meta: Path
    indptr: Path
    indices: Path
    deg: Path

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def derive_csr_paths(txt_path: Path, out_root: Path, mode_tag: str = "plain") -> CSRPaths:
    """
    Derive output locations for binary CSR based on the source graph path.
    The directory name includes a fingerprint (path + mtime + size) and a mode tag (plain/dir2undir),
    to prevent rebuilding when the source file or mode hasn't changed.
    """
    txt_path = txt_path.resolve()
    stat = txt_path.stat()
    fingerprint = _sha1(f"{txt_path}:{int(stat.st_mtime)}:{stat.st_size}")
    base = out_root / f"{txt_path.stem}.{fingerprint}.{mode_tag}"
    base.mkdir(parents=True, exist_ok=True)
    return CSRPaths(
        meta   = base / "meta.json",
        indptr = base / "indptr.i64.bin",
        indices= base / "indices.i32.bin",
        deg    = base / "deg.i32.bin",
    )

def _dedup_rows_and_rewrite(paths: CSRPaths, n: int, progress: bool = True) -> None:
    """
    Perform per-row deduplication on CSR that has already been "written both ways",
    producing a CSR for a simple undirected graph.
    Key change: copy the memmap slice before sorting/unique to avoid in-place sort errors on read-only views.
    """
    indptr = np.memmap(paths.indptr, dtype=np.int64, mode="r")
    indices = np.memmap(paths.indices, dtype=np.int32, mode="r")

    # ---- Pass 1: count degrees after dedup (read-only + per-row copy) ----
    deg2 = np.empty(n, dtype=np.int64)
    max_row = 0
    for u in range(n):
        s, e = int(indptr[u]), int(indptr[u+1])
        if e > s:
            seg = np.array(indices[s:e], dtype=np.int32, copy=True)
            uniq_vals = np.unique(seg)
            cnt = int(uniq_vals.size)
        else:
            cnt = 0
        deg2[u] = cnt
        if cnt > max_row:
            max_row = cnt
        if progress and (u % 1_000_000 == 0):
            print(f"[CSR][dedup] count rows ... {u}/{n}")

    # ---- Build new indptr/indices according to deg2 ----
    indptr2 = np.empty(n + 1, dtype=np.int64)
    indptr2[0] = 0
    np.cumsum(deg2, out=indptr2[1:])
    nnz2 = int(indptr2[-1])

    indices2_path = paths.indices.with_name("indices.unique.i32.bin")
    indptr2_path  = paths.indptr.with_name("indptr.unique.i64.bin")
    deg2_path     = paths.deg.with_name("deg.unique.i32.bin")

    indices2 = np.memmap(indices2_path, dtype=np.int32, mode="w+", shape=(nnz2,))
    cursor = indptr2[:-1].copy()

    # ---- Pass 2: actually write deduplicated adjacency ----
    for u in range(n):
        s, e = int(indptr[u]), int(indptr[u+1])
        if e > s:
            seg = np.array(indices[s:e], dtype=np.int32, copy=True)
            uniq_vals = np.unique(seg)   # already sorted unique
            pos = int(cursor[u])
            ln = int(uniq_vals.size)
            if ln:
                indices2[pos:pos + ln] = uniq_vals
                cursor[u] = pos + ln
        if progress and (u % 1_000_000 == 0):
            print(f"[CSR][dedup] write rows ... {u}/{n}")
    del indices2  # flush

    # Overwrite old files
    indptr2_path.write_bytes(indptr2.tobytes(order="C"))
    np.asarray(deg2, dtype=np.uint32).tofile(deg2_path)
    os.replace(indptr2_path, paths.indptr)
    os.replace(indices2_path, paths.indices)
    os.replace(deg2_path, paths.deg)

    # Update meta
    with open(paths.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["nnz"] = int(nnz2)
    meta["dedup"] = True
    meta["max_degree_after_dedup"] = int(max_row)
    with open(paths.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[CSR] dedup done: nnz -> {nnz2}, max_degree≈{max_row}")


def build_csr_from_txt(txt_path: Path, out_root: Path,
                       assume_undirected: bool = True,
                       directed_as_undirected: bool = False,
                       progress: bool = True) -> CSRPaths:
    """
    Two-pass CSR construction:
      Pass1: count degrees (O(n) memory);
      Pass2: write adjacency by degree prefix sum (memmap, O(E) on disk, no memory overhead).
    Text format: first line n; subsequent lines u v (0-based). Skip self-loops and empty lines.
    When directed_as_undirected=True, perform per-row dedup after building to obtain a simple undirected graph.
    """
    mode_tag = "dir2undir" if directed_as_undirected else "plain"
    paths = derive_csr_paths(txt_path, out_root, mode_tag=mode_tag)

    if paths.meta.exists() and paths.indptr.exists() and paths.indices.exists() and paths.deg.exists():
        if progress:
            print(f"[CSR] Exists: {paths.meta.parent} (skip rebuild)")
        return paths

    if progress:
        print(f"[CSR] Build CSR from text: {txt_path} → {paths.meta.parent}  (mode={mode_tag})")

    # ---------- Pass 0: read n ----------
    with open(txt_path, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            raise ValueError("Graph file is empty or missing vertex count n.")
        n = int(first.strip())

    # ---------- Pass 1: count degree for each vertex (count as "undirected double write") ----------
    deg = np.zeros(n, dtype=np.uint64)  # use u64 to avoid overflow, persisted as u32
    m_valid = 0
    bytes_total = txt_path.stat().st_size
    bytes_done = 0
    last_report = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # skip first line
        for line in f:
            bytes_done += len(line)
            if not line or line == "\n":
                continue
            parts = line.split()
            if not parts:
                continue
            u = int(parts[0]); v = int(parts[1])
            if u == v:
                continue
            if u < 0 or u >= n or v < 0 or v >= n:
                continue
            # Undirected: u<->v (even if input is directed, we first count as undirected and dedup later)
            deg[u] += 1
            deg[v] += 1
            m_valid += 1
            if progress and bytes_done - last_report >= 64 * 1024 * 1024:
                pct = (bytes_done / max(1, bytes_total)) * 100.0
                print(f"[CSR][pass1] {pct:.1f}%  edges(valid)={m_valid}", flush=True)
                last_report = bytes_done

    # ---------- Build indptr via prefix-sum ----------
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(deg.astype(np.int64, copy=False), out=indptr[1:])
    nnz = int(indptr[-1])  # 2*m_valid

    # Write indptr / deg (u32)
    paths.indptr.write_bytes(indptr.tobytes(order="C"))
    np.asarray(deg, dtype=np.uint32).tofile(paths.deg)

    # ---------- Pass 2: write indices according to indptr (memmap) ----------
    indices_mm = np.memmap(paths.indices, dtype=np.int32, mode="w+", shape=(nnz,))
    # Cursor: write position
    cursor = indptr[:-1].copy()  # int64
    bytes_done = 0; last_report = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            bytes_done += len(line)
            if not line or line == "\n__":
                continue
            parts = line.split()
            if not parts:
                continue
            u = int(parts[0]); v = int(parts[1])
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                continue
            # u -> v
            pos = int(cursor[u]); indices_mm[pos] = v; cursor[u] = pos + 1
            # v -> u
            pos = int(cursor[v]); indices_mm[pos] = u; cursor[v] = pos + 1
            if progress and bytes_done - last_report >= 64 * 1024 * 1024:
                pct = (bytes_done / max(1, bytes_total)) * 100.0
                filled = int((cursor - indptr[:-1]).sum())
                print(f"[CSR][pass2] {pct:.1f}%  filled={filled}/{nnz}", flush=True)
                last_report = bytes_done
    del indices_mm  # flush

    meta = {
        "version": BIN_VERSION,
        "n": int(n),
        "edges_valid": int(m_valid),
        "nnz": int(nnz),
        "undirected": bool(assume_undirected),
        "directed_as_undirected": bool(directed_as_undirected),
        "source": str(txt_path),
        "mode_tag": mode_tag,
    }
    with open(paths.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    size_gb = Path(paths.indices).stat().st_size / (1024**3)
    print(f"[CSR] Build finished: n={n}, edges(valid)={m_valid}, nnz={nnz}, indices≈{size_gb:.1f} GB")

    # If "directed → undirected (dedup)" is needed, perform row-wise dedup here
    if directed_as_undirected:
        print("[CSR] directed_as_undirected=True → start row-wise dedup (in-row unique)")
        _dedup_rows_and_rewrite(paths, n, progress=progress)

    return paths

@dataclass
class _CSR:
    n: int
    indptr: np.ndarray   # int64 memmap/ndarray
    indices: np.ndarray  # int32 memmap/ndarray
    inv_deg: np.ndarray  # float64 ndarray (size n)

def map_csr(paths: CSRPaths, progress: bool = True) -> _CSR:
    """Memory-map indptr/indices, read deg to compute inv_deg (float64)."""
    with open(paths.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["n"])
    indptr = np.memmap(paths.indptr, dtype=np.int64, mode="r")
    indices = np.memmap(paths.indices, dtype=np.int32, mode="r")
    deg_u32 = np.memmap(paths.deg, dtype=np.uint32, mode="r", shape=(n,))
    inv_deg = np.zeros(n, dtype=np.float64)
    nz = deg_u32 > 0
    inv_deg[nz] = 1.0 / deg_u32[nz].astype(np.float64, copy=False)
    if progress:
        mem_est = inv_deg.nbytes / (1024**2)
        print(f"[CSR] Mapping finished: n={n}, inv_deg≈{mem_est:.1f} MB (others are paged on demand)")
    return _CSR(n=n, indptr=indptr, indices=indices, inv_deg=inv_deg)

# ==================================================================================
# II. probewalk algorithm (**logic copied from probewalk.py**, only the graph reader is replaced by CSR mmap)
# ==================================================================================

# --- MoM ---
def _median_of_means(group_means: List[float]) -> float:
    s = sorted(group_means)
    m = len(s)
    mid = m // 2
    return s[mid] if (m & 1) else 0.5 * (s[mid - 1] + s[mid])

# --- SplitMix64 and PRP utilities ---
MASK64 = (1 << 64) - 1
_GAMMA = 0xD1342543DE82EF95  # constant mixed with trial

def _splitmix64_py(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z ^= (z >> 30); z &= MASK64
    z = (z * 0xBF58476D1CE4E5B9) & MASK64
    z ^= (z >> 27); z &= MASK64
    z = (z * 0x94D049BB133111EB) & MASK64
    z ^= (z >> 31); z &= MASK64
    return z

def _affine_params_py(seed: int, tau: int, n: int) -> Tuple[int, int, int, int]:
    import math as _math
    M = n if (n & 1) == 0 else (n + 1)
    s = _splitmix64_py((seed ^ ((tau * _GAMMA) & MASK64)) & MASK64)
    a = int(s % M)
    if (a & 1) == 0:
        a = (a + 1) % M
    if a == 0:
        a = 1
    while _math.gcd(a, M) != 1:
        a = (a + 2) % M
        if a == 0:
            a = 1
    b = int(_splitmix64_py(s ^ 0x9E3779B97F4A7C15) % M)
    inva = pow(a, -1, M)
    return M, a, b, inva

def _sigma_base_py(i: int, seed: int, tau: int) -> float:
    v = (i ^ seed ^ ((tau * _GAMMA) & MASK64)) & MASK64
    return 1.0 if (_splitmix64_py(v) & 1) == 0 else -1.0

def _partner_index_py(i: int, n: int, M: int, a: int, b: int, inva: int) -> int:
    r = (((i - b) % M) * inva) % M
    rp = r ^ 1
    j = (a * rp + b) % M
    return j  # j may equal n (dummy)

def _sign_matched_pair_py(i: int, n: int, seed: int, tau: int, M: int, a: int, b: int, inva: int) -> float:
    j = _partner_index_py(i, n, M, a, b, inva)
    base = i if (j >= n or i < j) else j
    s = _sigma_base_py(base, seed, tau)
    return s if (j >= n or i < j) else -s

# --- Python version accumulate (matched-pair) ---
def _accumulate_U_moments_py_matched(indptr: np.ndarray, indices: np.ndarray, inv_deg: np.ndarray,
                                     n: int, s: int, t: int, L: int, R: int,
                                     seed: int, trial: int) -> Tuple[float, float]:
    S1 = 0.0
    S2 = 0.0
    # PRP parameters (once per trial)
    M, a, b, inva = _affine_params_py(seed, trial, n)
    # Use SplitMix64 to generate step RNG
    st = _splitmix64_py((seed ^ (trial * 0x94D049BB133111EB)) & MASK64)
    for _ in range(R):
        x = s
        xt = t
        A = inv_deg[s] * _sign_matched_pair_py(s, n, seed, trial, M, a, b, inva)
        Bv = inv_deg[t] * _sign_matched_pair_py(t, n, seed, trial, M, a, b, inva)
        for _ in range(L):
            # x step (Lemire unbiased integer sampling)
            start = int(indptr[x]); end = int(indptr[x + 1]); d = end - start
            if d > 0:
                while True:
                    st = _splitmix64_py(st)
                    m = (st * d) & MASK64
                    if (m & MASK64) >= (-d) % d:
                        off = (st * d) >> 64
                        x = int(indices[start + int(off % d)])
                        break
            # xt step
            start_xt = int(indptr[xt]); end_xt = int(indptr[xt + 1]); d_xt = end_xt - start_xt
            if d_xt > 0:
                while True:
                    st = _splitmix64_py(st)
                    m2 = (st * d_xt) & MASK64
                    if (m2 & MASK64) >= (-d_xt) % d_xt:
                        off2 = (st * d_xt) >> 64
                        xt = int(indices[start_xt + int(off2 % d_xt)])
                        break
            A += inv_deg[x] * _sign_matched_pair_py(x, n, seed, trial, M, a, b, inva)
            Bv += inv_deg[xt] * _sign_matched_pair_py(xt, n, seed, trial, M, a, b, inva)
        u = A - Bv
        S1 += u
        S2 += u * u
    return S1, S2

# --- Numba version (optional) ---
if _HAS_NUMBA:
    @nb.njit(cache=True)
    def _splitmix64(x: np.uint64) -> np.uint64:
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z ^= (z >> np.uint64(30))
        z *= np.uint64(0xBF58476D1CE4E5B9)
        z ^= (z >> np.uint64(27))
        z *= np.uint64(0x94D049BB133111EB)
        z ^= (z >> np.uint64(31))
        return z

    @nb.njit(cache=True, inline='always')
    def _sigma_base(i: int, seed: int, tau: int) -> float:
        v = np.uint64(i) ^ np.uint64(seed) ^ (np.uint64(tau) * np.uint64(0xD1342543DE82EF95))
        return 1.0 if (_splitmix64(v) & np.uint64(1)) == np.uint64(0) else -1.0

    @nb.njit(cache=True, inline='always')
    def _partner_index(i: int, n: int, M: int, a: int, b: int, inva: int) -> int:
        r = (((i - b) % M) * inva) % M
        rp = r ^ 1
        j = (a * rp + b) % M
        return j

    @nb.njit(cache=True, inline='always')
    def _sign_matched_pair(i: int, n: int, seed: int, tau: int, M: int, a: int, b: int, inva: int) -> float:
        j = _partner_index(i, n, M, a, b, inva)
        base = i if (j >= n or i < j) else j
        s = _sigma_base(base, seed, tau)
        return s if (j >= n or i < j) else -s

    @nb.njit(cache=True)
    def _accumulate_U_moments_jit_matched(indptr, indices, inv_deg,
                                          n: int, s: int, t: int, L: int, R: int,
                                          seed: int, trial: int,
                                          M: int, a: int, b: int, inva: int) -> Tuple[float, float]:
        S1 = 0.0
        S2 = 0.0
        # Note: Numba cannot call the Python version _affine_params_py; compute above and pass in
        st = _splitmix64(np.uint64(seed) ^ (np.uint64(trial) * np.uint64(0x94D049BB133111EB)))
        for _ in range(R):
            x = s
            xt = t
            A = inv_deg[s] * _sign_matched_pair(s, n, seed, trial, M, a, b, inva)
            Bv = inv_deg[t] * _sign_matched_pair(t, n, seed, trial, M, a, b, inva)
            for _ in range(L):
                # x step: Lemire unbiased sampling
                start = indptr[x]
                end = indptr[x + 1]
                d = int(end - start)
                if d > 0:
                    bnd = np.uint64(d)
                    t_rej = (np.uint64(0) - bnd) % bnd
                    while True:
                        st = _splitmix64(st)
                        if st >= t_rej:
                            off = int(st % bnd)
                            x = int(indices[start + off])
                            break
                # xt step
                start_xt = indptr[xt]
                end_xt = indptr[xt + 1]
                d_xt = int(end_xt - start_xt)
                if d_xt > 0:
                    bnd2 = np.uint64(d_xt)
                    t2 = (np.uint64(0) - bnd2) % bnd2
                    while True:
                        st = _splitmix64(st)
                        if st >= t2:
                            off2 = int(st % bnd2)
                            xt = int(indices[start_xt + off2])
                            break
                A += inv_deg[x] * _sign_matched_pair(x, n, seed, trial, M, a, b, inva)
                Bv += inv_deg[xt] * _sign_matched_pair(xt, n, seed, trial, M, a, b, inva)
            u = A - Bv
            S1 += u
            S2 += u * u
        return S1, S2

# --- Timing dataclass ---
@dataclass
class BPATimings:
    preprocessing: float
    random_walk: float
    reverse_prop: float
    h_calculation: float
    key_algo: float
    total: float

# --- probewalk main class (algorithm same; only replace "read graph" with CSR mmap) ---
class probewalk:
    name = "probewalk"

    def __init__(self, csr_root: Optional[str] = None, directed_as_undirected: bool = False):
        # csr_root: binary CSR output & cache directory (default {graph_dir}/_csr_cache)
        self._csr_root = Path(csr_root) if csr_root else None
        self._csr_cache: Dict[Tuple[str, bool], _CSR] = {}
        self._jit_ready: bool = False
        # directed→undirected (dedup) switch (from CLI)
        self._directed_as_undirected: bool = bool(directed_as_undirected)

    def _get_csr(self, graph_path: str, force_rebuild: bool = False, progress: bool = True) -> _CSR:
        """Read or build CSR (text → binary → mmap), choose cache dir by mode."""
        gpath = Path(graph_path)
        csr_root = self._csr_root or (gpath.parent / "_csr_cache")
        csr_root.mkdir(parents=True, exist_ok=True)
        mode_tag = "dir2undir" if self._directed_as_undirected else "plain"
        paths = derive_csr_paths(gpath, csr_root, mode_tag=mode_tag)
        if force_rebuild or not (paths.meta.exists() and paths.indptr.exists() and paths.indices.exists() and paths.deg.exists()):
            build_csr_from_txt(
                gpath, csr_root,
                assume_undirected=True,
                directed_as_undirected=self._directed_as_undirected,
                progress=progress
            )
        return map_csr(paths, progress=progress)

    def _jit_warmup_once(self, csr: _CSR, start: int, end: int, seed: int) -> float:
        """Trigger JIT compilation once; return warmup milliseconds (compile only, excluded from timings)"""
        if not _HAS_NUMBA or self._jit_ready:
            return 0.0
        n = csr.n
        M, a, b, inva = _affine_params_py(seed, -1, n)
        t0 = time.perf_counter_ns()
        _ = _accumulate_U_moments_jit_matched(
            csr.indptr, csr.indices, csr.inv_deg,
            int(n), int(start), int(end), 1, 1, int(seed), -1,
            int(M), int(a), int(b), int(inva)
        )
        t1 = time.perf_counter_ns()
        self._jit_ready = True
        return (t1 - t0) / 1_000_000.0  # ms

    def run(self, graph_path: str, *, start: int, end: int,
            L: int = 100, R: int = 200, G: int = 40, m: int = 10,
            seed: int = 42, verbose: bool = True, use_numba: bool = True,
            jit_warmup: bool = True, force_rebuild_csr: bool = False) -> dict:

        ns2ms = lambda ns: ns / 1_000_000.0

        # ========== Read/build CSR ==========
        t0 = time.perf_counter_ns()
        cache_key = (graph_path, self._directed_as_undirected)
        csr = self._csr_cache.get(cache_key)
        if csr is None or force_rebuild_csr:
            csr = self._get_csr(graph_path, force_rebuild=force_rebuild_csr, progress=verbose)
            self._csr_cache[cache_key] = csr
            preprocessing_info = "built_or_mapped_CSR"
        else:
            preprocessing_info = "reuse_mapped_CSR"
        n = csr.n

        if start < 0 or start >= n or end < 0 or end >= n:
            raise ValueError("start/end out of range")
        if csr.inv_deg[start] == 0.0 or csr.inv_deg[end] == 0.0:
            raise ValueError("start/end must have degree > 0")
        t1 = time.perf_counter_ns()

        # ========== JIT warmup (excluded from timings) ==========
        warmup_ms = 0.0
        if use_numba and _HAS_NUMBA and jit_warmup:
            try:
                warmup_ms = self._jit_warmup_once(csr, start, end, seed)
                if verbose and warmup_ms > 0:
                    print(f"[probewalk] JIT warmup compiled in {warmup_ms:.2f} ms (excluded from timings)")
            except Exception as e:
                if verbose:
                    print(f"[probewalk] JIT warmup skipped due to: {e!r}")

        # ========== Main computation ==========
        K = G * m
        group_means: List[float] = []
        t2 = time.perf_counter_ns()

        sum_Y = 0.0
        k_in_group = 0

        for trial in range(K):
            # Independent PRP parameters per trial (O(1) to generate)
            M, a, b, inva = _affine_params_py(seed, trial, n)

            if use_numba and _HAS_NUMBA:
                S1, S2 = _accumulate_U_moments_jit_matched(
                    csr.indptr, csr.indices, csr.inv_deg,
                    int(n), int(start), int(end), int(L), int(R),
                    int(seed), int(trial),
                    int(M), int(a), int(b), int(inva)
                )
            else:
                S1, S2 = _accumulate_U_moments_py_matched(
                    csr.indptr, csr.indices, csr.inv_deg,
                    int(n), int(start), int(end), int(L), int(R),
                    int(seed), int(trial)
                )

            Y_k = (S1 * S1 - S2) / (R * (R - 1))
            sum_Y += Y_k
            k_in_group += 1
            if k_in_group == m:
                group_means.append(sum_Y / m)
                sum_Y = 0.0
                k_in_group = 0

        est = _median_of_means(group_means)
        t3 = time.perf_counter_ns()

        timings = BPATimings(
            preprocessing=ns2ms(t1 - t0),
            random_walk=ns2ms(t3 - t2),
            reverse_prop=0.0,
            h_calculation=0.0,
            key_algo=ns2ms(t3 - t2),
            total=ns2ms(t3 - t0),
        )

        return {
            "method": self.name,
            "bst_2": float(est),
            "timings_ms": timings.__dict__,
            "n": int(n),
            "L": int(L),
            "K": int(K),
            "R": int(R),
            "m": int(m),
            "workers": 1,  # single-threaded
            "use_numba": bool(use_numba and _HAS_NUMBA),
            "jit_warmup_ms": float(warmup_ms),
            "preprocessing_info": preprocessing_info,
            "note": "mmap CSR; on-the-fly matched-pair z via affine PRP; single-thread JIT; warmup excluded",
        }

# ===== NEW: multiprocess worker (no algorithm change, only distribute pairs) =====
# Note: Each process maintains its own probewalk and mmap handles to avoid passing large objects/file handles between processes.
_WORKER_probewalk: Optional[probewalk] = None
_WORKER_USE_NUMBA_DEFAULT: bool = True

def _worker_init(csr_root: Optional[str], directed_as_undirected: bool, env_threads: Optional[int]) -> None:
    # Ensure child processes also limit BLAS/OpenMP threads to avoid oversubscription
    set_env_threads(env_threads)
    global _WORKER_probewalk
    _WORKER_probewalk = probewalk(csr_root=csr_root, directed_as_undirected=directed_as_undirected)

def _run_one_pair(task: Tuple[int, int, int, str, int, Dict[str, Any], bool]) -> Tuple[bool, int, List[int], Any, float]:
    """
    Return: (ok, idx, pair, bst_or_err, elapsed_ms)
    """
    idx, s, t, graph_path, seed, params, use_numba = task
    assert _WORKER_probewalk is not None, "worker probewalk not initialized"
    try:
        res = _WORKER_probewalk.run(
            graph_path,
            start=int(s), end=int(t),
            seed=int(seed), verbose=False,
            use_numba=bool(use_numba),
            **params,
        )
        bst = float(res.get("bst_2"))
        elapsed = float(res.get("timings_ms", {}).get("key_algo", 0.0))
        return True, idx, [int(s), int(t)], bst, elapsed
    except Exception as e:
        return False, idx, [int(s), int(t)], repr(e), 0.0

# ==================================================================================
# III. Experiment execution (read JSON config, follow core.py logging logic)
# ==================================================================================

def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must be a JSON object (dict)")
    return cfg

def _pick_pairs(n: int, inv_deg: np.ndarray, k: int, seed: int = 42) -> List[Tuple[int,int]]:
    rng = np.random.default_rng(seed)
    nonzero = np.flatnonzero(inv_deg > 0.0)
    if len(nonzero) < 2:
        raise ValueError("Too few valid nodes in the graph (degree 0)")
    pairs = []
    for _ in range(k):
        s = int(rng.choice(nonzero))
        t = int(rng.choice(nonzero))
        while t == s:
            t = int(rng.choice(nonzero))
        pairs.append((s, t))
    return pairs

def run_from_json_config(cfg: Dict[str, Any], *, directed_as_undirected: bool = False) -> None:
    # Thread limits (optional)
    set_env_threads(cfg.get("env_threads"))

    out_root = Path(cfg.get("out_dir", "exp_logs/large_probewalk"))
    out_root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    # Compatibility for field spellings: random_pair_num / random_piar_num (your example had a typo)
    random_pair_num = int(cfg.get("random_pair_num", cfg.get("random_piar_num", 0)))
    pairs_num = int(cfg.get("pairs_num", 10))
    datasets = cfg["datasets"]
    methods = cfg["methods"]
    info_json = "./data/real-data/info.json"

    # ===== Parallelism: default 5; can be overridden by cfg["workers"] or CLI --workers =====
    workers = int(cfg.get("workers", 5))
    if workers < 1:
        workers = 1

    # Method registry (currently only probewalk) — read dir2undir from CLI
    probewalk = probewalk(csr_root=cfg.get("csr_root"), directed_as_undirected=directed_as_undirected)
    methods_registry = {"probewalk": probewalk}

    # Optional: read info.json (if provided and contains pairs/gold_ans)
    info = {}
    if info_json and os.path.exists(info_json):
        try:
            with open(info_json, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception as e:
            print(f"[warn] Failed to read info_json: {e!r}")

    for ds in datasets:
        name = ds.get("name") or Path(ds["graph"]).stem
        graph_path = ds["graph"]
        ds_dir = out_root / name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Map CSR (main process does it once to avoid concurrent rebuild in children)
        csr = probewalk._get_csr(graph_path, force_rebuild=False, progress=True)
        n = csr.n

        # Choose pairs: prefer those specified in info.json; otherwise sample randomly
        base_name = os.path.basename(graph_path)
        if (base_name in info) and ("pairs" in info[base_name]):
            pairs = [tuple(p) for p in info[base_name]["pairs"]]
            pairs_source = "info_json"
        else:
            K = random_pair_num if random_pair_num > 0 else max(pairs_num, 10)
            pairs = _pick_pairs(n, csr.inv_deg, k=K, seed=seed)
            pairs_source = "random_nonzero_deg"

        # gold (optional)
        gold_map: Dict[int, float] = {}
        if (base_name in info) and ("gold_ans" in info[base_name]):
            for idx, g in enumerate(info[base_name]["gold_ans"]):
                gold_map[idx] = float(g)

        for mdef in methods:
            method_key = mdef["method"]
            if method_key not in methods_registry:
                print(f"[warn] Unregistered method: {method_key}, skip")
                continue
            method = methods_registry[method_key]
            params: Dict[str, Any] = dict(mdef.get("params", {}))
            # No algorithm parameter changes; just extract use_numba to avoid passing twice
            use_numba_param = bool(params.pop("use_numba", True))

            log: Dict[str, Any] = {
                "dataset": name,
                "n": int(n),
                "method_name": getattr(method, "name", method_key),
                "seed": seed,
                "pairs_source": pairs_source,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "params": dict(mdef.get("params", {})),  # record as-is
                "directed_as_undirected": directed_as_undirected,  # record mode (from CLI)
                "avg_time": 0.0,
                "avg_re": None,
                "results": [],
                # Record parallel info (new metadata, no effect on algorithm)
                "parallel_workers": workers,
                "execution_mode": "pair_level_process_pool",
            }

            print(f"[run] {name}/{method_key} params={params}  total pairs={pairs_num}  workers={workers}")

            # ====== Submit tasks in parallel (one task per pair) ======
            tasks: List[Tuple[int,int,int,str,int,Dict[str,Any],bool]] = []
            for idx, (s, t) in enumerate(pairs[:pairs_num]):
                tasks.append((idx, int(s), int(t), graph_path, seed, params, use_numba_param))

            results_buffer: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
            time_sum = 0.0
            re_sum = 0.0
            re_cnt = 0

            # Child process init: each process creates its own probewalk and mmap
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(cfg.get("csr_root"), directed_as_undirected, cfg.get("env_threads")),
            ) as ex:
                futs = [ex.submit(_run_one_pair, t) for t in tasks]
                for fut in as_completed(futs):
                    ok, idx, pair, bst_or_err, elapsed = fut.result()
                    if ok:
                        bst = float(bst_or_err)
                        results_buffer[idx] = {
                            "idx": idx,
                            "pair": pair,
                            "bst_2": bst,
                            "elapsed_ms": float(elapsed),
                        }
                        print(f"  ✓ pair {idx}/{pairs_num}: ({pair[0]}, {pair[1]})  bst_2={bst:.6e}, time={elapsed:.1f} ms")
                        time_sum += float(elapsed)
                        if idx in gold_map and gold_map[idx] != 0.0:
                            re = abs(bst - gold_map[idx]) / abs(gold_map[idx])
                            re_sum += re
                            re_cnt += 1
                    else:
                        print(f"  ✗ pair {idx}: ({pair[0]}, {pair[1]})  failed: {bst_or_err}")

            # Aggregate
            log["results"] = [r for r in results_buffer if r is not None]
            avg_time = time_sum / max(1, len(log["results"]))
            log["avg_time"] = avg_time
            if re_cnt > 0:
                log["avg_re"] = re_sum / re_cnt

            # Output filename: method_avgErr_avgTime.json (consistent with core.py)
            re_tag = (f"{log['avg_re']:.6f}" if log["avg_re"] is not None else "NA")
            out_path = ds_dir / f"{method_key}_{re_tag}_{avg_time:.1f}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)

            print(f"[done] {name}/{method_key}  → {out_path}")

# ==================================================================================
# IV. CLI
# ==================================================================================

def main():
    parser = argparse.ArgumentParser(description="BD on Large Graphs (mmap CSR + probewalk, JSON config)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file (see example at the top of the script)")
    # NEW: CLI switch — treat directed input as undirected and dedup
    parser.add_argument("--dir2undir", action="store_true",
                        help="Treat directed edges as undirected, and perform row-wise dedup after building (simple undirected graph)")
    # NEW: number of parallel workers (default 20)
    parser.add_argument("--workers", type=int, default=25,
                        help="Number of parallel processes (default 20)")
    args = parser.parse_args()

    cfg = load_json_config(args.config)
    # CLI takes precedence over cfg for minimal intrusion
    cfg["workers"] = int(args.workers) if args.workers else int(cfg.get("workers", 25))
    run_from_json_config(cfg, directed_as_undirected=bool(args.dir2undir))

if __name__ == "__main__":
    main()
