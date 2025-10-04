#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate  λ = max{|λ2|, |λ_min|}  for S = D^{-1/2} A D^{-1/2}  using ARPACK (eigsh),
directly using indptr/indices/deg/meta.json from _csr_cache/<name.fingerprint[.dir2undir]>,
to avoid re-reading the huge original text graph.

This version:
- Removes command-line arguments; modify CONFIG directly.
- Prints whether Numba is successfully used at startup.
- During ARPACK iteration, prints progress (MV/s, elapsed time) according to matvec call count.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence

# ================
# CONFIG (edit directly)
# ================
DATASET_CACHE_DIRS: List[str] = [
    "./data/real-data/_csr_cache/friendster.b997c3ee43",
    # "./data/real-data/_csr_cache/twitter-2010.a45257d392.dir2undir",
    # "./data/real-data/_csr_cache/ABlueSky.xxxxxxxxxx",
]
ARPACK_TOL: float = 1e-3
ARPACK_MAXITER: int = 300
ARPACK_NCV_TOP: int = 64   # for which="LA"
ARPACK_NCV_BOT: int = 64   # for which="SA"
THREADS: int = 1           # limit backend parallelism to avoid implicit concurrency
OUT_JSON: str | None = None  # if you want to write results to file: "./lambda_estimates.json"

# Progress printing: print once every this many matvecs
PRINT_EVERY_MATVECS_TOP: int = 50
PRINT_EVERY_MATVECS_BOT: int = 50

# Optional: Numba (significantly accelerates matvec; falls back to pure NumPy loop if not available)
try:
    import numba as nb
    _HAS_NUMBA = True
except Exception:
    nb = None
    _HAS_NUMBA = False


# -------------------------
# Limit threads/BLAS (avoid implicit concurrency)
# -------------------------
def _set_env_threads(n: int | None) -> None:
    if not n:
        return
    n = int(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))


# -------------------------
# Progress tracker (based on matvec calls)
# -------------------------
class _MatvecProgress:
    def __init__(self, label: str, print_every: int):
        self.label = label
        self.print_every = max(1, int(print_every))
        self.calls = 0
        self.t0 = time.perf_counter()
        self.last_t = self.t0

    def tick(self):
        self.calls += 1
        if self.calls % self.print_every == 0:
            now = time.perf_counter()
            dt = now - self.last_t
            rate = (self.print_every / dt) if dt > 0 else float("inf")
            tot = now - self.t0
            print(f"[{self.label}] matvec calls={self.calls:,}  ~{rate:,.1f} MV/s  elapsed={tot:,.1f}s", flush=True)
            self.last_t = now

    def summary(self, extra: str = ""):
        tot = time.perf_counter() - self.t0
        print(f"[{self.label}] DONE  total matvec calls={self.calls:,}  elapsed={tot:,.1f}s {extra}", flush=True)


# -------------------------
# Read binary files from _csr_cache directory
# -------------------------
def map_csr_cache(cache_dir: str) -> Tuple[int, np.memmap, np.memmap, np.ndarray, Dict, int]:
    """
    Returns:
      n, indptr(int64 memmap), indices(int32 memmap), invsqrt_deg(float64 ndarray), meta(dict), nnz
    """
    base = Path(cache_dir)
    meta_path   = base / "meta.json"
    indptr_path = base / "indptr.i64.bin"
    indices_path= base / "indices.i32.bin"
    deg_path    = base / "deg.i32.bin"

    if not (meta_path.exists() and indptr_path.exists() and indices_path.exists() and deg_path.exists()):
        raise FileNotFoundError(f"cache dir missing required files: {base}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["n"])

    # memmap mapping; do not load entire graph into memory
    indptr  = np.memmap(indptr_path,  dtype=np.int64, mode="r")
    indices = np.memmap(indices_path, dtype=np.int32, mode="r")
    deg_u32 = np.memmap(deg_path,     dtype=np.uint32, mode="r", shape=(n,))

    invsqrt = np.zeros(n, dtype=np.float64)
    nz = deg_u32 > 0
    # invsqrt = 1/sqrt(deg); for zero-degree nodes set to 0 to avoid NaN
    invsqrt[nz] = 1.0 / np.sqrt(deg_u32[nz].astype(np.float64, copy=False))

    nnz = int(indices.shape[0])
    return n, indptr, indices, invsqrt, meta, nnz


# -------------------------
# Sx = D^{-1/2} A D^{-1/2} x matvec (with progress)
# -------------------------
def make_S_linear_operator(n: int,
                           indptr: np.ndarray,
                           indices: np.ndarray,
                           invsqrt: np.ndarray,
                           progress: _MatvecProgress,
                           use_numba: bool) -> LinearOperator:
    """
    Construct LinearOperator for S = D^{-1/2} A D^{-1/2} (symmetric, real, spectrum in [-1,1]).
    Avoid explicitly constructing csr_matrix to save memory from extra data array.
    """
    if use_numba:
        @nb.njit(cache=True, fastmath=True)
        def _matvec_jit(indptr_: np.ndarray, indices_: np.ndarray, invs_: np.ndarray, x_: np.ndarray) -> np.ndarray:
            # y_i = invs[i] * sum_{j in N(i)} invs[j] * x[j]
            y = np.empty(invs_.shape[0], dtype=np.float64)
            for i in range(invs_.shape[0]):
                s = 0.0
                b = int(indptr_[i]); e = int(indptr_[i + 1])
                for p in range(b, e):
                    j = int(indices_[p])
                    s += invs_[j] * x_[j]
                y[i] = invs_[i] * s
            return y

        def matvec(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64, order="C")
            y = _matvec_jit(indptr, indices, invsqrt, x)
            progress.tick()
            return y

        return LinearOperator(shape=(n, n), matvec=matvec, rmatvec=matvec, dtype=np.float64)

    else:
        invs = invsqrt

        def matvec(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64, order="C")
            y = np.empty(n, dtype=np.float64)
            for i in range(n):
                s = 0.0
                b = int(indptr[i]); e = int(indptr[i + 1])
                for p in range(b, e):
                    j = int(indices[p])
                    s += invs[j] * x[j]
                y[i] = invs[i] * s
            progress.tick()
            return y

        return LinearOperator(shape=(n, n), matvec=matvec, rmatvec=matvec, dtype=np.float64)


# -------------------------
# Estimate λ2 (second largest) and λmin (most negative) using ARPACK
# -------------------------
def estimate_lambda_from_cache(cache_dir: str,
                               tol: float = 1e-3,
                               ncv_top: int = 64,
                               ncv_bot: int = 64,
                               maxiter: int = 300,
                               print_every_top: int = 50,
                               print_every_bot: int = 50,
                               force_no_numba: bool = False) -> Dict:
    """
    Returns:
      {
        "dataset_cache_dir": ...,
        "n": n,
        "lambda_top": ~1.0,
        "lambda2": ...,
        "lambda_min": ...,
        "lambda": max(|lambda2|, |lambda_min|),
        ...
      }
    """
    n, indptr, indices, invsqrt, meta, nnz = map_csr_cache(cache_dir)

    # Decide whether to use Numba
    use_numba = _HAS_NUMBA and (not force_no_numba)
    print(f"[env] Numba available: {_HAS_NUMBA}, using Numba: {use_numba}")
    print(f"[graph] n={n:,}, nnz={nnz:,} (avg degree ≈ {nnz/n:.1f})")

    # --- Top eigenvalue stage (LA) ---
    prog_top = _MatvecProgress(label="ARPACK-LA", print_every=print_every_top)
    A_top = make_S_linear_operator(n, indptr, indices, invsqrt, progress=prog_top, use_numba=use_numba)

    # Trigger JIT warmup (only when using numba)
    if use_numba:
        x0 = np.random.default_rng(0).standard_normal(n, dtype=np.float64)
        t0 = time.perf_counter()
        _ = A_top @ x0  # one matvec triggers JIT compilation
        t1 = time.perf_counter()
        print(f"[jit] warmup compiled in {(t1 - t0)*1000:.2f} ms (excluded from ARPACK timing)")
        # warmup causes calls+1; reset counters for cleaner reporting
        prog_top.calls = 0
        prog_top.t0 = prog_top.last_t = time.perf_counter()

    # Solve LA (largest algebraic eigenvalues)
    k_top = 3 if n > 4 else max(2, min(2, n-1))  # if n>4 use 3, otherwise 2 (eigsh requires k < n)
    ncv_top = max(2 * k_top + 1, min(ncv_top, 256))
    t_la0 = time.perf_counter()
    try:
        vals_top, _ = eigsh(A_top, k=k_top, which="LA", tol=tol, maxiter=maxiter, ncv=ncv_top)
    except ArpackNoConvergence as e:
        vals_top = e.eigenvalues
        if vals_top.size < 2:
            vals_top, _ = eigsh(A_top, k=2, which="LA", tol=tol*10, maxiter=maxiter*2, ncv=max(20, ncv_top//2))
    t_la1 = time.perf_counter()
    prog_top.summary(extra=f"(solve time={(t_la1 - t_la0):.1f}s)")

    vals_top = np.sort(np.real(vals_top))[::-1]  # descending
    lambda_top = float(vals_top[0])
    eps = 1e-8
    cand = [v for v in vals_top if v < 1.0 - eps]
    lambda2 = float(max(cand)) if cand else 1.0

    # --- Smallest eigenvalue stage (SA) ---
    prog_bot = _MatvecProgress(label="ARPACK-SA", print_every=print_every_bot)
    A_bot = make_S_linear_operator(n, indptr, indices, invsqrt, progress=prog_bot, use_numba=use_numba)
    if use_numba:
        # short warmup for SA stage as well
        x1 = np.random.default_rng(1).standard_normal(n, dtype=np.float64)
        _ = A_bot @ x1
        prog_bot.calls = 0
        prog_bot.t0 = prog_bot.last_t = time.perf_counter()

    ncv_bot = max(20, min(ncv_bot, 2 * ncv_top))
    t_sa0 = time.perf_counter()
    try:
        vals_bot, _ = eigsh(A_bot, k=1, which="SA", tol=tol, maxiter=maxiter, ncv=ncv_bot)
    except ArpackNoConvergence as e:
        if e.eigenvalues.size > 0:
            vals_bot = e.eigenvalues[:1]
        else:
            vals_bot, _ = eigsh(A_bot, k=1, which="SA", tol=tol*10, maxiter=maxiter*2, ncv=max(20, ncv_bot//2))
    t_sa1 = time.perf_counter()
    prog_bot.summary(extra=f"(solve time={(t_sa1 - t_sa0):.1f}s)")

    lambda_min = float(np.real(vals_bot[0]))
    lam = max(abs(lambda2), abs(lambda_min))

    return {
        "dataset_cache_dir": str(cache_dir),
        "n": int(n),
        "nnz": int(nnz),
        "lambda_top": lambda_top,
        "lambda2": lambda2,
        "lambda_min": lambda_min,
        "lambda": lam,
        "which_matrix": "S = D^{-1/2} A D^{-1/2} (symmetric normalized adjacency)",
        "arpack": {"tol": float(tol), "ncv_top": int(ncv_top), "ncv_bot": int(ncv_bot), "maxiter": int(maxiter)},
        "numba": {"available": bool(_HAS_NUMBA), "used": bool(use_numba)},
        "meta": {"n": int(meta.get("n", n)),
                 "nnz": int(meta.get("nnz", nnz)),
                 "undirected": bool(meta.get("undirected", True)),
                 "source": meta.get("source", "")},
        "note": "eigsh on LinearOperator; per-iteration progress via matvec counting; Numba used if available."
    }


def main():
    _set_env_threads(THREADS)

    results: List[Dict] = []
    for cache_dir in DATASET_CACHE_DIRS:
        print(f"\n=== Estimating λ for: {cache_dir} ===")
        res = estimate_lambda_from_cache(
            cache_dir=cache_dir,
            tol=ARPACK_TOL,
            ncv_top=ARPACK_NCV_TOP,
            ncv_bot=ARPACK_NCV_BOT,
            maxiter=ARPACK_MAXITER,
            print_every_top=PRINT_EVERY_MATVECS_TOP,
            print_every_bot=PRINT_EVERY_MATVECS_BOT,
        )
        results.append(res)

        # Print summary results
        print(f"n           : {res['n']:,} (nnz={res['nnz']:,})")
        print(f"Numba       : available={res['numba']['available']}, used={res['numba']['used']}")
        print(f"λ_top       : {res['lambda_top']:.10f} (should be ≈ 1.0)")
        print(f"λ_2         : {res['lambda2']:.10f}")
        print(f"λ_min       : {res['lambda_min']:.10f}")
        print(f"λ = max(|.|): {res['lambda']:.10f}")

    if OUT_JSON:
        outp = Path(OUT_JSON)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n-> saved {len(results)} results to {outp}")


if __name__ == "__main__":
    main()
