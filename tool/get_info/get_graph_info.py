#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the SLEM (second largest eigenvalue modulus) of the transition matrix P = D^{-1}A.
Results are saved into info.json.
"""

import os, glob, json, gc
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ========== Config ==========
ROOT_DIR     = "./data/real-data"  # graph file directory
FILE_PATTERN = "asia-osm.txt"
OUTPUT_JSON  = os.path.join(ROOT_DIR, "info_new.json")

ARPACK_NMAX  = 500_000     # if number of nodes <= this, use eigsh
POWER_ITERS  = 200         # number of power iterations
POWER_TRIES  = 3           # number of random initializations for power method, take max
TOL          = 1e-6
SEED         = 0
# ============================


def read_edge_list(path):
    """Read undirected edge list, return (n, A)."""
    rows, cols = [], []
    n_hint = None
    with open(path, "r") as f:
        first = f.readline().strip()
        if first.isdigit():
            n_hint = int(first)
        else:
            parts = first.split()
            if len(parts) == 2:
                u, v = map(int, parts)
                rows.append(u); cols.append(v)
                rows.append(v); cols.append(u)

        for line in f:
            s = line.strip()
            if not s: continue
            parts = s.split()
            if len(parts) < 2: continue
            u, v = int(parts[0]), int(parts[1])
            rows.append(u); cols.append(v)
            rows.append(v); cols.append(u)

    n = n_hint if n_hint else (max(rows+cols)+1 if rows else 0)
    if n == 0:
        return 0, sp.csr_matrix((0,0))

    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.setdiag(0)
    A = A.tocsr().maximum(A.T.tocsr())
    A.data[:] = 1.0
    A.eliminate_zeros()

    # Remove isolated vertices
    deg = np.asarray(A.sum(axis=1)).ravel()
    mask = deg > 0
    if mask.sum() < n:
        A = A[mask][:, mask]
        n = mask.sum()
    return n, A


def linear_operator_N(A):
    """Return LinearOperator for the symmetric matrix N = D^{-1/2} A D^{-1/2}."""
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    sqrt_d = np.sqrt(deg, dtype=np.float64)
    inv_sqrt_d = np.zeros_like(sqrt_d)
    nz = sqrt_d > 0
    inv_sqrt_d[nz] = 1.0 / sqrt_d[nz]
    def mv(v):
        y = v * inv_sqrt_d
        z = A.dot(y)
        return z * inv_sqrt_d
    return spla.LinearOperator(A.shape, matvec=mv, rmatvec=mv, dtype=np.float64), sqrt_d


def slem_arpack(A, k=6):
    """Compute SLEM directly using eigsh."""
    N, _ = linear_operator_N(A)
    vals, _ = spla.eigsh(N, k=min(k, A.shape[0]-1), which="LM")
    vals = np.sort(np.abs(vals))[::-1]
    # Discard values close to 1
    filtered = [v for v in vals if abs(v-1) > 1e-6]
    return float(filtered[0]) if filtered else 1.0


def slem_power(A, max_iter=200, tries=3, tol=1e-6, seed=0):
    """Estimate SLEM using the power method."""
    N, u = linear_operator_N(A)
    u_norm2 = float(u @ u)
    rng = np.random.default_rng(seed)
    best = 0.0
    for t in range(tries):
        v = rng.standard_normal(A.shape[0])
        if u_norm2 > 0:
            v = v - u * (float(v @ u) / u_norm2)
        v /= np.linalg.norm(v) if np.linalg.norm(v) > 0 else 1.0
        lam_prev = 0
        for it in range(max_iter):
            Nv = N.matvec(v)
            if u_norm2 > 0:
                Nv = Nv - u * (float(Nv @ u) / u_norm2)
            lam = float(v @ Nv) / float(v @ v)
            Nv_norm = np.linalg.norm(Nv)
            if Nv_norm > 0:
                v = Nv / Nv_norm
            if abs(lam - lam_prev) < tol: break
            lam_prev = lam
        best = max(best, abs(lam_prev))
    return float(best)


def process_file(path, idx, total):
    name = os.path.basename(path)
    print(f"[{idx}/{total}] Processing {name} ...", flush=True)
    n, A = read_edge_list(path)
    m = A.nnz // 2
    if n == 0 or m == 0:
        print(f"  Empty graph, SLEM=0")
        return {"n": n, "m": m, "slem": 0.0, "gap": 1.0, "method": "none"}

    try:
        if n <= ARPACK_NMAX:
            slem = slem_arpack(A)
            method = "eigsh"
        else:
            slem = slem_power(A, POWER_ITERS, POWER_TRIES, TOL, SEED)
            method = "power"
    except Exception as e:
        print(f"  Computation failed: {e}, falling back to power method")
        slem = slem_power(A, POWER_ITERS, POWER_TRIES, TOL, SEED)
        method = "power(fallback)"

    slem = min(1.0, max(0.0, slem))
    gap = 1.0 - slem
    print(f"  n={n}, m={m}, SLEM≈{slem:.6f}, gap≈{gap:.6f}, method={method}", flush=True)
    del A; gc.collect()
    return {"n": n, "m": m, "slem": slem, "gap": gap, "method": method}


def main():
    files = sorted(glob.glob(os.path.join(ROOT_DIR, FILE_PATTERN)))
    if not files:
        print("No files found!")
        return
    print(f"[INFO] Found {len(files)} graph file(s), starting processing...")
    results = {}
    for i, fp in enumerate(files, 1):
        results[os.path.basename(fp)] = process_file(fp, i, len(files))
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Results written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
