from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

# =========================
# Optional: Numba (single-thread JIT)
# =========================
try:
    import numba as nb
    _HAS_NUMBA = True
except Exception:
    nb = None
    _HAS_NUMBA = False


@dataclass
class BPATimings:
    preprocessing: float
    random_walk: float
    reverse_prop: float
    h_calculation: float
    key_algo: float
    total: float


@dataclass
class _CSR:
    n: int
    indptr: np.ndarray   # int64
    indices: np.ndarray  # int32
    inv_deg: np.ndarray  # float64


def _build_csr_from_adj(degree: List[int], adj: List[List[int]]) -> _CSR:
    n = len(adj)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    total = 0
    for i in range(n):
        total += len(adj[i])
        indptr[i + 1] = total
    indices = np.empty(total, dtype=np.int32)
    pos = 0
    for i in range(n):
        nei = adj[i]
        ln = len(nei)
        if ln:
            indices[pos:pos + ln] = np.asarray(nei, dtype=np.int32)
            pos += ln
    deg = np.asarray(degree, dtype=np.int64)
    inv_deg = np.zeros(n, dtype=np.float64)
    nz = deg > 0
    inv_deg[nz] = 1.0 / deg[nz]
    return _CSR(n=n, indptr=indptr, indices=indices, inv_deg=inv_deg)


def _median_of_means(group_means: List[float]) -> float:
    s = sorted(group_means)
    m = len(s)
    mid = m // 2
    return s[mid] if (m & 1) else 0.5 * (s[mid - 1] + s[mid])


# =====================================================
# Constants and base mixing
# =====================================================
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

# ---------- Generate affine PRP parameters for matching (mod M = n or n+1) ----------
def _affine_params_py(seed: int, tau: int, n: int) -> Tuple[int, int, int, int]:
    """
    Return (M, a, b, inva), where π(x) = (a*x + b) % M is a permutation on [0, M).
    If n is odd, use M = n + 1 (introduce a dummy position to ensure an even number of pairs).
    """
    import math
    M = n if (n & 1) == 0 else (n + 1)
    s = _splitmix64_py((seed ^ ((tau * _GAMMA) & MASK64)) & MASK64)
    a = int(s % M)
    if (a & 1) == 0:
        a = (a + 1) % M
    if a == 0:
        a = 1
    # Ensure a is coprime with M, guaranteeing invertibility
    while math.gcd(a, M) != 1:
        a = (a + 2) % M
        if a == 0:
            a = 1
    b = int(_splitmix64_py(s ^ 0x9E3779B97F4A7C15) % M)
    inva = pow(a, -1, M)  # Python 3.8+: modular inverse
    return M, a, b, inva

def _sigma_base_py(i: int, seed: int, tau: int) -> float:
    v = (i ^ seed ^ ((tau * _GAMMA) & MASK64)) & MASK64
    return 1.0 if (_splitmix64_py(v) & 1) == 0 else -1.0

def _partner_index_py(i: int, n: int, M: int, a: int, b: int, inva: int) -> int:
    # r = π^{-1}(i) = inva * (i - b) (mod M), partner rank rp = r ^ 1, then map back j = π(rp)
    r = (((i - b) % M) * inva) % M
    rp = r ^ 1
    j = (a * rp + b) % M
    return j  # j may equal n (dummy)

def _sign_matched_pair_py(i: int, n: int, seed: int, tau: int, M: int, a: int, b: int, inva: int) -> float:
    j = _partner_index_py(i, n, M, a, b, inva)
    base = i if (j >= n or i < j) else j
    s = _sigma_base_py(base, seed, tau)
    return s if (j >= n or i < j) else -s


# ---------- Python version accumulate (matched-pair) ----------
def _accumulate_U_moments_py_matched(csr: _CSR, s: int, t: int, L: int, R: int,
                                     seed: int, trial: int) -> Tuple[float, float]:
    S1 = 0.0
    S2 = 0.0
    indptr, indices, inv_deg = csr.indptr, csr.indices, csr.inv_deg
    n = csr.n
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


# =====================================================
# Numba version (optional): matched-pair z (O(1) per coordinate)
# =====================================================
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
        v = np.uint64(i) ^ np.uint64(seed) ^ (np.uint64(tau) * np.uint64(_GAMMA & MASK64))
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


# =====================================================
# probewalk: on-demand y + matched-pair z + JIT (with warmup)
# =====================================================
class probewalk:
    name = "probewalk"

    def __init__(self):
        self._graph_cache: Dict[str, Tuple[int, List[int], List[List[int]]]] = {}
        self._csr_cache: Dict[str, _CSR] = {}
        self._jit_ready: bool = False   # whether JIT warmup has been done once

    def _get_graph(self, graph_path: str, force_reload: bool):
        if (not force_reload) and (graph_path in self._graph_cache):
            return self._graph_cache[graph_path], True
        from src.utils.data_loader import DataLoader
        loader = DataLoader(graph_path)
        n, degree, adj = loader.get_graph()
        self._graph_cache[graph_path] = (n, degree, adj)
        return (n, degree, adj), False

    def _get_csr(self, graph_path: str, degree: List[int], adj: List[List[int]]):
        csr = self._csr_cache.get(graph_path)
        if csr is None:
            csr = _build_csr_from_adj(degree, adj)
            self._csr_cache[graph_path] = csr
        return csr

    def _jit_warmup_once(self, csr: _CSR, start: int, end: int, seed: int) -> float:
        """Trigger a single JIT compilation; return warmup milliseconds (compilation only, excluded from timings)"""
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

    def run(
        self,
        graph_path: str,
        *            ,
        start: int,
        end: int,
        L: int = 100,
        R: int = 200,
        G: int = 40,
        m: int = 10,
        seed: int = 42,
        verbose: bool = False,
        force_reload: bool = False,
        use_numba: bool = True,     # Numba single-thread JIT
        jit_warmup: bool = True,    # warmup
    ) -> dict:

        ns2ms = lambda ns: ns / 1_000_000.0

        # ========== Load graph ==========
        t0 = time.perf_counter_ns()
        (n, degree, adj), cached = self._get_graph(graph_path, force_reload)
        csr = self._get_csr(graph_path, degree, adj)
        if start < 0 or start >= n or end < 0 or end >= n:
            raise ValueError("start/end out of range")
        if degree[start] == 0 or degree[end] == 0:
            raise ValueError("start/end must have degree > 0")
        inv_deg = csr.inv_deg
        t1 = time.perf_counter_ns()

        # ========== JIT warmup (excluded from timing) ==========
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
            # Independent PRP parameters per trial (generated in O(1))
            M, a, b, inva = _affine_params_py(seed, trial, n)

            if use_numba and _HAS_NUMBA:
                S1, S2 = _accumulate_U_moments_jit_matched(
                    csr.indptr, csr.indices, inv_deg,
                    int(n), int(start), int(end), int(L), int(R),
                    int(seed), int(trial),
                    int(M), int(a), int(b), int(inva)
                )
            else:
                S1, S2 = _accumulate_U_moments_py_matched(
                    csr, int(start), int(end), int(L), int(R),
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
            "workers": 1,  # no parallelism
            "use_numba": bool(use_numba and _HAS_NUMBA),
            "jit_warmup_ms": float(warmup_ms),
            "preprocessing_info": ("use cached graph" if cached else "loaded graph via DataLoader"),
            "note": "on-the-fly matched-pair z via affine PRP mod (n or n+1); single-thread JIT; warmup excluded from timings",
        }
