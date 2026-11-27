"""Pure-stdlib linear algebra for paskay.

No numpy dependency by doctrine (szl engine benches: "Stdlib only").
Small dense matrices only: hidden-state probe fits run on the trainer
lane with torch; this module powers the verified math core and the
offline verifier, which must run anywhere with nothing installed.

Limitations, stated honestly (fail-closed doctrine):
- svd() pairs singular vectors by descending singular value and fixes
  signs via u^T M v > 0. Repeated singular values leave an unresolved
  subspace ambiguity. Random measured data is distinct-valued almost
  surely; degeneracy is a named INCOMPARABLE risk upstream, not here.
"""

from __future__ import annotations

import math


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("matmul shape mismatch")
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def frobenius(a: list[list[float]]) -> float:
    return math.sqrt(sum(x * x for row in a for x in row))


def is_orthogonal(r: list[list[float]], tol: float = 1e-8) -> bool:
    """True iff R^T R == I within tol (semi-orthogonal when rectangular)."""
    rt = transpose(r)
    rt_r = matmul(rt, r)
    n = len(rt_r)
    for i in range(n):
        for j in range(n):
            want = 1.0 if i == j else 0.0
            if abs(rt_r[i][j] - want) > tol:
                return False
    return True


def jacobi_eigh(a_in: list[list[float]], sweeps: int = 100, tol: float = 1e-13):
    """Cyclic Jacobi eigensolver for a real symmetric matrix.

    Returns (eigenvalues, eigenvectors) sorted descending; eigenvectors
    are the columns of the returned matrix.
    """
    n = len(a_in)
    if any(len(row) != n for row in a_in):
        raise ValueError("jacobi_eigh needs a square symmetric matrix")
    a = [row[:] for row in a_in]
    v = identity(n)
    for _ in range(sweeps):
        off = math.sqrt(sum(a[p][q] ** 2 for p in range(n) for q in range(p + 1, n)))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-300:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp = v[k][p]
                    vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    pairs = sorted(
        ((a[i][i], i) for i in range(n)),
        key=lambda p: p[0],
        reverse=True,
    )
    vals = [p[0] for p in pairs]
    vecs = [[v[i][p[1]] for i in range(n)] for p in pairs]  # columns
    return vals, vecs


def svd(m: list[list[float]]):
    """Thin-by-construction SVD of a dense matrix via symmetric eigensolves.

    Returns (U, singular_values, V) with U columns, V columns, singular
    values descending and non-negative. Signs fixed so u_i^T M v_i > 0.
    """
    rows = len(m)
    cols = len(m[0]) if rows else 0
    if rows == 0 or cols == 0:
        raise ValueError("svd of empty matrix")
    mt = transpose(m)
    mmt = matmul(m, mt)   # rows x rows
    mtm = matmul(mt, m)   # cols x cols
    ev_u, u_cols = jacobi_eigh(mmt)
    ev_v, v_cols = jacobi_eigh(mtm)
    rank = min(rows, cols)
    u_out, s_out, v_out = [], [], []
    for i in range(rank):
        sv = math.sqrt(max(0.0, min(ev_u[i] if i < rows else 0.0,
                                    ev_v[i] if i < cols else 0.0)))
        u = u_cols[i] if i < rows else [0.0] * rows
        v = v_cols[i] if i < cols else [0.0] * cols
        # sign consistency: u^T M v should be positive
        mv = [sum(m[r][c] * v[c] for c in range(cols)) for r in range(rows)]
        p = sum(u[r] * mv[r] for r in range(rows))
        if p < 0:
            v = [-x for x in v]
        u_out.append(u)
        s_out.append(sv)
        v_out.append(v)
    return u_out, s_out, v_out


def random_orthogonal(n: int, seed: int) -> list[list[float]]:
    """Deterministic pseudo-random orthogonal matrix for controls.

    Gram-Schmidt over a seeded LCG gaussian: reproducible on any machine,
    no third-party RNG (receipt doctrine: controls must be replayable).
    """
    state = (seed * 2 + 1) & 0xFFFFFFFF
    def gauss():
        nonlocal state
        # sum of uniforms -> approximately normal (adequate for controls)
        acc = 0.0
        for _ in range(12):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            acc += state / 0xFFFFFFFF
        return acc - 6.0
    cols = []
    for _ in range(n):
        cols.append([gauss() for _ in range(n)])
    # Gram-Schmidt
    basis = []
    for col in cols:
        v = col[:]
        for b in basis:
            dot = sum(x * y for x, y in zip(v, b))
            v = [x - dot * y for x, y in zip(v, b)]
        norm = math.sqrt(sum(x * x for x in v))
        if norm < 1e-10:
            raise ValueError("degenerate random basis; change seed")
        basis.append([x / norm for x in v])
    return transpose(basis)
