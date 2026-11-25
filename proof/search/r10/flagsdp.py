#!/usr/bin/env python3
"""Finite-N flag sum-of-squares SDP for tournaments (Erdos #902, session 2026-11-13, direction M1).

COMPUTED / search oracle.  Derivation and notation: proof/FLAGSDP_2026-11-13.md.

Unknowns: x_H = (number of s-subsets of V inducing H) / C(N, s), H over the isomorphism classes of tournaments on s
vertices.  Every "rooted moment"
        E[ prod_i c_{F_i} ] = sum over root embeddings theta of prod_i c_{F_i}(theta)
is an exact linear form in x at the given N (finite-N overlap terms included): it counts configurations
(theta, extensions) and a configuration whose vertex union has t elements lies in C(N-t, s-t) s-subsets.

Flag pattern convention: for an extension vertex a and roots theta_0..theta_{r-1}, bit i of the pattern is 1 iff
a -> theta_i (a beats root i).  The all-ones pattern is a dominator of the root set.
"""
import itertools
import os
import pickle
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- tournaments, isomorphism classes
def pair_index(s):
    idx, k = {}, 0
    for i in range(s):
        for j in range(i + 1, s):
            idx[(i, j)] = k
            k += 1
    return idx


def perm_codes(s, perm):
    """codes of all labelled s-tournaments after relabelling v -> perm[v]."""
    idx = pair_index(s)
    m = len(idx)
    codes = np.arange(1 << m, dtype=np.int64)
    new = np.zeros_like(codes)
    for (i, j), k in idx.items():
        bit = (codes >> k) & 1
        pi, pj = perm[i], perm[j]
        if pi < pj:
            new |= bit << idx[(pi, pj)]
        else:
            new |= (1 - bit) << idx[(pj, pi)]
    return new


_CLASS_CACHE = {}


def classes(s):
    """returns (cls, reps): cls[code] = class index, reps = list of canonical codes (min code of the orbit)."""
    if s in _CLASS_CACHE:
        return _CLASS_CACHE[s]
    fn = os.path.join(HERE, f"classes_{s}.npz")
    if os.path.exists(fn):
        d = np.load(fn)
        out = (d["cls"], d["reps"])
        _CLASS_CACHE[s] = out
        return out
    m = s * (s - 1) // 2
    lab = np.arange(1 << m, dtype=np.int64)
    gens = []
    for i in range(s - 1):
        p = list(range(s))
        p[i], p[i + 1] = p[i + 1], p[i]
        gens.append(perm_codes(s, p))
    while True:
        old = lab.copy()
        for g in gens:
            lab = np.minimum(lab, lab[g])
        if np.array_equal(old, lab):
            break
    reps = np.unique(lab)
    cls = np.searchsorted(reps, lab).astype(np.int32)
    np.savez_compressed(fn, cls=cls, reps=reps)
    _CLASS_CACHE[s] = (cls, reps)
    return cls, reps


def adj_from_code(code, s):
    A = np.zeros((s, s), dtype=np.int8)
    for (i, j), k in pair_index(s).items():
        if (int(code) >> k) & 1:
            A[i, j] = 1
        else:
            A[j, i] = 1
    return A


def code_of(A, verts):
    """code of the labelled tournament induced on the ordered tuple verts."""
    c, k = 0, 0
    n = len(verts)
    for i in range(n):
        for j in range(i + 1, n):
            if A[verts[i]][verts[j]]:
                c |= 1 << k
            k += 1
    return c


def root_types(r):
    """canonical labelled root tournaments on r vertices: list of codes (one per isomorphism class)."""
    if r <= 1:
        return [0]
    _, reps = classes(r)
    return [int(c) for c in reps]


def density_vector(A, s, fix0=False):
    """x_H for a tournament with adjacency matrix A.  fix0: only subsets containing vertex 0 (vertex-transitive A)."""
    N = A.shape[0]
    cls, reps = classes(s)
    idx = pair_index(s)
    counts = np.zeros(len(reps), dtype=np.int64)
    if fix0:
        it = (((0,) + c) for c in itertools.combinations(range(1, N), s - 1))
    else:
        it = itertools.combinations(range(N), s)
    while True:
        chunk = list(itertools.islice(it, 400000))
        if not chunk:
            break
        C = np.array(chunk, dtype=np.int64)
        code = np.zeros(len(C), dtype=np.int64)
        for (i, j), k in idx.items():
            code |= A[C[:, i], C[:, j]].astype(np.int64) << k
        counts += np.bincount(cls[code], minlength=len(reps))
    return counts / counts.sum(), counts


def paley(p):
    qr = {(x * x) % p for x in range(1, p)}
    A = np.zeros((p, p), dtype=np.int8)
    for x in range(p):
        for y in range(p):
            if x != y and ((y - x) % p) in qr:
                A[x, y] = 1
    assert (A + A.T + np.eye(p, dtype=np.int8) == 1).all()
    return A


# ----------------------------------------------------------------------------- rooted moments inside each class
def single_moments(s, r, sigma, dmax):
    """mom[mono] = array (s+1, nH): number of (theta, a_1..a_d) in H with union size t, theta inducing sigma,
    pattern(a_i) = mono[i].  mono = nondecreasing tuple of patterns; repetition of extension vertices allowed."""
    fn = os.path.join(HERE, f"mom_s{s}_r{r}_sig{sigma}_d{dmax}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    _, reps = classes(s)
    nH = len(reps)
    mom = {}
    for h, code in enumerate(reps):
        A = adj_from_code(code, s)
        for theta in itertools.permutations(range(s), r):
            if r >= 2 and code_of(A, theta) != sigma:
                continue
            out = [v for v in range(s) if v not in theta]
            pats = [sum(int(A[a][theta[i]]) << i for i in range(r)) for a in out]
            for d in range(dmax + 1):
                for tup in itertools.product(range(len(out)), repeat=d):
                    ps = tuple(pats[i] for i in tup)
                    if any(ps[i] > ps[i + 1] for i in range(d - 1)):
                        continue
                    t = r + len(set(tup))
                    arr = mom.get(ps)
                    if arr is None:
                        arr = mom[ps] = np.zeros((s + 1, nH), dtype=np.int64)
                    arr[t, h] += 1
    with open(fn, "wb") as f:
        pickle.dump(mom, f)
    return mom


def pair_flag(A, theta, a1, a2):
    r = len(theta)
    b1 = sum(int(A[a1][theta[i]]) << i for i in range(r))
    b2 = sum(int(A[a2][theta[i]]) << i for i in range(r))
    arc = int(A[a1][a2])
    return min((b1, b2, arc), (b2, b1, 1 - arc))


def pair_moments(s, r, sigma, dsingle):
    """pm[(mono, F, F')] = array (s+1, nH): number of (theta, singles, (a1,a2), (a1',a2')) with the given single
    patterns (nondecreasing), pair flags F and F' (ordered pairs, flag canonical under swapping the pair)."""
    fn = os.path.join(HERE, f"pmom_s{s}_r{r}_sig{sigma}_d{dsingle}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    _, reps = classes(s)
    nH = len(reps)
    pm = {}
    for h, code in enumerate(reps):
        A = adj_from_code(code, s)
        for theta in itertools.permutations(range(s), r):
            if r >= 2 and code_of(A, theta) != sigma:
                continue
            out = [v for v in range(s) if v not in theta]
            pats = {a: sum(int(A[a][theta[i]]) << i for i in range(r)) for a in out}
            prs = [(a1, a2, pair_flag(A, theta, a1, a2)) for a1 in out for a2 in out if a1 != a2]
            for d in range(dsingle + 1):
                for tup in itertools.product(out, repeat=d):
                    ps = tuple(pats[a] for a in tup)
                    if any(ps[i] > ps[i + 1] for i in range(d - 1)):
                        continue
                    base = set(tup)
                    for (a1, a2, F) in prs:
                        u1 = base | {a1, a2}
                        for (b1, b2, G) in prs:
                            t = r + len(u1 | {b1, b2})
                            key = (ps, F, G)
                            arr = pm.get(key)
                            if arr is None:
                                arr = pm[key] = np.zeros((s + 1, nH), dtype=np.int64)
                            arr[t, h] += 1
    with open(fn, "wb") as f:
        pickle.dump(pm, f)
    return pm


def weights(N, s):
    """w[t] = C(N,t)/C(s,t): coefficient turning 'count in H with union size t' into a form in the densities x."""
    return np.array([comb(N, t) / comb(s, t) for t in range(s + 1)])


# ----------------------------------------------------------------------------- polynomials in the single-flag counts
def pmul(p, q):
    out = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(sorted(m1 + m2))
            out[m] = out.get(m, 0) + c1 * c2
    return out


def padd(p, q, cq=1):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, 0) + cq * c
    return out


def pdeg(p):
    return max((len(m) for m in p), default=0)


ONE = {(): 1}


def dominator_form(r, sigma, S, npat):
    """polynomial counting the dominators of the root subset S: extension vertices whose pattern contains S, plus
    the other roots that beat all of S."""
    A = adj_from_code(sigma, r) if r >= 2 else np.zeros((1, 1), dtype=np.int8)
    mask = sum(1 << i for i in S)
    p = {(b,): 1 for b in range(npat) if b & mask == mask}
    internal = sum(1 for j in range(r) if j not in S and all(A[j][i] for i in S))
    if internal:
        p[()] = internal
    return p


# ----------------------------------------------------------------------------- fast tables (all root types at once)
def _stirling2(nmax):
    S = [[0] * (nmax + 1) for _ in range(nmax + 1)]
    S[0][0] = 1
    for n in range(1, nmax + 1):
        for k in range(1, n + 1):
            S[n][k] = k * S[n - 1][k] + S[n - 1][k - 1]
    return S


def _falling(n, j):
    v = 1
    for i in range(j):
        v *= n - i
    return v


def single_moments_all(s, r, D):
    """same content as single_moments, for every canonical root type sigma of size r at once.
    Uses: the number of maps from m positions to n vertices with image of size j is S(m,j) * n^(j)."""
    fn = os.path.join(HERE, f"smom_s{s}_r{r}_d{D}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    _, reps = classes(s)
    nH = len(reps)
    sig = set(root_types(r))
    St = _stirling2(max(D, 1))
    out_tab = {sg: {} for sg in sig}
    for h, code in enumerate(reps):
        A = adj_from_code(code, s).tolist()
        for theta in itertools.permutations(range(s), r):
            sg = code_of(A, theta) if r >= 2 else 0
            if sg not in sig:
                continue
            mom = out_tab[sg]
            nb = {}
            for a in range(s):
                if a in theta:
                    continue
                b = 0
                for i in range(r):
                    if A[a][theta[i]]:
                        b |= 1 << i
                nb[b] = nb.get(b, 0) + 1
            present = sorted(nb)
            for d in range(D + 1):
                for mono in itertools.combinations_with_replacement(present, d):
                    poly = [1]            # poly[u] = number of configurations with u distinct extension vertices
                    i = 0
                    while i < d:
                        b = mono[i]
                        m = 1
                        while i + m < d and mono[i + m] == b:
                            m += 1
                        i += m
                        n = nb[b]
                        fac = [0] + [St[m][j] * _falling(n, j) for j in range(1, m + 1)]
                        new = [0] * (len(poly) + m)
                        for u, cu in enumerate(poly):
                            if cu:
                                for j in range(1, m + 1):
                                    if fac[j]:
                                        new[u + j] += cu * fac[j]
                        poly = new
                    arr = mom.get(mono)
                    if arr is None:
                        arr = mom[mono] = np.zeros((s + 1, nH), dtype=np.int64)
                    for u, cu in enumerate(poly):
                        if cu:
                            arr[r + u, h] += cu
    with open(fn, "wb") as f:
        pickle.dump(out_tab, f)
    return out_tab


def pair_flag_list(r):
    fl = set()
    for b1 in range(1 << r):
        for b2 in range(1 << r):
            for arc in (0, 1):
                fl.add(min((b1, b2, arc), (b2, b1, 1 - arc)))
    return sorted(fl)


def pair_moments_all(s, r, ds):
    """pm[sigma][mono] = int array (nf, nf, s+1, nH); same content as pair_moments, vectorised over pairs of pairs."""
    fn = os.path.join(HERE, f"spmom_s{s}_r{r}_d{ds}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    _, reps = classes(s)
    nH = len(reps)
    sig = set(root_types(r))
    flags = pair_flag_list(r)
    fi = {F: i for i, F in enumerate(flags)}
    nf = len(flags)
    popc = np.array([bin(i).count("1") for i in range(1 << s)], dtype=np.int64)
    pm = {sg: {} for sg in sig}
    for h, code in enumerate(reps):
        A = adj_from_code(code, s).tolist()
        for theta in itertools.permutations(range(s), r):
            sg = code_of(A, theta) if r >= 2 else 0
            if sg not in sig:
                continue
            tab = pm[sg]
            out = [v for v in range(s) if v not in theta]
            pats = {a: sum(int(A[a][theta[i]]) << i for i in range(r)) for a in out}
            masks, fidx = [], []
            for a1 in out:
                for a2 in out:
                    if a1 != a2:
                        masks.append((1 << a1) | (1 << a2))
                        fidx.append(fi[pair_flag(A, theta, a1, a2)])
            masks = np.array(masks, dtype=np.int64)
            fidx = np.array(fidx, dtype=np.int64)
            U2 = masks[:, None] | masks[None, :]
            I = np.broadcast_to(fidx[:, None], U2.shape)
            J = np.broadcast_to(fidx[None, :], U2.shape)
            for d in range(ds + 1):
                for tup in itertools.product(out, repeat=d):
                    ps = tuple(pats[a] for a in tup)
                    if any(ps[i] > ps[i + 1] for i in range(d - 1)):
                        continue
                    base = 0
                    for a in tup:
                        base |= 1 << a
                    t = r + popc[U2 | base]
                    arr = tab.get(ps)
                    if arr is None:
                        arr = tab[ps] = np.zeros((nf, nf, s + 1, nH), dtype=np.int32)
                    np.add.at(arr, (I, J, t, h), 1)
    res = (flags, pm)
    with open(fn, "wb") as f:
        pickle.dump(res, f)
    return res
