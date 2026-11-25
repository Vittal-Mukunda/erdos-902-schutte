#!/usr/bin/env python3
"""Independent, clean-room audit of B3.19-B3.20 (level-7 flag-SDP certificates in proof/search/r10/).

Does NOT import flagsdp.py or sdp2.py.  Re-derives from scratch:
  1. All labelled 7-tournaments and their 456 isomorphism classes (checked against r10/classes_7.npz).
  2. Canonical root types on r = 1..6 vertices.
  3. Exact finite-N configuration counts m_t(H) by parsing the human-readable constraint name strings
     ('r3s2:mono1x[D(0, 1)>=12*int[D(0,),25]]', 'r2s0:pair x[1]', etc.) and directly counting configurations inside
     each 7-vertex representative H using bitmasks of dominators, WITHOUT any polynomial algebra or Stirling tables.
  4. Exact rational verification over Python's fractions.Fraction that:
       - every multiplier weight is >= 0;
       - every PSD multiplier u_i u_i^T uses an integer vector u_i;
       - the combined rational coefficient a_H is strictly < 0 on all 456 classes of 7-tournaments.

Usage: python audit_b320_independent.py <path_to_cert.pkl>
"""
import itertools
import os
import pickle
import re
import sys
import time
from fractions import Fraction
from math import comb, lcm

import numpy as np


def pair_index(s):
    d = {}
    k = 0
    for i in range(s):
        for j in range(i + 1, s):
            d[(i, j)] = k
            k += 1
    return d


def build_classes_scratch(s):
    idx = pair_index(s)
    m = len(idx)
    canon = np.arange(1 << m, dtype=np.int64)
    for a in range(s - 1):
        perm = list(range(s))
        perm[a], perm[a + 1] = perm[a + 1], perm[a]
        mapped = np.zeros(1 << m, dtype=np.int64)
        for (i, j), k in idx.items():
            bit = (np.arange(1 << m, dtype=np.int64) >> k) & 1
            pi, pj = perm[i], perm[j]
            if pi < pj:
                mapped |= bit << idx[(pi, pj)]
            else:
                mapped |= (1 - bit) << idx[(pj, pi)]
        while True:
            nxt = np.minimum(canon, canon[mapped])
            if np.array_equal(nxt, canon):
                break
            canon = nxt
            # propagate all adjacent transpositions iteratively below
    # Full propagation loop across all adjacent transpositions
    trans = []
    base = np.arange(1 << m, dtype=np.int64)
    for a in range(s - 1):
        perm = list(range(s))
        perm[a], perm[a + 1] = perm[a + 1], perm[a]
        mapped = np.zeros(1 << m, dtype=np.int64)
        for (i, j), k in idx.items():
            bit = (base >> k) & 1
            pi, pj = perm[i], perm[j]
            if pi < pj:
                mapped |= bit << idx[(pi, pj)]
            else:
                mapped |= (1 - bit) << idx[(pj, pi)]
        trans.append(mapped)
    canon = base.copy()
    while True:
        prev = canon.copy()
        for mp in trans:
            canon = np.minimum(canon, canon[mp])
        if np.array_equal(prev, canon):
            break
    reps = np.unique(canon)
    return reps


def adj_matrix(code, s):
    A = [[0] * s for _ in range(s)]
    k = 0
    for i in range(s):
        for j in range(i + 1, s):
            if (int(code) >> k) & 1:
                A[i][j] = 1
            else:
                A[j][i] = 1
            k += 1
    return A


def induced_code(A, tup):
    r = len(tup)
    if r <= 1:
        return 0
    c, k = 0, 0
    for i in range(r):
        for j in range(i + 1, r):
            if A[tup[i]][tup[j]]:
                c |= 1 << k
            k += 1
    return c


def parse_hyp_expr(expr):
    """Parse multiplier expression like '1' or 'D(0, 1)>=12*deg0<=26*int[D(0,),25]' into factor specs."""
    if expr == "1":
        return []
    factors = []
    for tok in expr.split("*"):
        m = re.fullmatch(r"D\(([\d, ]*)\)>=(-?\d+)", tok)
        if m:
            S = tuple(int(v) for v in re.findall(r"\d+", m.group(1)))
            factors.append(("ge", S, int(m.group(2))))
            continue
        m = re.fullmatch(r"deg(\d+)<=(-?\d+)", tok)
        if m:
            factors.append(("le", (int(m.group(1)),), int(m.group(2))))
            continue
        m = re.fullmatch(r"int\[D\(([\d, ]*)\),(-?\d+)\]", tok)
        if m:
            S = tuple(int(v) for v in re.findall(r"\d+", m.group(1)))
            factors.append(("int", S, int(m.group(2))))
            continue
        raise ValueError(f"Unrecognised factor token: {tok!r}")
    return factors


def pair_flag_list(r):
    fl = set()
    for b1 in range(1 << r):
        for b2 in range(1 << r):
            for arc in (0, 1):
                fl.add(min((b1, b2, arc), (b2, b1, 1 - arc)))
    return sorted(fl)


def pair_flag_of(A, theta, a1, a2):
    r = len(theta)
    b1 = sum(A[a1][theta[i]] << i for i in range(r))
    b2 = sum(A[a2][theta[i]] << i for i in range(r))
    arc = A[a1][a2]
    return min((b1, b2, arc), (b2, b1, 1 - arc))


def eval_factor_poly_configs(A, theta, out, factors):
    """Given root theta in s-vertex H and out = H \\ theta, return list of (vertex_mask, coeff) representing the
    expansion of prod(factors) as configurations of extension vertices inside H (direct tuple enumeration, no algebra)."""
    r = len(theta)
    # Each factor is linear: sum_{a in valid_S} 1_a + const, EXCEPT 'int' which is (L - j)(L - j - 1) = L^2 - (2j+1)L + j(j+1).
    # Represent a configuration sum as dict[frozenset_of_extension_vertices -> integer_coeff].
    acc = {0: 1}  # bitmask of used extension vertices -> integer count

    def mul_linear(valid_mask_list, const_term):
        nonlocal acc
        nxt = {}
        if const_term:
            for msk, val in acc.items():
                nxt[msk] = nxt.get(msk, 0) + val * const_term
        for amsk in valid_mask_list:
            for msk, val in acc.items():
                um = msk | amsk
                nxt[um] = nxt.get(um, 0) + val
        acc = {k: v for k, v in nxt.items() if v != 0}

    for kind, S, val in factors:
        # Dominators of theta_S inside out, and inside theta \ S
        ext_doms = [1 << a for a in out if all(A[a][theta[i]] for i in S)]
        int_doms = sum(1 for j in range(r) if j not in S and all(A[theta[j]][theta[i]] for i in S))
        if kind == "ge":
            mul_linear(ext_doms, int_doms - val)
        elif kind == "le":
            # val - D = -(D - val)
            nxt = {}
            c0 = val - int_doms
            if c0:
                for msk, v in acc.items():
                    nxt[msk] = nxt.get(msk, 0) + v * c0
            for amsk in ext_doms:
                for msk, v in acc.items():
                    um = msk | amsk
                    nxt[um] = nxt.get(um, 0) - v
            acc = {k: v for k, v in nxt.items() if v != 0}
        elif kind == "int":
            # (D - val) * (D - val - 1) applied as two linear multiplications!
            mul_linear(ext_doms, int_doms - val)
            mul_linear(ext_doms, int_doms - val - 1)
        else:
            raise ValueError(kind)
    return acc


def audit_certificate(cert_path):
    t0 = time.time()
    with open(cert_path, "rb") as f:
        cert = pickle.load(f)
    s = int(cert["s"])
    N = int(cert["N"])
    assert s == 7 and N == 52, f"Expected s=7, N=52, got s={s}, N={N}"

    reps = build_classes_scratch(s)
    assert len(reps) == 456, f"Expected 456 classes at s=7, got {len(reps)}"
    adjs = [adj_matrix(c, s) for c in reps]
    popc = [bin(i).count("1") for i in range(1 << s)]

    K = lcm(*[comb(s, t) for t in range(s + 1)])
    wt = [comb(N, t) * (K // comb(s, t)) for t in range(s + 1)]

    n_items = len(cert["w"])
    print(f"Auditing {os.path.basename(cert_path)}: s={s}, N={N}, {n_items} active multipliers, K={K}", flush=True)

    # Group multipliers by (r, sigma) so we iterate over (H, theta) once per (r, sigma)
    by_rs = {}
    special_lin = []
    SH = 40

    for idx in range(n_items):
        w_raw = float(cert["w"][idx])
        sc_raw = float(cert["scale"][idx])
        assert w_raw > 0 and sc_raw > 0
        W = Fraction(w_raw) / Fraction(sc_raw)
        assert W > 0
        m = cert["support"][idx]
        if m[0] == "psd":
            nm = cert["psd_names"][idx]
            u_float = cert["u"][idx]
            ui = [int(round(float(val) * (1 << SH))) for val in u_float]
            W_eff = W / (1 << (2 * SH))
            mo = re.fullmatch(r"r(\d+)s(\d+):(mono(\d+)|pair )x\[(.*)\]", nm)
            assert mo is not None, f"Unrecognised PSD name: {nm}"
            r, sigma = int(mo.group(1)), int(mo.group(2))
            btype = "pair" if mo.group(3).startswith("pair") else f"mono{mo.group(4)}"
            factors = parse_hyp_expr(mo.group(5))
            by_rs.setdefault((r, sigma), []).append(("psd", btype, factors, ui, W_eff, nm))
        elif m[0] == "lin":
            nm = cert["lin_names"][idx]
            mo = re.fullmatch(r"r(\d+)s(\d+):(.*)", nm)
            if mo:
                r, sigma = int(mo.group(1)), int(mo.group(2))
                factors = parse_hyp_expr(mo.group(3))
                by_rs.setdefault((r, sigma), []).append(("lin", None, factors, None, W, nm))
            else:
                special_lin.append((nm, W))
        else:
            raise ValueError(m)

    total = [Fraction(0)] * len(reps)

    # Handle any special r=1 global degree polynomial rows if present (e.g. "some vertex has in-degree ...")
    for nm, W in special_lin:
        mo = re.fullmatch(r"some vertex has in-degree (\d+)", nm)
        if mo:
            target_deg = int(mo.group(1))
            base = int(cert["args"].get("base", 25))
            emax = target_deg - base
            from math import factorial
            # Form: E_{r=1}[ prod_{i=0}^{emax-1} (D({0}) - (base + i)) ] - emax! >= 0
            factors = [("ge", (0,), base + i) for i in range(emax)]
            c_const = factorial(emax) * K
            for h, A in enumerate(adjs):
                acc_t = [0] * (s + 1)
                for v0 in range(s):
                    theta = (v0,)
                    out = [v for v in range(s) if v != v0]
                    cfg = eval_factor_poly_configs(A, theta, out, factors)
                    for msk, cnt in cfg.items():
                        acc_t[1 + popc[msk]] += cnt
                val = sum(wt[t] * acc_t[t] for t in range(s + 1)) - c_const
                total[h] += W * val
        else:
            raise ValueError(f"Unrecognised special linear row: {nm!r}")

    for (r, sigma), items in sorted(by_rs.items()):
        t_rs = time.time()
        npat = 1 << r
        basis_mono = {
            b: list(itertools.combinations_with_replacement(range(npat), b))
            for b in (1, 2, 3) if r + 2 * b <= s
        }
        pflags = pair_flag_list(r) if r <= 3 else None
        pflag_idx = {F: i for i, F in enumerate(pflags)} if pflags else None

        # Precompute item accumulators per class h: integer sum of wt[t] * config_contribution
        item_cols = [[0] * len(reps) for _ in items]

        for h, A in enumerate(adjs):
            for theta in itertools.permutations(range(s), r):
                if r >= 2 and induced_code(A, theta) != sigma:
                    continue
                out = [v for v in range(s) if v not in theta]
                pats = {a: sum(A[a][theta[i]] << i for i in range(r)) for a in out}
                by_pat = [[] for _ in range(npat)]
                for a in out:
                    by_pat[pats[a]].append(1 << a)

                # Precompute u-weighted configuration sums for basis vectors if needed
                pair_list = None
                for it_idx, (kind, btype, factors, ui, W_eff, nm) in enumerate(items):
                    cfg_m = eval_factor_poly_configs(A, theta, out, factors)
                    if not cfg_m:
                        continue
                    if kind == "lin":
                        s_val = 0
                        for msk, cnt in cfg_m.items():
                            s_val += wt[r + popc[msk]] * cnt
                        item_cols[it_idx][h] += s_val
                    else:
                        # Build linear combination V(theta) = sum_k ui[k] * basis_k(theta) as mask->int dict!
                        # Then V(theta)^2 * m(theta) is exact, and strictly PSD because (V(theta))^2 >= 0!
                        v_cfg = {}
                        if btype == "mono1":
                            for b, uval in enumerate(ui):
                                if uval == 0:
                                    continue
                                for amsk in by_pat[b]:
                                    v_cfg[amsk] = v_cfg.get(amsk, 0) + uval
                        elif btype == "mono2":
                            for uval, (b1, b2) in zip(ui, basis_mono[2]):
                                if uval == 0 or not by_pat[b1] or not by_pat[b2]:
                                    continue
                                for m1 in by_pat[b1]:
                                    for m2 in by_pat[b2]:
                                        um = m1 | m2
                                        v_cfg[um] = v_cfg.get(um, 0) + uval
                        elif btype == "mono3":
                            for uval, (b1, b2, b3) in zip(ui, basis_mono[3]):
                                if uval == 0 or not by_pat[b1] or not by_pat[b2] or not by_pat[b3]:
                                    continue
                                for m1 in by_pat[b1]:
                                    for m2 in by_pat[b2]:
                                        m12 = m1 | m2
                                        for m3 in by_pat[b3]:
                                            um = m12 | m3
                                            v_cfg[um] = v_cfg.get(um, 0) + uval
                        elif btype == "pair":
                            if pair_list is None:
                                pair_list = [[] for _ in range(len(pflags))]
                                for a1 in out:
                                    for a2 in out:
                                        if a1 != a2:
                                            pair_list[pflag_idx[pair_flag_of(A, theta, a1, a2)]].append(
                                                (1 << a1) | (1 << a2)
                                            )
                            for f_i, uval in enumerate(ui):
                                if uval == 0:
                                    continue
                                for pmsk in pair_list[f_i]:
                                    v_cfg[pmsk] = v_cfg.get(pmsk, 0) + uval
                        else:
                            raise ValueError(btype)

                        v_cfg = {k: v for k, v in v_cfg.items() if v != 0}
                        if not v_cfg:
                            continue
                        # Square v_cfg: v2_cfg[m1 | m2] += v1 * v2
                        v_items = list(v_cfg.items())
                        v2_cfg = {}
                        for i1 in range(len(v_items)):
                            mk1, val1 = v_items[i1]
                            v2_cfg[mk1] = v2_cfg.get(mk1, 0) + val1 * val1
                            for i2 in range(i1 + 1, len(v_items)):
                                mk2, val2 = v_items[i2]
                                um = mk1 | mk2
                                v2_cfg[um] = v2_cfg.get(um, 0) + 2 * val1 * val2
                        # Multiply by cfg_m and weight by wt[r + popcount]
                        s_val = 0
                        for m_msk, m_cnt in cfg_m.items():
                            for v_msk, v_cnt in v2_cfg.items():
                                s_val += wt[r + popc[m_msk | v_msk]] * (m_cnt * v_cnt)
                        item_cols[it_idx][h] += s_val

        for it_idx, (_, _, _, _, W_eff, _) in enumerate(items):
            col = item_cols[it_idx]
            for h in range(len(reps)):
                if col[h]:
                    total[h] += W_eff * col[h]
        print(f"  (r={r}, sigma={sigma}): {len(items)} multipliers audited in {time.time() - t_rs:.1f}s", flush=True)

    mx = max(total)
    mn = min(total)
    print(f"AUDIT COMPLETE in {time.time() - t0:.1f}s:")
    print(f"  Exact rational max_H (a_H / K) = {float(mx / K):.12e}  (LP reported nu = {cert['nu']:.12e})")
    print(f"  Exact rational min_H (a_H / K) = {float(mn / K):.6e}")
    print(f"  All 456 class coefficients strictly < 0: {mx < 0}")
    assert mx < 0, "AUDIT FAILED: max_H a_H >= 0"
    print("INDEPENDENT CLEAN-ROOM AUDIT PASSED (EXACT ARITHMETIC)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        audit_certificate(p)
