#!/usr/bin/env python3
"""A real tournament T* on 52 vertices with all in-degrees in {25, 26} and every pair having >= 12 dominators:
a doubly regular tournament on 51 vertices (Goethals-Seidel, proof/search/r6/drt51.json) plus a vertex x whose
in-neighbourhood is N^-(w) + {w} for w = vertex 0.  COMPUTED.  Used as an end-to-end test of the balanced-Case-A SDP."""
import itertools, json, os
import numpy as np
import flagsdp as fs

D = np.array(json.load(open(os.path.join(fs.HERE, "..", "r6", "drt51.json")))[0]["A"], dtype=np.int8)
n = D.shape[0]
assert (D + D.T + np.eye(n, dtype=np.int8) == 1).all()
A = np.zeros((n + 1, n + 1), dtype=np.int8)
A[:n, :n] = D
x, w = n, 0
for v in range(n):
    if v == w or D[v, w]:        # v in N^-(w) or v = w : v -> x
        A[v, x] = 1
    else:
        A[x, v] = 1
assert (A + A.T + np.eye(n + 1, dtype=np.int8) == 1).all()
np.save(os.path.join(fs.HERE, "drt_twin52.npy"), A)
N = n + 1
indeg = A.sum(axis=0)
Ai = A.astype(np.int64)
C2 = Ai.T @ Ai
np.fill_diagonal(C2, 10**6)
print("in-degrees:", sorted(set(indeg.tolist())), "counts", {int(d): int((indeg == d).sum()) for d in set(indeg.tolist())})
print("min / max pair dominators:", C2.min(), np.where(C2 < 10**6, C2, 0).max())
At = A.T.astype(bool)                      # At[v] = in-neighbourhood indicator of v
k3 = {}
min4 = 10**6
for u, v in itertools.combinations(range(N), 2):
    duv = At[u] & At[v]
    for y in range(v + 1, N):
        d3 = duv & At[y]
        k = int(d3.sum())
        k3[k] = k3.get(k, 0) + 1
        if k <= 6:
            for z in range(y + 1, N):
                min4 = min(min4, int((d3 & At[z]).sum()))
print("triple dominator histogram:", dict(sorted(k3.items())))
print("min 4-set dominators (over triples with k <= 6):", min4)
