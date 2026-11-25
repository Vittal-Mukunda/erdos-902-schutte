"""C2 check (proof/ORDER_NOISE_2026-11-25.md): bipartite doubling through the game value.
Bipartite tournament (A, B), M[a,b] = 1 iff b -> a.  nuA_s = min_{Y in C(A,s)} #{b : M[y,b]=1 for all y in Y},
nuB_s = min_{Z in C(B,s)} #{a : M[a,z]=0 for all z in Z}.  v = value of the game (B picks b, A picks a, payoff 1[b -> a]).
Claim (any bipartite tournament): nuA_{s+1} <= v nuA_s and nuB_{s+1} <= (1-v) nuB_s for 0 <= s < min(|A|,|B|)."""
import itertools, numpy as np
from scipy.optimize import linprog

def value(M):
    K, m = M.shape                      # rows a in A, columns b in B
    # B-player: maximise v s.t. sum_b Q(b) M[a,b] >= v for all a
    c = np.zeros(m + 1); c[-1] = -1
    A_ub = np.hstack([-M.astype(float), np.ones((K, 1))]); b_ub = np.zeros(K)
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)], method="highs")
    return -r.fun

def nus(M, smax):
    K, m = M.shape
    nA = [m]; nB = [K]
    for s in range(1, smax + 1):
        nA.append(min(int(np.all(M[list(Y), :] == 1, axis=0).sum()) for Y in itertools.combinations(range(K), s)))
        nB.append(min(int(np.all(M[:, list(Z)] == 0, axis=1).sum()) for Z in itertools.combinations(range(m), s)))
    return nA, nB

def check(name, M, smax, n=None):
    v = value(M); nA, nB = nus(M, smax)
    okA = all(nA[s+1] <= v * nA[s] + 1e-9 for s in range(smax))
    okB = all(nB[s+1] <= (1 - v) * nB[s] + 1e-9 for s in range(smax))
    extra = ""
    if n is not None:
        extra = f"  S_(n,n): nuA_n={nA[n]}, nuB_n={nB[n]};  sqrt(Km)={np.sqrt(M.size):.2f} >= (n+1)2^(n-1)={(n+1)*2**(n-1)}"
    print(f"{name}: v={v:.4f}  nuA={nA}  nuB={nB}  A-ok={okA} B-ok={okB}{extra}")
    return okA and okB

rows = ["111000", "110100", "100011", "010011", "001110", "001101"]
M6 = np.array([[int(c) for c in r] for r in rows])
check("6x6 S_(2,2) witness (audit 25 Nov)", M6, 3, n=2)

p = 19; qr = {(x*x) % p for x in range(1, p)}
M19 = np.array([[1 if a != b and ((a - b) % p) in qr else 0 for b in range(p)] for a in range(p)])  # b -> a iff a-b in QR
check("double of QR_19 (n=3)", M19, 4, n=3)

rng = np.random.default_rng(7); allok = True
for trial in range(300):
    K, m = rng.integers(3, 12, size=2)
    M = (rng.random((K, m)) < rng.uniform(0.2, 0.8)).astype(int)
    v = value(M); nA, nB = nus(M, min(K, m, 4) - 1)
    allok &= all(nA[s+1] <= v*nA[s] + 1e-9 for s in range(len(nA)-1)) and all(nB[s+1] <= (1-v)*nB[s] + 1e-9 for s in range(len(nB)-1))
print("300 random bipartite matrices (3..11 x 3..11): all per-level inequalities hold:", allok)
