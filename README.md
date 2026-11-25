# Improved lower bounds for Schütte's tournament problem (Erdős Problem #902)

**Paper:** [`paper/schutte_lower_bounds.pdf`](paper/schutte_lower_bounds.pdf) (LaTeX source: [`paper/schutte_lower_bounds.tex`](paper/schutte_lower_bounds.tex))
**Author:** Vittal Mukunda (vittal.muku@gmail.com)
**Version:** tag `schutte-paper-v1`, 25 November 2026

A tournament has property *S_n* if every set of *n* vertices is beaten by a single vertex outside the set; *f(n)* is the least
number of vertices of such a tournament. The classical bounds are (n+2)·2^(n−1) − 1 ≤ f(n) ≤ (ln 2 + o(1))·n²·2ⁿ
(Szekeres–Szekeres 1965, Erdős 1963). The paper proves:

| result | proof |
|---|---|
| f(4) ≥ 52; every tournament on ≤ 51 vertices has a triple with at most 4 common dominators | by hand |
| f(n) ≥ (n + 5/2)·2^(n−1) − 1 for every n ≥ 4 | by hand |
| f(n) ≥ (c* − o(1))·n·2ⁿ with c* = 0.76950… (previously 1/2) | by hand |
| f(5) ≥ 121, f(6) ≥ 281, f(7) ≥ 665, f(8) ≥ 1485, f(9) ≥ 3388, f(10) ≥ 7398, f(11) ≥ 16569, f(12) ≥ 35655 | exact integer recursion |
| Bipartite version: f_bip(n) ≥ (n+1)·2ⁿ and f_bip(n) ≥ 2(c* − o(1))·n·2ⁿ | by hand |
| An S_4 tournament on 52 vertices with no tight pair has a vertex of in-degree ≥ 30 | computer-assisted (exact certificates) |

**The problem is not solved.** The order of magnitude of f(n) (between n·2ⁿ and n²·2ⁿ) and the question f(4) ≥ 53 remain open.

## Checking the computations

Requirements: Python 3.10+, `numpy`, `scipy`.

| what | command | expected |
|---|---|---|
| Numerical steps of Sections 2–6 (Theorem 1.2 inequality, c* to 40 digits, Table 1, brute-force checks on QR_7, QR_19, QR_67 and the 70-vertex S_4 tournament) | `python proof/verify_proof.py` | `Failures: 0` (output saved in `proof/verify_proof.out`; about 1 minute) |
| Certificates of Proposition 9.3 | `python proof/search/r11/audit_b320_independent.py proof/search/r10/cert_s7_N52_balF1.pkl` (likewise `_e2`, `_e3`, `_e4`, `_balA`) | `All 456 class coefficients strictly < 0: True` and `INDEPENDENT CLEAN-ROOM AUDIT PASSED` |
| Lemma 5.1 on the 6+6 example of Remark 5.2 | `python proof/search/r16/bip_gamevalue.py` | `A-ok=True B-ok=True` |
| Remark 9.4 (DRT(51) plus a twin vertex) | `python proof/search/r10/drt_twin.py` | in-degrees 25 and 26, every pair ≥ 12 dominators, smallest triple count 1 |

The certificates are Python pickle files. Loading a pickle can run code, so only load files from this repository.

| file | content |
|---|---|
| `proof/thin_S4_N70.txt` | the 70-vertex S_4 tournament of Section 10 (line v lists the out-neighbours of v) |
| `proof/search/r10/cert_s7_N52_balF1.pkl` | Proposition 9.3(a) |
| `proof/search/r10/cert_s7_N52_e2.pkl`, `_e3.pkl`, `_e4.pkl` | Proposition 9.3(b), m = 2, 3, 4 |
| `proof/search/r10/cert_s7_N52_balA.pkl` | (a) with the 4-set condition added |
| `proof/search/r10/classes_7.npz` | the 456 isomorphism classes of 7-vertex tournaments |
| `proof/search/r6/drt51.json` | a doubly regular tournament on 51 vertices |

## Other material in this repository

`proof/` also holds the working notes behind the paper: `ERDOS902_PROOF.md` (Sections 2–6), `F4_GE_52_HAND_PROOF.md`
(Sections 7–8) and earlier session files. `ai-debate/`, `research/` and the `erdos902_debate_*.md` files are earlier
exploratory work. They are kept for the record. The paper is the reference version.
