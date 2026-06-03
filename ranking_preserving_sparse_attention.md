# Ranking-Preserving Low-Rank Sparse Attention

A research design note. Working title for the method: **CrossCov-SVD** (with cheaper and learned variants).

---

## 1. Motivation

### 1.1 The bottleneck and the pattern

Autoregressive LLM inference is bandwidth-bound: at every generation step the full KV cache must be fetched from memory, and this dominates latency at long sequence lengths. A productive class of methods reduces this cost by computing **approximate attention scores in a reduced space**, using them to select the few tokens that matter, then computing exact attention on only those tokens.

Two representatives:

- **Loki** (PCA on keys): keys empirically lie in a low-rank subspace, so project queries and keys onto the top-$d$ principal components of the key covariance, score in that subspace, select top-$k$, then attend exactly over the selected tokens. Full KV cache retained (no eviction).
- **SparQ** (top-$r$ query magnitude): query components are heavy-tailed, so per step select the $r$ largest-magnitude coordinates of $|q|$, gather those coordinates of $K$, score, select top-$k$, attend exactly. Online, no calibration.

Both share the same skeleton — **cheap approximate scores → top-$k$ token selection → exact attention on the selected full vectors, with all tokens retained** — and both expose the same two compression knobs: the **score-pass width** ($d$ for Loki, $r$ for SparQ) and the **selection budget** $k$.

### 1.2 What speedups Loki and SparQ achieve today

**Loki (compute cost model):**
$$
\text{speedup} = \frac{1}{d_f/2 + k_f}, \qquad d_f = d/D,\; k_f = k/S.
$$
- Recommended $d_f = k_f = 0.25 \Rightarrow$ **2.6×** theoretical.
- Attention-only empirical: **~45% (≈1.8×)** at long prompts (Llama2-13B, prompt 3072).
- End-to-end through HuggingFace: **~9%**, because >80% of the time is KV-cache *append* (would need vLLM / paged attention to unblock). No memory overhead.

**SparQ (bandwidth cost model):** $M_{\text{SparQ}} = Sr + 2kd_h + 4d_h$ vs $M_{\text{dense}} = 2Sd_h + 2d_h$; headline up to **8×** transfer savings.
- Microbenchmark ($r=32, k=128, S=4096$): **3.0×** A100, **4.2×** A10G, **7.4×** IPU.
- End-to-end: **~2.5×** CPU, **~2×** H100 at 1/8 compression and long $S$.
- Costs **+50% memory** (stores $K$ twice); the single-$K$ variant only reaches ~1.3×.

So the realistic, quality-preserving range today is roughly **2.6× theoretical / 1.8–3× microbenchmark / 2–2.5× end-to-end**, capped by *method-agnostic* bottlenecks: KV-cache append, the top-$k$ operation being about as costly as the small matmuls, small-batch parallelism limits, and SparQ's memory tax. A better scorer does not touch any of these.

### 1.3 The research bet — why this work exists

A drop-in scorer replacement **does not change either cost model**: at a matched operating point its theoretical speedup is identical to Loki's or SparQ's. The entire value proposition is therefore the **accuracy–compression frontier** — a scorer with higher top-$k$ recall *at the same width* lets you run at a more aggressive point, and that shift is where realized speedup is created. Two facts make the lever real:

- In Loki's denominator $d_f/2$ is the *smaller* term; the **selection budget $k_f$ dominates**, and both papers' ablations confirm $k$ matters more than $d$/$r$ for quality. So the payoff comes from spending a recall gain on a *smaller selected set* $k$ (Loki) or a *smaller score width* $r$ (SparQ, where the $Sr$ term dominates at long $S$) — not on the score width alone.
- Concretely: matching Loki's quality at half $d_f$ nudges $2.6× \to 3.2×$; matching it at half $k_f$ moves $2.6× \to 4×$ (≈1.5× relative). Energy-SparQ holding quality at $r=16$ vs 32 roughly halves the dominant transfer term, sliding one compression tier toward SparQ's ~2× end-to-end regime.

The defensible claim is therefore **not "faster" in the abstract, but equal speedup at higher fidelity, or equal fidelity at one compression tier deeper** — with speedup as a downstream consequence of a better-aligned scorer. Because the gain is realized through the selection budget, the quantity to optimize is exactly **top-$k$ recall**, which is why §6 makes recall the evaluation currency and §7 gates every early phase on it. This note proposes a better way to build that scorer.

---

## 2. The shared pattern — and the shared flaw

The only job of the cheap scorer is to make `argtopk` of the approximate scores match `argtopk` of the true scores. The relevant metric is **top-$k$ set agreement (recall / Jaccard)**, not score reconstruction.

Yet:

- **Loki optimizes key reconstruction.** PCA minimizes $\mathbb{E}\lVert k - \Pi k\rVert^2$. Preserving key variance is not the same as preserving the attention ranking.
- **SparQ optimizes a query-magnitude proxy.** A large $|q_i|$ is uninformative for *ranking* if the keys barely vary along coordinate $i$ — a near-constant coordinate is an additive offset that does not change `argtopk`.

Placed in a design space, the two differ on three axes: query-adaptivity (SparQ yes, Loki no), basis freedom (SparQ axis-locked, Loki arbitrary rotation but query-agnostic), and objective (both reconstruction-flavored, neither selection-aware). **Neither optimizes the quantity that determines which tokens are selected.**

---

## 3. Core idea

> Choose the reduced subspace to **preserve the attention ranking**, not to reconstruct the keys.

A reduced direction helps token selection only when **two conditions hold at once**: the query carries weight along that direction *and* the keys vary along it (and the two co-occur). Loki checks only the key condition; SparQ checks only the query condition. The natural object that captures *both at once* is the **query–key cross-covariance** $C = \mathbb{E}[q k^\top]$, whose SVD generalizes PCA from "one variable with itself" (variance) to "two variables together" (shared score energy). Its left/right singular vectors give **asymmetric** query and key bases — a degree of freedom both Loki and SparQ discard.

---

## 4. Mathematical formulation

### 4.1 Setup

The score between the current query $q \in \mathbb{R}^D$ and a past key $k \in \mathbb{R}^D$ is the dot product $s = q^\top k$. Reducing dimensionality means projecting both through a matrix $P \in \mathbb{R}^{d\times D}$ ($d < D$) with orthonormal rows. The approximate score is

$$
\hat s = (Pq)^\top (Pk) = q^\top \Pi\, k, \qquad \Pi = P^\top P,
$$

where $\Pi$ is a rank-$d$ orthogonal projector ($\Pi^2=\Pi$, $\Pi^\top=\Pi$).

### 4.2 Score error under projection

With the true coupling being the identity ($s = q^\top I k$),

$$
s - \hat s = q^\top (I - \Pi)\, k .
$$

$(I-\Pi)$ keeps exactly the discarded directions, so **the error is the overlap of $q$ and $k$ in the dropped subspace**. It vanishes unless *both* $q$ and $k$ have weight there.

### 4.3 Why PCA optimizes the wrong objective

Let $R_k = \mathbb{E}[kk^\top]$ (key auto-covariance) and $R_q = \mathbb{E}[qq^\top]$ (query covariance). PCA minimizes key reconstruction

$$
\mathbb{E}\lVert (I-\Pi)k \rVert^2,
$$

solved by the top-$d$ eigenvectors of $R_k$. This shrinks the discarded part of the *key* uniformly across directions — equivalent to assuming the query is equally likely to point anywhere, i.e. $R_q = I$. That is Loki's hidden assumption and its leak: it keeps high-variance key directions even if queries never look there, and discards low-variance directions queries may weight heavily.

### 4.4 The right objective and the cross-covariance

We want the **score** error small over the query–key pairs that occur:

$$
\min_{\Pi}\; \mathbb{E}\big[(q^\top (I-\Pi)k)^2\big].
$$

This depends on the *interaction* of $q$ and $k$. The natural second-order summary of that interaction is the cross-covariance

$$
C = \mathbb{E}[q k^\top] \;\;(\text{generally non-symmetric}),
$$

with a cheap closed-form proxy $C = W_q\, \mathbb{E}[hh^\top]\, W_k^\top$ ($h$ = shared hidden state). $C$ is non-trivial precisely because the model learned $W_q, W_k$ so that related tokens produce aligned $q$ and $k$; that learned alignment is what $C$ measures. Estimate it empirically by averaging $q k^\top$ over scored pairs in a calibration run.

### 4.5 Cross-covariance SVD (headline method)

The tool for a non-symmetric matrix is the SVD (PCA is the special symmetric case). Take

$$
C = U \Sigma V^\top, \quad \text{keep the top } d.
$$

- Project keys with the top-$d$ **left** singular vectors: $P_k = [u_1,\dots,u_d]^\top$.
- Project queries with the top-$d$ **right** singular vectors: $P_q = [v_1,\dots,v_d]^\top$.

By Eckart–Young these are the rank-$d$ query/key bases that best preserve the score's bilinear structure; $\sigma_i$ = "how much ranking signal" lives in the $i$-th query–key direction pair (the analogue of an eigenvalue being "variance").

Two payoffs:

1. **Asymmetric bases.** $C$ is non-symmetric, so $u_i \neq v_i$ and $P_q \neq P_k$ — queries and keys should be reduced in *different* subspaces.
2. **Double condition.** A pair earns large $\sigma_i$ only when queries weight $v_i$ **and** keys weight $u_i$ **and** they co-occur — the formal version of "the question cares about it and the notes differ on it."

Setting $R_q = I$ (collapse $C$ to the key auto-covariance) recovers Loki exactly, so this is a strict generalization with the same inference cost and the same contiguous-slice memory pattern.

### 4.6 Where the clean math stops

The truly correct target (get `argtopk` exactly right) is a ranking objective and a higher-order moment of the joint $(q,k)$ distribution — no clean closed form. CrossCov-SVD is the best *second-order* approximation (preserve scores in expectation, via Eckart–Young), exactly as PCA is the second-order tool for reconstruction. This is why a **learned** variant exists: initialize at the SVD, then fine-tune against a ranking loss to recover what the second-order picture misses.

**Rigor knob:** to make "preserve scores in expectation" an exact Eckart–Young problem, whiten first ($q \mapsto R_q^{-1/2}q$, $k \mapsto R_k^{-1/2}k$) and take the SVD of $R_q^{-1/2} C\, R_k^{-1/2}$, then map back. Raw SVD of $C$ is the simplification valid when $q,k$ are roughly isotropic.

---

## 5. Method variants

### 5.1 CrossCov-SVD projection — headline (closed-form)
Estimate $C$ (empirical or $W_q\mathbb{E}[hh^\top]W_k^\top$), SVD, use $P_q$/$P_k$ = top-$d$ right/left singular vectors. Calibration-light, zero extra inference cost vs Loki, contiguous slicing preserved, asymmetric bases.

### 5.2 Energy-SparQ — cheap, training-free
Replace SparQ's $\text{TopR}(|q|)$ with
$$
I = \text{TopR}\big(|q| \odot \sigma_K\big),
$$
where $\sigma_K$ is the running per-coordinate key standard deviation (post-rotary). A coordinate is useful only if the query weights it *and* keys vary along it. One-line change to SparQ, no training.

### 5.3 Learned Top-K-Preserving projection (TKP) — supervised ceiling
Learn $P_q, P_k$ initialized at the CrossCov-SVD basis, trained on calibration pairs against a top-$k$ ranking objective (prefer a differentiable top-$k$ surrogate — soft-sort or Sinkhorn/OT top-$k$ — over a plain pairwise hinge). **Note the degeneracy:** for scoring only the product $M = P_q^\top P_k$ matters; the SVD factorization fixes the ambiguity and the cache-storage side ($\tilde k = P_k^\top k$) determines $P_k$. Key comparison: does supervision beat its own SVD init?

### 5.4 Mixture of retrieval subspaces — extension
Cluster the query distribution offline, compute one CrossCov-SVD basis per cluster, route each query with a tiny $m$-way dot product. SparQ-style adaptivity with arbitrary rotations and offline amortization.

### 5.5 Uncertainty-adaptive dimension budget — extension
Allocate $d$ per *decision*: coarse score pass, then fetch more dimensions only for candidates near the top-$k$ cutoff where the ranking is ambiguous. Bounded extra cost.

---

## 6. Evaluation currency: top-$k$ recall

The cheap, decisive metric is **top-$k$ recall / Jaccard agreement** between the approximate scorer's selected set and full attention's true top-$k$, measured per layer and head, across sequence lengths. Almost every decision can be made on recall before touching generation, perplexity, or kernels.

---

## 7. Experiment plan (ordered, with go/no-go gates)

Pick **one model first** (Llama2-7B; both baseline papers use it, fits a single A100, full results in ~3 h). Do not run the model zoo until the method works on one. Each early phase has an explicit gate on recall@$k$; do not build kernels or the mixture model until the core recall result is solid.

**Phase 0 — Harness + baseline reproduction.**
Build the rig: for any reduced scorer, compute top-$k$ recall / Jaccard vs true top-$k$ per layer/head across a few $S$. Reproduce Loki recall ($\approx 0.9$ at their settings) and SparQ top-$k$ correspondence.
*Gate:* baselines reproduce, else fix the harness first.

**Phase 1 — Central diagnostic: CrossCov-SVD vs PCA on recall@$k$ (offline).**
Estimate $\mathbb{E}[hh^\top]$, form $C$, SVD, compare recall@$k$ of the cross-covariance subspace vs Loki's key-PCA at matched $d$. Test both **pre- and post-rotary** (Loki showed this swings hard).
*Gate:* CrossCov-SVD $\ge$ PCA on recall, else the principled story is wrong → pivot.

**Phase 2 — Energy-SparQ vs SparQ on recall@$k$ (offline).**
One-line change, calibration-free; runs in parallel with Phase 1. Lowest-effort potential win.
*Gate:* Energy-SparQ $\ge$ SparQ recall at matched $r$.

**Phase 3 — Length-generalization stress test (offline).**
Estimate the static basis on short calibration sequences; evaluate recall@$k$ at long $S$ (e.g. 8k–32k).
*Gate:* decides whether any static linear map (CrossCov-SVD *and* learned TKP) survives RoPE's position dependence. Do this **before** building training infrastructure — if static maps crack here, TKP is doomed.

**Phase 4 — Learned TKP (only if Phase 3 passes).**
Init from CrossCov-SVD, train with a soft-top-$k$ loss. Primary comparison: TKP vs its own SVD init.
*Gate:* if TKP barely beats the init, ship CrossCov-SVD as the headline (a clean result in itself).

**Phase 5 — Downstream quality (winners only) + speedup projection.**
Promote recall winners to WikiText-2 perplexity and LongBench. Do not run these for methods that lost on recall. For each method, find the **most aggressive operating point that matches a fixed quality bar** (smallest $d_f, k_f$ / $r$ holding perplexity and LongBench within tolerance), then plug those into the §1.2 cost models to report a *projected* speedup vs Loki/SparQ at equal quality — this is the headline number the motivation in §1.3 promises, before any kernel work.

**Phase 6 — Latency & bandwidth microbenchmarks (last).**
Kernel work, including the asymmetric $P_q/P_k$ contiguity question. Only for a method already shown to preserve quality. Reuse/extend Loki's Triton kernels.

**Phase 7 — Extensions.**
Mixture of subspaces and uncertainty-adaptive $d$, as a second paper section, only after the core holds.

**Phase 8 — Generalization across the model zoo.**
Repeat the validated pipeline on Llama2-13B, Llama3-8B, Mistral-7B, etc., for the camera-ready breadth.

---

## 8. Baselines, models, datasets, metrics

| Category | Items |
|---|---|
| **Baselines** | Full attention; Exact Top-K (upper bound); Loki PCA (pre/post); SparQ top-$r$; random low-rank projection; CrossCov-SVD (also serves as TKP init) |
| **Candidates** | CrossCov-SVD; Energy-SparQ; Learned TKP; (later) Mixture, Adaptive-$d$ |
| **Models** | Llama2-7B first; then 13B, Llama3-8B, Mistral-7B, Mixtral-8x7B |
| **Calibration** | WikiText-103 (cross-check C4 / BookCorpus for generalization, as Loki) |
| **Recall eval** | per-layer/head top-$k$ recall & Jaccard vs full attention |
| **Quality eval** | WikiText-2 perplexity; LongBench (English tasks) |
| **Efficiency** | attention-only latency, memory-transfer model, end-to-end if time permits |
| **Settings** | match Loki/SparQ budgets, e.g. $k_f \in \{0.25, 0.125\}$, $d_f \in \{0.25, 0.5\}$ / $r$ swept |

Primary proof point: **higher top-$k$ recall at the same $d$** than Loki/SparQ, translating to smaller quality degradation at matched compression.

---

## 9. Risks and open questions

- **RoPE / position dependence (highest risk).** A static linear map cannot represent a relative-position-dependent rotation. Phase 3 is designed to surface this early. Pre- vs post-rotary choice may matter as much as it did for Loki.
- **Train-short / eval-long shift.** Static learned projections are the most overfit-prone of the three; recall must hold at deployment $S$.
- **Factorization degeneracy.** Scoring depends only on $M = P_q^\top P_k$; CrossCov-SVD resolves the ambiguity and supplies the canonical factorization. The cache stores $P_k^\top k$, so $P_k$ conditioning matters.
- **Novelty / prior art.** Learned low-rank KV / retrieval projections are a populated area. The distinctive claims must stay narrow: *ranking-objective* basis, *selection-only* use (full $K$ retained), and *asymmetric* $q/k$ bases from the cross-moment. **Run a focused prior-art pass before committing.**
- **Whitening.** Decide empirically whether whitened cross-covariance beats raw $C$-SVD; report both.

---

## 10. Minimal first milestone

Phases 0–3 on Llama2-7B answer "do we have a paper?" within a few days and require no training infrastructure:

1. Recall harness reproduces Loki/SparQ.
2. CrossCov-SVD beats PCA on recall@$k$ at matched $d$ (pre and post rotary).
3. Energy-SparQ beats SparQ on recall@$k$ at matched $r$.
4. The static basis survives short-train / long-eval.

If (2) and (4) hold, CrossCov-SVD alone is publishable; TKP and the extensions become upside.
