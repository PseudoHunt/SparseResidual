# SVD-Based KV Cache Compression via Factor Absorption

## Full Derivation with All Intermediate Steps

---

## 1. Notation and Baseline Attention

Consider one attention head. Dimensions:

- $d$ — model (hidden) dimension
- $d_h$ — head dimension
- $r$ — chosen SVD rank, with $r < d_h$
- $x_t \in \mathbb{R}^{1 \times d}$ — hidden state of token $t$ (row vector convention)
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$ — per-head projections
- $W_O \in \mathbb{R}^{d_h \times d}$ — the slice of the output projection belonging to this head

Standard attention for a query token $x_q$ attending over past tokens $t = 1 \dots T$:

**Step 1.1 — projections:**

$$q = x_q W_Q, \qquad k_t = x_t W_K, \qquad v_t = x_t W_V$$

**Step 1.2 — scores and weights:**

$$s_t = \frac{q\, k_t^\top}{\sqrt{d_h}}, \qquad a_t = \frac{e^{s_t}}{\sum_{j=1}^{T} e^{s_j}}$$

**Step 1.3 — head output:**

$$o = \Big(\sum_{t=1}^{T} a_t v_t\Big) W_O$$

**Baseline cache cost:** we store $k_t, v_t \in \mathbb{R}^{d_h}$ for every token, so $2 d_h$ values per token per head.

---

## 2. Low-Rank Factorization of the Projections

**Step 2.1 — exact SVD.** Any matrix $W \in \mathbb{R}^{d \times d_h}$ has the decomposition

$$W = U \Sigma V^\top, \qquad U \in \mathbb{R}^{d \times d},\ \Sigma \in \mathbb{R}^{d \times d_h},\ V \in \mathbb{R}^{d_h \times d_h}$$

with $U, V$ orthogonal and $\Sigma$ diagonal with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.

**Step 2.2 — rank-$r$ truncation.** Keep the top $r$ singular values:

$$W \approx U_r \Sigma_r V_r^\top, \qquad U_r \in \mathbb{R}^{d \times r},\ \Sigma_r \in \mathbb{R}^{r \times r},\ V_r \in \mathbb{R}^{d_h \times r}$$

By the Eckart–Young theorem this is the best rank-$r$ approximation of $W$ in Frobenius and spectral norm.

**Step 2.3 — split into two factors.** Group the terms as

$$A = U_r \Sigma_r \in \mathbb{R}^{d \times r}, \qquad B = V_r^\top \in \mathbb{R}^{r \times d_h}$$

so that $W \approx AB$. (Any split works, e.g. $U_r \Sigma_r^{1/2}$ and $\Sigma_r^{1/2} V_r^\top$; the algebra below does not depend on the choice.)

Apply this to both cache projections:

$$W_V \approx A_V B_V, \qquad W_K \approx A_K B_K$$

**Step 2.4 — define the cached latents.** Instead of $k_t$ and $v_t$, store

$$c_t^K = x_t A_K \in \mathbb{R}^{1 \times r}, \qquad c_t^V = x_t A_V \in \mathbb{R}^{1 \times r}$$

The original vectors can always be recovered (approximately) as

$$k_t \approx c_t^K B_K, \qquad v_t \approx c_t^V B_V$$

The whole point of the next two sections is to show that we never need to perform this reconstruction explicitly.

---

## 3. V-Side: Absorbing $B_V$ into $W_O$

**Step 3.1 — start from the head output:**

$$o = \Big(\sum_{t} a_t v_t\Big) W_O$$

**Step 3.2 — substitute the factorized value:**

$$o = \Big(\sum_{t} a_t \,(c_t^V B_V)\Big) W_O$$

**Step 3.3 — pull the scalar $a_t$ inside.** Since $a_t$ is a scalar, $a_t (c_t^V B_V) = (a_t c_t^V) B_V$:

$$o = \Big(\sum_{t} (a_t c_t^V)\, B_V\Big) W_O$$

**Step 3.4 — factor $B_V$ out of the sum.** Matrix multiplication distributes over addition, and $B_V$ is the same for every $t$:

$$\sum_t (a_t c_t^V) B_V = \Big(\sum_t a_t c_t^V\Big) B_V$$

so

$$o = \Big(\sum_{t} a_t c_t^V\Big) B_V\, W_O$$

**Step 3.5 — use associativity to merge the two constant matrices:**

$$o = \Big(\sum_{t} a_t c_t^V\Big) \big(B_V W_O\big) = \Big(\sum_{t} a_t c_t^V\Big) W_O'$$

**Step 3.6 — define the folded weight (computed once, offline):**

$$\boxed{\,W_O' = B_V W_O \in \mathbb{R}^{r \times d}\,}$$

**Result.** At inference time the value path is: attention weights $a_t$ hit the cached latents $c_t^V$ directly, and the weighted sum (an $r$-dimensional vector) goes straight through $W_O'$. The full $v_t \in \mathbb{R}^{d_h}$ is never materialized.

**Multi-head note.** With $H$ heads, the model-level output projection is applied to the concatenation of head outputs. The absorption is done per head: build the block matrix

$$W_O' = \mathrm{blockdiag}(B_V^{(1)}, \dots, B_V^{(H)})\; W_O^{\text{full}} \in \mathbb{R}^{Hr \times d}$$

so the concatenated latent contexts $\big[\sum_t a_t^{(1)} c_t^{V,(1)} \,\|\, \dots \,\|\, \sum_t a_t^{(H)} c_t^{V,(H)}\big] \in \mathbb{R}^{1 \times Hr}$ multiply it directly.

---

## 4. K-Side: Absorbing $B_K$ into $W_Q$

**Step 4.1 — start from the raw score:**

$$s_t \sqrt{d_h} = q\, k_t^\top$$

**Step 4.2 — substitute both factorized forms:**

$$q\, k_t^\top = (x_q W_Q)\,\big(c_t^K B_K\big)^\top$$

**Step 4.3 — apply the transpose rule** $(AB)^\top = B^\top A^\top$:

$$\big(c_t^K B_K\big)^\top = B_K^\top (c_t^K)^\top$$

so

$$q\, k_t^\top = x_q W_Q\, B_K^\top\, (c_t^K)^\top$$

**Step 4.4 — regroup by associativity:**

$$q\, k_t^\top = x_q\, \big(W_Q B_K^\top\big)\, (c_t^K)^\top$$

**Step 4.5 — define the folded query projection (computed once, offline):**

$$\boxed{\,W_Q' = W_Q B_K^\top \in \mathbb{R}^{d \times r}\,}$$

**Step 4.6 — the new score computation:**

$$q' = x_q W_Q' \in \mathbb{R}^{1 \times r}, \qquad s_t = \frac{q'\, (c_t^K)^\top}{\sqrt{d_h}}$$

**Result.** The query is projected directly into the rank-$r$ latent space and dotted against the cached $c_t^K$. The full $k_t$ is never materialized. Note the softmax scale stays $\sqrt{d_h}$ — the score is (approximately) the same number as before, only computed through a different factorization, so nothing about the temperature changes.

---

## 5. Cost Accounting

Per token, per head:

| Quantity | Baseline | Compressed |
|---|---|---|
| Cache size | $2 d_h$ | $2r$ |
| K/V projection FLOPs | $2\, d\, d_h$ | $2\, d\, r$ |
| Query projection FLOPs | $d\, d_h$ | $d\, r$ |
| Score dot product | $d_h$ per past token | $r$ per past token |
| Value weighted sum | $d_h$ per past token | $r$ per past token |
| Output projection | $d_h \cdot d$ | $r \cdot d$ |

Everything scales by $r / d_h$. Compression ratio of the cache is exactly $r / d_h$; e.g. $r = d_h/4$ gives a 4× smaller KV cache **and** cheaper attention arithmetic. This is why the trick is strictly better than post-hoc compressing $k_t, v_t$: the reconstruction cost is folded away instead of paid at runtime.

Memory-bandwidth view (the one that matters for decode): decode is bound by reading the cache, so token latency at long context improves roughly by the same $r/d_h$ factor.

---

## 6. Why RoPE Breaks the K-Side (and What To Do)

**Step 6.1 — where RoPE sits.** RoPE applies a position-dependent orthogonal rotation $R_t \in \mathbb{R}^{d_h \times d_h}$ *after* the projection:

$$\tilde{q} = x_q W_Q R_q, \qquad \tilde{k}_t = x_t W_K R_t$$

**Step 6.2 — try the same substitution:**

$$\tilde{q}\, \tilde{k}_t^\top = x_q W_Q R_q \big(c_t^K B_K R_t\big)^\top = x_q W_Q\, \underbrace{R_q R_t^\top}_{\text{depends on both positions}}\, B_K^\top (c_t^K)^\top$$

Wait — careful with ordering. Applying the transpose rule:

$$\big(c_t^K B_K R_t\big)^\top = R_t^\top B_K^\top (c_t^K)^\top$$

so

$$\tilde{q}\, \tilde{k}_t^\top = x_q\, W_Q\, R_q\, R_t^\top\, B_K^\top\, (c_t^K)^\top$$

**Step 6.3 — the obstruction.** To absorb $B_K^\top$ into the query weights we would need to move it left past $R_q R_t^\top$. But $R_t$ changes with the position of token $t$, so the merged matrix $W_Q R_q R_t^\top B_K^\top$ would be different for every key position — there is no single constant $W_Q'$ anymore. Rotation matrices do not commute with an arbitrary $B_K$, so the factor is stuck in the middle.

(The V side is unaffected: RoPE is never applied to values, so Section 3 stays exact.)

**Step 6.4 — standard workarounds.**

1. **Decoupled RoPE (MLA's solution).** Split the key into two parts: a compressed, position-free part $c_t^K$ (rank $r$, absorbed as in Section 4) plus a small dedicated RoPE sub-vector $k_t^{\text{rope}} \in \mathbb{R}^{d_r}$ cached as-is. The score is the sum of the two dot products. You pay $d_r$ extra cache dims but keep everything exact.
2. **RoPE on the latent.** Apply the rotation to $c_t^K$ directly. Cheap, but no longer mathematically equivalent to the original model — needs fine-tuning or an accuracy check.
3. **Restrict to non-RoPE paths.** NoPE models, ALiBi-style models (bias added to scores, not rotations), or the value-only variant everywhere.

---

## 7. Practical Improvements Over Plain SVD

**7.1 — Joint SVD across grouped heads.** Instead of factorizing each head's $W_K^{(h)}$ separately, stack $G$ heads' projections into $W \in \mathbb{R}^{d \times G d_h}$ and take one SVD. Heads share redundant subspaces, so a joint rank budget compresses meaningfully better at equal quality (this is the core finding of Palu).

**7.2 — Activation-aware (whitened) SVD.** Plain SVD minimizes $\|W - AB\|_F$, but the quantity that matters is the output error $\|XW - XAB\|_F$ on real inputs $X$. Fix: compute a whitening factor $S$ from the input covariance ($S S^\top \approx \mathbb{E}[x^\top x]$, e.g. by Cholesky), factorize $SW$ instead, and un-whiten:

$$SW \approx \tilde{A}\tilde{B} \;\Rightarrow\; W \approx (S^{-1}\tilde{A})\,\tilde{B}$$

This is the SVD-LLM / ASVD recipe and typically buys a large accuracy margin at the same rank.

**7.3 — Per-layer rank allocation.** Sensitivity to rank truncation varies a lot across layers (early layers usually tolerate less compression on K). Allocate $r$ per layer from a Fisher- or perturbation-based sensitivity score under a global cache budget rather than using one uniform rank.

**7.4 — Composition with quantization.** The latents $c_t^K, c_t^V$ can additionally be quantized (e.g. to 4 or 3 bits). Low-rank and quantization errors are roughly independent, so the two multiply into the total compression ratio.

---

## 8. Summary of the Final Inference-Time Algorithm

Offline (once):

1. $W_K \approx A_K B_K$, $W_V \approx A_V B_V$ (whitened, joint over head groups)
2. $W_Q' = W_Q B_K^\top$
3. $W_O' = \mathrm{blockdiag}(B_V^{(h)})\, W_O$

Per new token $x_t$:

1. Cache $c_t^K = x_t A_K$ and $c_t^V = x_t A_V$ (that's the entire KV cache)

Per query $x_q$:

1. $q' = x_q W_Q'$
2. $s_t = q' (c_t^K)^\top / \sqrt{d_h}$, then softmax $\to a_t$
3. $o = \big(\sum_t a_t c_t^V\big) W_O'$

No $k_t$ or $v_t$ is ever formed. Cache and attention arithmetic both shrink by $r / d_h$.
