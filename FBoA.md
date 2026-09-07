# Factorized BoA — attention-output-aware quantization for models with every layer as `W = B·A`

**Object:** a transformer where each linear layer is stored as `W = B A`, `A ∈ ℝ^{r×d_in}`, `B ∈ ℝ^{d_out×r}`, initialised by SVD (Σ folded into `B`) and then trained. No operation between `A` and `B` (see §10 if there is one).
**Goal:** quantize both factors of every layer under BoA/TurboBoA's objectives — attention-score error for q/k, attention-output error for v through `W_o`, layer-output error elsewhere — without new forward passes, and remove the Σ-induced outliers by construction rather than by mixed precision.

---

## 1. The objective for a factor pair, and why it stays Kronecker

BoA's loss for a dense layer with perturbation `ΔW` is the two-sided quadratic
```
L(ΔW) = tr( H_out · ΔW · H_in · ΔWᵀ )
```
with `H_in` the input covariance and `H_out` the row metric (key covariance for q, query covariance for k, `W_oᵀW_o` per head for v, `I` for o and MLP).

For a factor pair, quantize the inner factor first. Exact decomposition:
```
ΔW = B̂Â − BA = B̂·ΔA + ΔB·A          (ΔA = Â−A, ΔB = B̂−B)
```
**Inner factor with `B` held fixed:** `ΔW = B ΔA`, so
```
L_A(ΔA) = tr( (Bᵀ H_out B) · ΔA · H_in · ΔAᵀ )
```
The inner factor is a two-sided problem with `H_in^A = H_in` and an **`r×r` row metric `H_out^A = Bᵀ H_out B`**. This holds for *every* layer, including the ones BoA treats one-sided: with `H_out = I`, `H_out^A = BᵀB`, which is far from identity whenever Σ sits in `B` (`BᵀB ≈ Σ²`). Plain GPTQ on `A` ignores exactly the anisotropy that Σ created.

**Outer factor with `Â` fixed:** `B`'s input is `z = Â x`, so `H_in^B = Â H_in Âᵀ` (`r×r`, computed from `H_in` — no forward pass) and `H_out^B = H_out` as in BoA. The error `Â` already injected is handled in §4.

Both problems are single Kronecker products, so BoA's/TurboBoA's two-sided solver runs unchanged on each. Per-head `H_out` collapses for the inner factor: `Bᵀ H_out B = Σ_h B_hᵀ H_out,h B_h`, one `r×r` matrix, no head loop.

## 2. The gauge, and the theorem that removes the outliers

`(B, A)` and `(B G⁻¹, G A)` are the same layer for any invertible `G ∈ ℝ^{r×r}`. Quantization is not gauge-invariant. Two facts fix `G`:

**Fact 1 — the inner factor's loss is invariant to diagonal gauges.** With per-row scales on `A`, scaling row `i` by `d_i` scales its quantization error by `d_i` and its row-metric weight `(Bᵀ H_out B)_ii` by `1/d_i²`; the product is unchanged. So a diagonal gauge should be spent **entirely on the outer factor**. Σ-in-`B` is the worst diagonal gauge: it puts the one thing per-row scales can absorb (row scale) on the side where it appears as *column* scale, which per-row scales cannot absorb.

**Fact 2 — there is a gauge that makes the inner problem exactly one-sided.** Choose `G` with `GᵀG = Bᵀ H_out B`, i.e. `B̃ = B G⁻¹` satisfies `B̃ᵀ H_out B̃ = I_r`. Then `H_out^A = I` and the inner factor is a plain GPTQ problem *exactly*, while the outer factor `B̃` has `H_out`-orthonormal columns — entries of uniform scale, no Σ, no outliers by construction. All of Σ (and whatever training did to it) moves into `Ã = G A`, whose rows are quantized with per-row scales that absorb the diagonal part of `G`.

Three square roots to ablate:
- **polar** `G = (Bᵀ H_out B)^{1/2}` (symmetric; closest to identity, least mixing of `A`'s rows);
- **Cholesky** `G = chol(Bᵀ H_out B)ᵀ` (triangular; cheapest);
- **diagonal only** `G = diag(‖H_out^{1/2} B_{:,i}‖)` (the Σ-to-`A` fix; leaves `H_out^A` with off-diagonals but removes the scale imbalance).

Recommendation: polar as default; diagonal as the cheap baseline; a latent Hadamard after polar (`G ← H G`) as the incoherence ablation.

## 3. Per-layer metrics

Notation: `x` = block input (post-norm) for q/k/v; head `h` selects rows of `B`. GQA: kv heads as in BoA (query covariances averaged over shared heads).

| layer | inner factor `A` | outer factor `B` |
|---|---|---|
| **q** | `H_in = E[xxᵀ]`, `H_out^A = Σ_h B_q,hᵀ K_h B_q,h` where `K_h` = BoA's rotated key covariance (or the length-extrapolated `K_h(w_L)`) | `H_in^B = Â_q H_in Â_qᵀ`, `H_out^B_h = K_h` |
| **k** | same with query covariances `Q_h` and kv-head grouping | `H_out^B_g = Σ_{h∈g} Q_h` per kv head |
| **v** | shared across heads; exact loss is `Σ_h (H_in,h) ⊗ (B_v,hᵀ M_h B_v,h)` with `H_in,h = E[(A_h x)(A_h x)ᵀ]` (block_v) — a **sum of Kronecker products** (§5) | `H_in^B_h = Â_v H_in,h Â_vᵀ`, `H_out^B_h = M_h` |
| **o** | `H_in = E[attn_out attn_outᵀ]`, `H_out^A = B_oᵀ B_o` | `H_in^B = Â_o H_in Â_oᵀ`, `H_out = I` (one-sided) |
| **up / gate / down** | `H_out^A = BᵀB`, `H_in` as GPTQ | one-sided, `H_in^B = Â H_in Âᵀ` |

`M_h` is v's exact output metric through the *factorized* `W_o`: `W_o,h = B_o A_o,h` (columns of `A_o` for head `h`), so `M_h = A_o,hᵀ B_oᵀ B_o A_o,h` — `d_h × d_h`, static, exact.

After the §2 gauge, every `H_out^A` in the table becomes `I` for q, k, o, up, gate, down (v needs §5). The two-sided solves that remain are the outer factors, whose row metrics are BoA's own.

## 4. Exact within-layer error propagation (the factorized GPTAQ)

After `Â` is fixed, the outer factor's target is not "`B` applied to the same input" — it is `B A x`, while its input is `Â x`. The best FP outer factor for that input is the least-squares refit
```
B* = B · A H_in Âᵀ · (Â H_in Âᵀ)⁻¹          (r×r solve; exact under the H_in metric)
```
Quantize `B*` — not `B` — with `H_in^B = Â H_in Âᵀ` and `H_out`. This compensates `Â`'s error in the span of `Â`'s outputs exactly and costs one `r×r` solve. It is TurboBoA's `consider_dX` mechanism applied inside a layer, with the residual term computed in closed form (`dXXᵀ = −ΔA H_in Âᵀ`) rather than by an extra FP/quantized forward pass, and with no `alpha` heuristic: the refit is the exact optimum.

Optional: one alternation — refit `A* = argmin ‖B̂ A* − B A‖_{H_out,H_in}` given `B̂`, re-quantize, refit `B`. CALDERA's alternating structure under BoA's metric; stop when the exact loss stops decreasing.

## 5. The shared inner value factor

`A_v` is one matrix but each head weighs its input differently (`H_in,h`). Two options:

- **Pooled (drop-in):** `H_in^{A_v} ≈ mean_h H_in,h`, `H_out^{A_v} = Σ_h B_v,hᵀ M_h B_v,h`, then the §2 gauge. This is the same approximation BoA already makes for GQA (`.mean(dim=1)`), applied across heads.
- **Exact (cheap here):** the loss is a sum of `n_heads` Kronecker products over an `r × d_in` weight. The coordinate-descent solver spec'd earlier (implicit Hessian, column-block updates) handles a sum of Kronecker terms directly; with `r ≪ d_out` this is the one place in the model where that solver is both needed and cheap. Use pooled as init, CD to refine, report the gap — that is the ablation that justifies the solver.

## 6. What carries over from TurboBoA unchanged

Joint out-channel quantization (rows of `B̃`, rows of `Ã`), adaptive grid and coordinate-descent scale refinement, cross-*block* `consider_dX` (block inputs from the quantized prefix), act-order on both axes. The only new components are the `r×r` algebra of §1–4 and the gauge.

## 7. Algorithm (per block, same layer order as BoA: k, v, q, o, up/gate, down)

```
for each layer (A, B) with metrics (H_in, H_out):
  1. G ← sqrt(Bᵀ H_out B)  [polar | chol | diag]           # gauge
     Ã ← G A ;  B̃ ← B G⁻¹                                   # now B̃ᵀ H_out B̃ = I
  2. Â ← quantize(Ã ; H_in, H_out^A = I)                     # one-sided GPTQ/TurboBoA row-joint
     (v only: pooled or sum-of-Kronecker CD, §5)
  3. B* ← B̃ · Ã H_in Âᵀ (Â H_in Âᵀ)⁻¹                        # exact refit to Â
  4. B̂ ← quantize(B* ; H_in^B = Â H_in Âᵀ, H_out)           # two-sided where H_out ≠ I
  5. store (Â, B̂) as int codes + scales; layer output = B̂(Âx)
propagate quantized block outputs to the next block as BoA/TurboBoA do.
```
No forward passes beyond the ones BoA already makes. Every new matrix is `r×r` or `r×d`.

## 8. Gates and ablations, in the order to run them

1. **Correctness (toy, CPU):** `H_out^A = Bᵀ H_out B` against finite differences of the exact loss; refit `B*` strictly reduces the exact loss vs quantizing `B`; diagonal-gauge invariance of the inner loss (Fact 1) numerically; gauge-of-§2 gives `H_out^A = I` to 1e-10.
2. **Deletion test on the team's checkpoint, RTN + GPTQ, W4/W3:** as-trained gauge vs diagonal vs polar. This is the experiment that decides whether the outlier problem is the gauge (expected) or something else. One afternoon.
3. **Does the inner row metric matter once the gauge is fixed?** Compare inner-factor quantization with `H_out^A = BᵀH_outB` (no gauge, two-sided) vs gauge-then-one-sided. They optimise the same objective; if PPL differs, it is the grid/clipping behaviour of the two parametrisations, which is the empirical content of Fact 2.
4. **Refit on/off** (§4) — the within-layer propagation's contribution.
5. **Attention metric on/off** for the outer factors of q/k/v — the BoA-vs-GPTQ question, now on factorized layers. Our dense results say q's metric is dispensable at W3 and k's is not; see whether that survives factorization.
6. **Pooled vs CD** for `A_v` (§5).
7. Three seeds, KL-to-FP alongside PPL, zero-shot on stored checkpoints from a cheap card.

## 9. Cost

Gauge: one `r×r` square root per layer. Refit: one `r×r` solve. Inner solves: `r×d_in` with identity row metric — cheaper than dense GPTQ on `W`. Outer solves: `d_out×r` with BoA's row metric and an `r×r` input Hessian — much cheaper than dense BoA. Net: faster than BoA on the dense model, no additional calibration passes.

## 10. If something sits between `A` and `B`

- **Norm or nonlinearity `φ`** between the factors (`y = Bφ(Ax)`): the gauge collapses to positive diagonal (and permutations); §2's theorem is lost; the inner row metric becomes `E_t[D_t Bᵀ H_out B D_t]` with `D_t = diag(φ'(A x_t))` — the gate-pooling structure from Phase 4, handled by the pooled Hadamard form `(BᵀH_outB) ⊙ E[d dᵀ]`. Everything else in §3–7 survives.
- **Residual or bias** between the factors: bias is fine (fold as usual); a residual branch means `W = BA + I`-type structure — the inner metric acquires an identity term; workable, ask first.

## 11. Claims this supports, if the gates pass

1. For a factor pair under any two-sided quadratic objective, there is a gauge that makes the inner problem exactly one-sided and the outer factor `H_out`-orthonormal; Σ-in-`B` is the gauge that maximises the outer factor's column-scale imbalance.
2. Within-layer error propagation for a factor pair has a closed form (the refit), removing TurboBoA's residual-weight heuristic for this case.
3. BoA's attention-score and attention-output objectives extend to factorized q/k/v exactly, through `Bᵀ H_out B` and the factorized `W_o`, with no new calibration cost.
4. The shared value factor is the one place a sum-of-Kronecker solver is both necessary and cheap.

Positioning: PivGa names the gauge for parameter count; Delta-CoMe/LoRAQuant/IO-SVD answer Σ-scale with mixed precision; CALDERA quantizes correction factors with random equalization; LatentLLM is attention-aware truncation; GPTQ-intrinsic LoRA augments the Hessian for `Q+LR`. None has the object, the gauge-from-objective, the refit, or the chain metric.
