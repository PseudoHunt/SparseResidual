# Handoff: Make CrossCov-U Fast (Section 7.4 of the paper)

**Audience:** a fresh Codex instance with no prior context.
**Repo:** `PseudoHunt/SparseResidual` (private). Paper skeleton: `crosscov_u.pdf`.
**Owner:** Sanjay. Ask him before any change to the calibration statistic or the basis.
**Date:** 2026-09-30.

---

## 0. Read this first

CrossCov-U is a KV-cache *selection* method. It changes one thing about Loki: the offline basis. Instead of key-PCA (eigenvectors of Σk = E[kkᵀ]), it uses the top-r left singular vectors of the query–key cross-covariance C = E[kqᵀ]. For GQA it pools: A = Σ_g C_g C_gᵀ, basis = top eigenvectors of A (paper Theorem 4.14). Everything online (Algorithm 2) is Loki's algorithm verbatim.

**The problem:** the paper says so itself (Section 5.3): same kernels as Loki, so same runtime. Loki has no real speedup. This handoff is about giving CrossCov-U a real one, so Section 7.4 has numbers.

**Do not** change the basis, the calibration, or the accuracy protocol. Only the online path changes.

---

## 1. Why Loki / CrossCov-U are slow (established from the papers)

| # | Cause | Evidence | Fix |
|---|---|---|---|
| 1 | Budget too fat. r/d = 1/4, k/S = 1/8 reads Sr + 2kd = 25% of dense bytes. Ceiling is 4×. | CrossCov-U Table 1; Loki Sec 5 | Evaluate at 1–3% token budgets like Quest/ShadowKV/HATA |
| 2 | Scan reads r fp16 channels per token = 12.5% of K bytes at r=32. | CrossCov-U Table 1 | Store latent channels in a separate 4-bit contiguous buffer |
| 3 | `torch.topk` over all S tokens per head per layer per step. | Loki Fig 7 text | RAFT `select_k` radix top-k (Quest, HATA use it) |
| 4 | Selection is per **query** head. Under GQA (G=4) the union of 4 index sets can be ~half the cache. | CrossCov-U Alg 2 selects per query head; Thm 4.14 pools only the basis | Max-pool latent scores over the group, one index list per KV head |
| 5 | Separate gather of K and V rows into a dense buffer, then attention. | HATA Fig 9: fusing gather into FA2 = 24% of their gain | Use HATA's `flash_index_decode` (FA2 fork with in-kernel gather) |
| 6 | Loki benchmarked vs HuggingFace eager attention at 2–3K prompts. | Loki Sec 6 | Benchmark vs FlashInfer / FA2 dense at 32K–256K |

---

## 2. Target design (online path only)

```
per layer, per decode step, per KV head h:
  q_g  = U_fullᵀ q_g            for each query head g in group   (d×d rotate, O(d²))
  s_g  = K̂_lat[:, :r] · q_g[:r]  scan 4-bit latent buffer         (O(S r))
  s    = max_g s_g               pool over group                  (O(S G))
  T    = radix_topk(s, k)        RAFT select_k                    (~5–10 µs @ <128K)
  y_g  = flash_index_decode(q_g, K_cache, V_cache, T)  gather+attn fused, exact
```

**Memory layout**
- `K_cache`: full rotated keys K̂ = K U_full, fp16/bf16, [S, H_kv, d]. Exact scores need all d rotated coords (paper Lemma 4.2). Unchanged from Loki.
- `K_lat`: first r columns of K̂, per-token-group symmetric 4-bit quant with per-token scale (Double Sparsity label-cache style). [S, H_kv, r/2] bytes + scale. Written once when a key is appended.
- `V_cache`: untouched.

**Byte accounting per token per KV head, d=128, bf16, r=32, budget b = k/S:**
- Dense: 2·128·2 = 512 B
- Scan: 32·0.5 + 2 (scale) = 18 B   (3.5%)  — vs 64 B fp16 (12.5%)
- Selected: b · 512 B
- At b = 1.5%: 18 + 7.7 ≈ 26 B → 19.7× fewer bytes than dense. That's the ceiling; expect 4–6× wall-clock at 128K after top-k + launch overheads.

**Sanity invariants**
- `r = d` and 4-bit disabled and `k = S` must reproduce dense attention to bf16 tolerance.
- Feeding keys on both sides of calibration must reproduce Loki's basis (paper Lemma 4.13). There should already be a test for this; keep it green.
- Recall of the 4-bit scan vs fp16 scan at the same r must be within 0.5 pt (mass recall). If not, try 8-bit before touching r.

---

## 3. Implementation plan

### Step 0 — baseline the current code (1 day)
- Get Loki's Triton kernels running end to end on Llama-3.1-8B-Instruct (GQA, G=4) and Mistral-7B-v0.2.
- Profile one decode step at S ∈ {8K, 32K, 128K}, k/S = 1/8, r = 32, batch 1, on the H100. Use `torch.profiler` + Nsight Compute for DRAM bytes. Record: scan time, topk time, gather time, attention time, total. This is the "before" column.
- **Measure the GQA union size**: for each layer, |∪_g T_g| / S. Log the mean. Prediction: 0.3–0.5. This one number justifies Step 1.

### Step 1 — per-KV-head selection (1 day)
- After the scan, `s = s_g.view(H_kv, G, S).amax(dim=1)`. One top-k per KV head.
- Accuracy check: mass recall and relL2 vs per-query-head selection at the same **total** bytes read (per-KV-head with budget k vs per-query-head with budget k/G·something — report both matched-k and matched-bytes).
- Expected: recall drops slightly at matched k, bytes read drop ~G×. At matched bytes, per-KV-head should win.

### Step 2 — 4-bit latent scan buffer (2–3 days)
- Add `K_lat` as described. Quantize on append. Per-token symmetric int4, scale in fp16.
- Triton scan kernel: reads `K_lat`, dequantizes in registers, dots with `q[:r]` (fp16), writes fp16 scores [H_kv·G, S]. Fuse the group-max into the same kernel (write [H_kv, S]).
- Reference: Double Sparsity's label-cache kernel (arXiv 2408.07092, SGLang port exists) — same shape of problem.
- Accuracy check: recall@k of int4 scan vs fp16 scan, r ∈ {16, 32}. Kill if > 0.5 pt loss at r = 32; fall back to int8.

### Step 3 — radix top-k (1 day)
- Replace `torch.topk` with RAFT `select_k` (pylibraft) or the kernel from Quest's repo (`mit-han-lab/Quest`, RAFT-based). HATA's repo (`gpzlx1/HATA`, `src/topk.cu`) has a self-contained one; easiest to vendor.
- Verify: returned set identical to `torch.topk` (ties aside).

### Step 4 — fused gather + attention (2–3 days)
- Vendor `flash_index_decode(q, K_cache, V_cache, idx, scale)` from HATA (`src/cuda-attn/`, FA2 fork, `idx` shape `[batch, H_kv, topk]`). It already handles GQA by pointing query heads at their KV head's index list.
- Because K_cache is stored rotated (K̂ = K U_full) and q is rotated too, the exact score is q̂ᵀk̂ = qᵀk. No un-rotation needed. Values are not rotated.
- Split-KV variant for batch 1 at long S: HATA has it.
- Verify: output matches dense attention restricted to set T, bf16 tolerance.

### Step 5 — end-to-end benchmark (2 days)
- Configs: S ∈ {32K, 64K, 128K, 256K}; budgets 512, 1024, 2048 tokens (≈1.5% at 128K); r ∈ {16, 32}; batch ∈ {1, 4}; first 2 layers dense (Quest/HATA convention); sink 4 + recent 64 always kept.
- Baselines, all on the same H100, same FlashInfer version:
  - FlashInfer dense decode
  - Loki with the **same** fast path (only the basis differs) — this isolates the accuracy claim, exactly as the paper wants
  - Quest (official repo, page 16, per-head page lists)
  - UNIQUE if code is public by then (arXiv 2605.27740); otherwise reimplement its estimator (page mean + λ‖q‖·std, λ=0.5) on FlashInfer paged decode — it's ~200 lines
  - HATA (official repo)
- Report: attention-step latency vs S; end-to-end tok/s; DRAM bytes (Nsight); mass recall, relL2; RULER-128K and LongBench at matched budget.

---

## 4. Gates and kill conditions

| Gate | Pass | Kill |
|---|---|---|
| G0: GQA union | union ≥ 0.25·S on average | if union < 0.15·S, Step 1 gain is small; still do it, but drop the "GQA explains it" claim |
| G1: per-KV-head recall | ≤ 1 pt mass-recall loss at matched k; wins at matched bytes | loses at matched bytes → revert to per-query-head + shared gather, keep union small via k/G per head |
| G2: int4 scan | ≤ 0.5 pt recall loss at r=32 | > 0.5 → int8; if int8 also fails, keep fp16 and accept 12.5% scan |
| G3: kernel speed | ≥ 4× attention-step over FlashInfer dense at 128K, 2048 budget | < 2× → profile; the scan or topk is wrong. Do not ship a < 2× number |
| G4: accuracy at 1.5% | within 2 pts of Quest on RULER-128K, and beats Loki-same-path | loses to Quest by > 3 pts → the small-r basis isn't enough at this budget; report honestly, raise r to 32 |

---

## 5. What NOT to do

- Do not build page-level bounds in the CrossCov basis. The basis keeps alignment directions and discards key-energy directions, so the residual ‖k⊥‖ is large; the Cauchy–Schwarz term (paper Lemma 4.3) is loosest there. That's a separate research thread, not this task.
- Do not use the latent scores as final attention logits (paper Section 5.5 explains the softmax-flattening failure).
- Do not change r, calibration data, centering (uncentered moments), or pre/post-RoPE choice (post-RoPE default; fit and apply must match).
- Do not mix pre-RoPE basis with post-RoPE keys. Recall goes to the random floor.
- Do not benchmark against HuggingFace eager attention. Reviewers will discount it.

---

## 6. Prior art you must position against (already read; cite these)

- **Quest** (ICML'24, arXiv 2406.10774): full-dim per-channel min/max per 16-token page, exact bound, FlashInfer paged decode. 7× attention at 32K. Per-query-head, no GQA handling, metadata 6.25% of KV.
- **UNIQUE** (Microsoft, May 2026, arXiv 2605.27740): page mean + scalar std, per-KV-head max pooling, radix topk, FlashInfer paged attention. 11.4× attention, 5.3× e2e over vLLM.
- **HATA** (ACL Findings 2025, arXiv 2506.02572): 128-bit learned hash, Hamming scan, fused gather+FA2. 6.5× at 256K. Their optimized Loki (same fusion) ≈ 3×. **Their Loki-with-fusion number is your floor.**
- **ShadowKV** (ICML'25, arXiv 2410.21465): chunk-mean landmarks in full post-RoPE space, low-rank only for storage. ~3× throughput via batch size.
- **SAKI** (Aug 2026, arXiv 2608.03228): score-aware basis from whitened Σq^½·WQᵀWK·Σk^½, token-level, no kernels. This is the paper's Appendix A.3 whitened variant. Cite as concurrent.
- **Locks** (Jul 2026, arXiv 2607.24555), **COBS** (Jul 2026, arXiv 2607.09052): page-local / query-subspace second-order summaries. Different granularity; cite in related work.
- **Double Sparsity** (arXiv 2408.07092): 4-bit label cache of selected channels — the direct precedent for Step 2.
- **DSA** (DeepSeek-V3.2, arXiv 2512.02556): token-level selection is fast only with a trained 128-d FP8 indexer; at 128K the indexer scan is ~93% of remaining attention bytes.

---

## 7. Deliverables

1. `bench/profile_step.py` — one-step profiler with per-stage timing and DRAM bytes; runs before and after.
2. `kernels/scan_int4.py` (Triton), vendored `topk` and `flash_index_decode` with a build script.
3. `results/speed.csv`, `results/accuracy.csv`, and the plots for Section 7.4: attention-step latency vs S (log-log), and recall vs bytes-read Pareto with Quest/UNIQUE/HATA/Loki-same-path.
4. A `RESULTS.md` with the gate outcomes filled in, including the kills.
5. Every number that goes in the paper must be reproducible from one `make bench` on the H100.

---

## 8. Verification checklist for the reviewer (Claude or human)

- [ ] Lemma 4.13 test still green (keys-both-sides == Loki basis).
- [ ] r = d, fp16 scan, k = S reproduces dense to bf16 tolerance.
- [ ] `flash_index_decode` output == dense attention over T (bf16 tol).
- [ ] Radix topk set == torch.topk set.
- [ ] GQA union size logged per layer and reported.
- [ ] All baselines on the same GPU, driver, FlashInfer version, and sequence set.
- [ ] Speed numbers are vs FlashInfer / FA2, not HF eager.
- [ ] Budgets stated in tokens and as % of S.
- [ ] Recall numbers use the paper's mass-recall definition, averaged over heads and positions.
- [ ] Kill conditions checked and outcomes written down, including the ones that fired.

---

## 9. Contacts / open questions for Sanjay

- Which H100 node and CUDA/FlashInfer versions to pin.
- Whether the RULER-128K harness from the RoPE-KV project can be reused.
- Whether Mistral-7B (G=4, d=128) or Llama-3.1-8B is the headline model for Section 7.4.
