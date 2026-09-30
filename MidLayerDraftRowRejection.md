# Lossless Mid-Layer Draft-Row Rejection for Speculative Decoding on MoE Models

Research write-up and paper plan. Self-contained; assumes no prior context.

---

## 1. One-paragraph summary

Speculative decoding (SD) on Mixture-of-Experts (MoE) models has a cost problem that dense models do not have: every draft token in the verification pass routes to its own experts, so the pass loads the union of experts across all draft rows. Rejected rows load experts for nothing. On offloaded or on-device MoEs, where expert weights come from CPU memory or flash, this wasted traffic can erase the speedup entirely. We propose a verify-side fix: at a middle layer L of the verifier's own forward pass, a small probe reads the verifier's hidden states and predicts which draft rows will be rejected. Those rows are dropped from layer L onward, so their experts in layers L+1..N are never loaded. The output is unchanged, because rejected rows' logits are never used by the accept rule. A wrong drop costs accepted tokens, never correctness. The method needs no change to the target model, no change to the drafter, and no early-exit training. It stacks on any drafter and any expert-offloading system.

---

## 2. Background and the problem

### 2.1 Speculative decoding
A drafter proposes K tokens. The target model verifies all K in one forward pass, plus one extra position. Tokens are accepted up to the first mismatch (greedy) or by rejection sampling (sampling). The last accepted position's logits give a bonus token. Lossless means the output distribution equals plain autoregressive (AR) decoding.

### 2.2 Why MoE breaks the usual cost model
In a dense model, verifying K+1 rows costs about the same as verifying 1 row at batch size 1, because the weight read is shared. In an MoE, each row picks its own top-k experts per layer. The verify pass must load the union of those experts. Let U_l be the number of unique experts touched at layer l by the pass. The extra expert traffic scales with U_l, not with K.

The Limits-of-Speculation paper (arXiv 2609.22156) measured this on Qwen3-Coder-30B-A3B with an EAGLE-3 drafter on an A100: verification time is predicted almost entirely by the layer-averaged expert union, with one extra unique expert costing about 0.26 ms once the pass has 5+ rows. Even a perfect offline policy for choosing the draft length caps speedup at about 2.34x and accepted length at about 2.1, because the routing cost of low-probability draft tokens outweighs their chance of acceptance. The paper leaves the design of an online early-exit rule as future work.

### 2.3 Our own measurements (Sept 2026, OLMoE-1B-7B, 64 experts, top-8)
Using a training-free mask-token drafter (ESP, arXiv 2603.17942) and a per-layer LRU expert cache simulation with capacity C=16:
- Per-node tree with 4 siblings (10 real rows per pass, tau=1.41): 3.24x the expert bytes per accepted token of plain AR.
- Same run with every mask row deleted (verification rows only): still 2.18x AR.
- The 5 verification rows alone touched about 20 distinct experts per layer per pass, versus 8 for a single AR token.
- An oracle that drops the losing rows' masks from layer 4 of 16 onward cut bytes by 14%.

Two conclusions: (a) rejected verification rows are the dominant waste, and (b) dropping rows mid-pass does reduce expert traffic, and the saving grows with how early the drop happens and how many rows are dropped.

### 2.4 The offloaded regime is where this matters
On a GPU with experts in HBM, one extra expert costs ~0.26 ms. On a phone with experts in flash, one 12.6 MB expert (OLMoE size, bf16) costs tens of milliseconds. The same wasted union that costs 20% on HBM costs the whole speedup on flash. The existing on-device MoE SD systems (SP-MoE, MoE-SpeQ, SpecMoEOff, S2-MoE, DraftExpert) win by prefetching and overlapping expert loads with compute. They hide the traffic. None of them reduces it. Our method reduces it, and stacks on top of theirs.

---

## 3. The method

### 3.1 Setting
- Target: an unmodified MoE model with N layers.
- Drafter: any. The method only needs draft rows to be present in the verifier's forward pass. This includes EAGLE-style chains and trees, mask-token in-path drafters, and native MTP heads.
- Verification pass input: the last accepted token b, plus K draft rows (a chain, or a tree of siblings and children), with the standard tree attention mask so each row attends only to its ancestors.

### 3.2 The probe
At layer L, for each draft row r, the probe reads:
- h_L(r): the row's hidden state at layer L
- h_L(parent(r)): its parent's hidden state
- optionally the row's router logits at layer L and its draft probability
and outputs a score s(r). For a chain, the probe predicts the first rejection position. For a tree, it predicts which sibling under each parent is the accepted one (softmax over siblings plus a "none" slot).

The probe is a 2-layer MLP, a few hundred thousand parameters. It is a passive readout: it does not change any hidden state. So it can be trained offline on cached activations from ordinary SD runs, with the true accept/reject outcome as the label.

### 3.3 The drop
At layer L, rows the probe marks as rejected are removed from the hidden-state tensor, the attention mask, and the position ids. Layers L+1..N run on the smaller set. On a static-shape runtime this is one fixed-size gather (keep the accepted path plus a fixed number of survivors, pad the rest).

### 3.4 Why it is lossless
The greedy accept rule reads logits only along the accepted path: row r's own logits matter only if r is accepted, and the bonus token comes from the deepest accepted row. A rejected row's logits are never read. So dropping a row that would have been rejected changes nothing.

If the probe drops a row that would have been accepted, that row is now missing, so acceptance stops at its parent and the bonus comes from the parent's true logits. The output is still exactly what AR would produce; we only lose the tokens that row and its descendants would have given. Cost: tau. Never correctness.

For sampling-based verification the same holds with rejection sampling along the surviving path; the resample distribution at the first rejected position uses the true target logits of the parent, which are never dropped.

### 3.5 The decision rule
The Limits paper shows that the optimal offline policy reduces to a local rule: continue a row only if its marginal expected cost over its marginal expected accepted tokens stays under a constant slope. We apply this rule mid-pass with better information than any pre-verification method has:
- expected accepted tokens from the probe's accept probability
- expected cost from the row's predicted experts in layers L+1..N (estimated from its router logits at L, or from the average union growth per row measured offline)
Drop the row if cost / expected-gain exceeds the threshold. The threshold is tuned once per model and cache size.

### 3.6 What is new
- Verify-side and mid-pass. All existing MoE-SD cost control (Cascade, EcoSpec, EVICT, MoE-Spec) decides before verification, from draft-side signals. We decide inside the pass, from the target's own states, which are a better acceptance signal (our earlier probe on a dense FlexDraft backbone: 0.68 top-1 at rejection positions vs 0.47 for the drafter).
- No model change. MoE-Spec, AcceptMoE and Sparse Verification cap or skip experts in the target, which is lossy. We only drop rows, which is lossless by construction.
- No early-exit training. HiSpec (arXiv 2510.01336) does intermediate verification on dense models but needs early-exit-trained models and targets throughput. We use a probe on an unmodified target and target expert bytes.
- Stacks on prefetch systems. Fewer rows past layer L means fewer experts to prefetch or load.

---

## 4. Cost model and metrics

Per verification pass:
- rows_in(l): number of real rows at layer l (drops after L)
- U_l: unique experts touched at layer l by the real rows
- LRU cache per layer with capacity C; misses(l) = experts needed at l not in cache; bytes_loaded = sum_l misses(l) * expert_bytes
- accepted: tokens accepted this pass (including the bonus)

Headline metric: bytes per accepted token, reported as a ratio to plain AR under the same cache. Secondary: tau, rows per accepted token, probe accuracy, wall-clock on GPU (HBM regime) as a sanity check only.

Report at C in {8, 16, 32} for a 64-expert model, and at the equivalent fractions for larger models.

---

## 5. Evidence in hand and evidence needed

### 5.1 In hand
- The bound: verification rows alone put ESP at 1.3–2.2x AR bytes on OLMoE; mask rows add another 1x on top. Rejected rows are the waste.
- Mid-pass dropping works: oracle drop from layer 4 of 16 cut bytes by 14% when only mask rows were dropped.
- Mid-layer states carry accept/reject signal: 0.68 vs 0.47 top-1 (dense FlexDraft, earlier work).
- A trained in-path drafter is cheap to make: gated-LoRA mask path (Apple MTP recipe) on Qwen3-1.7B reached 93% depth-1 acceptance in 30 minutes, base path bit-exact.
- A bit-exact greedy harness is buildable in ~1.5 h with known tricks (Section 8).

### 5.2 Needed for the POC (one 5–6 h session, A100 40GB, OLMoE)
- Probe top-1 per layer on a trained drafter, against draft-probability baselines.
- Bytes per accepted token: no drop vs oracle drop vs probe drop, per L, per C.
- tau loss from probe mistakes.
- Online run matching the offline replay, bit-exact output.

Gate: at L = N/2, C = 16: probe top-1 >= 0.75 and > baselines by 0.05; bytes down >= 20%; tau loss <= 5%.

### 5.3 Needed for the paper (80GB card, ~25–30 h)
- Headline pair: Qwen3-30B-A3B (or Qwen3-Coder-30B-A3B) with the public EAGLE-3 SpecForge head. This is the pair the Limits paper calibrated, so the cost model and the 2.34x ceiling are already published reference points.
- Chain K in 1..8 and a small tree; SpecBench subsets (math, code, chat) plus GSM8K.
- Probe per layer; drop rule with the cost-ratio threshold; sweep L and C.
- One real offloaded run (experts on CPU, on-demand load) to show wall-clock, plus the byte simulation across C.
- Second model for generality: OLMoE with the mask-path drafter, or DeepSeek-V2-Lite if an EAGLE head can be trained in budget.

---

## 6. Paper plan

### 6.1 Title direction
"Reject Early, Load Less: Lossless Mid-Layer Draft Rejection for Speculative Decoding on Mixture-of-Experts"

### 6.2 Claims
1. In MoE speculative decoding, rejected draft rows are the dominant source of wasted expert traffic, and the waste scales with the number of rows, not accepted tokens. (Measurement, extends Limits paper to the offloaded regime.)
2. The target model's own mid-layer hidden states predict rejection better than draft-side confidence, early enough to matter. (Probe accuracy per layer vs baselines.)
3. Dropping predicted-reject rows at layer L is lossless and cuts expert bytes per accepted token by X% at <=5% tau loss, on an unmodified target with any drafter. (Main result, two models, two drafters.)
4. The saving stacks with existing prefetch/offload systems and with draft-side budget methods. (Combination experiment with one prefetch policy and one draft-side cap.)

### 6.3 Structure
1. Introduction: the MoE verification cost problem; the offloaded regime; the gap (no verify-side, mid-pass, lossless method).
2. Background and cost model: expert union, LRU bytes, the Limits paper's linear boundary.
3. Method: probe, drop, lossless proof, decision rule, static-shape implementation.
4. Experiments: setup; waste measurement; probe accuracy; main result tables (bytes, tau, wall-clock HBM, wall-clock offloaded); ablations (L, C, probe features, tree vs chain, threshold); stacking with prefetch and with EcoSpec-style draft budgets.
5. Related work: draft-side cost control, verify-side sparsity (lossy), prefetch systems, self-speculation on MoE, intermediate verification on dense models (HiSpec), the Limits paper.
6. Limitations: needs a probe per target model (cheap, offline); gains depend on rejection rate, so very high-acceptance drafters benefit less; sampling-mode verification needs the resample path kept.

### 6.4 Venue
Systems-leaning ML venues where the cost table is the argument: MLSys, EuroSys/ASPLOS workshops, or an EMNLP/ACL industry or efficient-NLP track. Do not pitch against dense-model tau records; the comparison is bytes and tokens/sec on MoE under offloading.

### 6.5 Collision checks to run before writing
- HiSpec (2510.01336): intermediate verification with early-exit models. Confirm it is dense-only and EE-trained.
- "Making Every Verified Token Count: Adaptive Verification for MoE SD" (2605.00342): confirm it prunes the tree before verification, not mid-pass.
- Sparse Verification (2512.21911): expert skipping in verification, lossy. Confirm.
- Search terms to run: "early rejection speculative decoding MoE", "layer-wise verification expert loading", "intermediate verifier mixture of experts", "speculative decoding verification early exit lossless".

---

## 7. Risks and what would kill it

- Probe accuracy too low at any useful L. If rejects only become visible in the last few layers, the saving is small. The dense probe result (0.68 at rejection positions) suggests otherwise, but it has not been measured on a MoE. This is the first thing the POC checks.
- High-acceptance drafters leave little to drop. With EAGLE-3 chains at 70–80% per-token acceptance, most rows survive. The saving then comes from the tail of long chains, which is exactly where the Limits paper says the waste is. Report the saving as a function of K.
- The static-shape gather is awkward on real NPU runtimes. Report the GPU version and the simulated static version; do not claim NPU numbers we cannot measure.
- Wall-clock on HBM may show little gain because 0.26 ms per expert is small. Say so. The claim is bytes, and the offloaded run is where wall-clock moves.

---

## 8. Harness notes (learned the hard way, keep them)

- Greedy equality in bf16 fails unless every decode pass is padded to a fixed row count (32) and attention gathers each query's keys into the same column positions they have in plain AR decoding. The AR reference must use the same kernels and padding. Do not use HF `generate` as the reference.
- Use `inputs_embeds` for the whole step; embed real tokens by hand. Never mix with `input_ids`.
- After a pass, gather the accepted path's KV entries by index and crop; the bonus token has no KV entry and becomes the first row of the next step. Pass `cache_position` explicitly and fix the cache's seen-token counter after a manual gather.
- Routers see padded rows; capture must return only real rows. Test it, including after a prune.
- Expert bytes: three matrices of hidden x intermediate per expert, in the deployed dtype.
- Push to a private GitHub repo before the first model load and after every phase. Source in git; large logs in a tarball copied off the box.

---

## 9. References
- Limits of Speculation: Bounding SD in MoE. arXiv 2609.22156.
- HiSpec: Hierarchical Speculative Decoding. arXiv 2510.01336.
- Making Every Verified Token Count: Adaptive Verification for MoE SD. arXiv 2605.00342.
- MoE-Spec: Expert Budgeting for Efficient SD. arXiv 2602.16052.
- AcceptMoE. arXiv 2608.02989.
- Less Experts, Faster Decoding (EcoSpec). arXiv 2607.12696.
- Utility-Driven SD for MoE (Cascade). arXiv 2506.20675.
- MoESD. arXiv 2505.19645 (NeurIPS 2025).
- Accelerate SD with Sparse Computation in Verification. arXiv 2512.21911.
- SP-MoE. arXiv 2510.10302. MoE-SpeQ. arXiv 2511.14102. SpecMoEOff / hiding offloading latency. arXiv 2508.21706.
- SpecMoE. arXiv 2604.10152. S2-MoE. arXiv 2608.15018. DraftExpert. arXiv 2607.24434.
- Efficient MoE with SD via Expert Coactivation. arXiv 2609.22471.
- EAGLE-3. arXiv 2503.01840. SpecForge head: lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge.
- ESP: Training-Free MTP via Embedding-Space Probing. arXiv 2603.17942.
- Apple MTP (gated LoRA mask path). arXiv 2507.11851.
