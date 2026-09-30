# CODEX_TASK: Lossless mid-layer draft-row rejection on an MoE (OLMoE POC)

Self-contained brief. Assume zero prior context. Budget: one session of 5–6 hours on an A100 40GB. Plan on that from minute one.

## 0. Rule zero: the repo

Before loading any model: `git init`, create a **private** GitHub repo under `8BitSpacemanSpiff`, push. Commit and push at the end of every phase below, and every time a test goes green. Source, scripts, results JSON, RESULTS.md and small logs all go in git. Large logs go in a tarball that is uploaded to the repo's releases or copied off the box before the session ends. Nothing lives only on the box.

## 1. Goal

Prove one mechanism on a small MoE: at a middle layer L of the verifier's forward pass, a cheap probe on the verifier's own hidden states can tell which draft rows will be rejected. Dropping those rows from layer L onward saves their expert loads in the deeper layers. Output is unchanged, because rejected rows' logits are never used.

Deliverables: probe accuracy per layer, expert bytes per accepted token with and without the drop, τ with and without the drop, all with output bit-identical to plain greedy decoding.

This POC does not need to show that speculation beats plain decoding. It needs to show the drop is lossless and the saving is real.

## 2. Hard constraints

- Greedy decoding only. Every run asserts token-for-token equality with plain autoregressive greedy from the same model and kernels.
- Batch size 1. HF `transformers`, raw model access, `inputs_embeds` path. No vLLM/SGLang.
- bf16. All decode-step forward passes are padded to a fixed 32 rows (see 5.1).
- The only trained modules: one mask embedding vector, a gated LoRA on the mask path, and a small probe. Base weights never change.

## 3. Model and data

- Model: `allenai/OLMoE-1B-7B-0125-Instruct`. Read `config.json` for layer count (expect 16), expert count (expect 64), top-k (expect 8). Confirm `output_router_logits=True` returns per-layer router logits; otherwise hook the gate modules.
- Data: GSM8K test, first 30 prompts for logging runs, first 100 for the final τ table if time allows. `max_new_tokens=128`. Chat template.
- Training data for the mask path: GSM8K train (all) + 3k UltraChat_200k samples, first two turns, truncated to 1024 real tokens.

## 4. Drafter: gated-LoRA mask path (per-node, k=1)

### 4.1 Decoding layout
State: accepted prefix in the KV cache, last accepted token `b`.

Step input rows: `[b, c_1..c_W, m_b, m_{c_1}..m_{c_W}]` where `c_i` are the W draft candidates (siblings, all children of `b`) and `m_x` is a mask row attached after node `x`. W=4, so 10 real rows.

Attention: candidates attend to cache + `b` + themselves. Mask `m_x` attends to cache + path to `x` + itself. Siblings never attend to each other. Masks never attend to other nodes' masks. Real tokens never attend to masks. Build as one 4D additive mask.

Position ids: `b` at cache_len; all siblings at cache_len+1; `m_b` at cache_len+1; `m_{c_i}` at cache_len+2.

Verification: `c_i` is accepted iff `argmax(logits at b) == c_i`. At most one sibling is accepted. Bonus token = argmax at the accepted sibling, or at `b` if none accepted. Accepted count per pass = 1 or 2. Next step's candidates = top-W of the mask attached to the accepted node (or `m_b` if none).

KV cache after the pass: keep cache + `b` + accepted sibling (gather by index), crop everything else. Pass `cache_position` explicitly. Fix the cache's seen-token counter after any manual gather.

### 4.2 Mask embedding and gated LoRA
- One learned mask vector, init = mean of the embedding table + N(0, 0.02).
- LoRA rank 16, alpha 32, on q/k/v/o and the MoE gate-input projections only (skip expert weights; they are the memory we are measuring). Gate: multiply the LoRA delta by a per-row gate that is 1 on mask rows, 0 on real rows. Real rows compute exactly the base model.
- Gate test before and after training: real-token logits equal the base model bit-exactly on 10 prompts.

### 4.3 Training (target: 40 min)
- Block layout: `[x_1..x_n, m_1..m_n]`, `m_i` at position i+1, attends to `x_1..x_i` and itself. Loss: CE of `lm_head(h(m_i))` vs `x_{i+2}` on assistant positions. No other loss.
- AdamW lr 2e-4, cosine, 100 warmup, ~16k real tokens per step, as many steps as fit in 40 minutes. Checkpoint (LoRA + mask vector only) every 10 minutes; log held-out depth-1 top-1 accuracy at each checkpoint. Stop early if the curve is flat for two checkpoints.

## 5. Harness details that were hard-won last time

### 5.1 bf16 determinism
Greedy equality fails in bf16 unless: (a) every decode pass is padded to a fixed 32 rows so all GEMMs run at one shape; (b) attention gathers each query's keys into the same column positions they would have in plain AR decoding, so tree side-branches do not change rounding. Do both. The AR reference loop must use the same kernels and the same 32-row padding. Do not use HF `generate` as the reference.

### 5.2 Router capture
Routers see the padded rows (they must, for fixed GEMMs), but capture returns only real rows. Write a test that asserts padding rows are excluded from every union count, cache simulation, and probe record, including after a prune.

### 5.3 Expert cost simulation
- Per layer, per row: top-k expert ids.
- `union[l]` = unique experts across real rows at layer l. Split into `union_root`, `union_siblings`, `union_masks`.
- Per-layer LRU cache with capacity C ∈ {8, 16, 32}. Misses per pass × expert bytes (three matrices of hidden×intermediate in bf16, from config) = `bytes_loaded`. Update after the pass.
- AR reference: same simulation on the plain greedy run. Report ESP/AR ratio of bytes per accepted token.

### 5.4 Probe records
Per pass: hidden states at layers {4, 5, 8, 11} for `b` and each sibling (fp16), which sibling was accepted (or none), each sibling's draft probability, its router top-k at those layers. Grouped per pass, per prompt.

## 6. The mechanism test

### 6.1 Probe
Input per sibling: `[h_L(sibling), h_L(b)]`. Output: score. Softmax over the W siblings plus a "none accepted" slot. 2-layer MLP, width 512, AdamW 1e-3, 5 epochs. Train on 24 prompts, test on 6, split by prompt.

Baselines the probe must beat: (a) always pick top draft-probability sibling, (b) draft_prob (score = p(sibling) × p(no deeper accept), which here reduces to p(sibling)).

Report top-1 at each L.

### 6.2 Offline replay (exact, no forward passes)
From layer L onward, keep `b` + the probe's chosen sibling + that sibling's mask; drop the other siblings and their masks. Recompute `union`, LRU misses and `bytes_loaded` from the logs. Dropped rows do not change surviving rows (they never attend to each other), so the replay is exact.

Recompute τ under probe drops: a wrong pick loses the accepted sibling for that pass, so that pass yields 1 token instead of 2.

Report oracle drop (true sibling) and probe drop, at each L and each C.

### 6.3 Online prune (only if time allows)
Manual layer loop over `model.model.layers`. At layer L run the probe, do one fixed-size gather to `[b, chosen sibling, its mask]` + padding to 32 rows, continue. Equality test on. Online τ and bytes must match the replay within noise.

## 7. Timeline (5.5 h)

- 0:00–0:15 — repo, env, model load, config checks, router capture test.
- 0:15–1:30 — harness: per-node decoding, tree mask, 32-row padding, key gather, cache rule, AR reference, equality test. Use ESP-style mean-of-prompt mask so the harness can be tested before training. Commit when the equality test passes on 10 prompts.
- 1:30–2:15 — gated-LoRA wiring, gate test, training (runs in background from ~1:45).
- 1:45–2:30 — while training runs: LRU sim, probe-record writer, replay script. Test them on the untrained ESP-mask run over 10 prompts.
- 2:30–3:15 — load best checkpoint, gate test, equality test, log 30 prompts.
- 3:15–3:45 — probe training and replay. Write the gate table.
- 3:45–4:45 — online prune at the best L, 30 prompts, equality on.
- 4:45–5:15 — RESULTS.md, final push, tarball of logs off the box.
- 5:15–5:30 — buffer.

If behind schedule at 2:30, skip the online prune (6.3); the replay is exact for bytes and τ. If behind at 1:30, run the whole mechanism test on the untrained ESP mask instead of training the LoRA; the rejects are easier to spot, so say so in the write-up.

## 8. Pre-registered gate

On the trained drafter, at L = 8 of 16, C = 16:
- probe top-1 ≥ 0.75 and above both baselines by ≥ 0.05
- bytes per accepted token down ≥ 20% versus no drop
- τ loss ≤ 5% relative
- output bit-identical to plain greedy on every prompt

Report the full L × C table either way. Do not tune past the budget.

## 9. Pitfalls

- `inputs_embeds` and `input_ids` cannot both be passed; embed real tokens by hand.
- The gate must be a multiply on the LoRA delta, not a Python branch.
- LoRA on expert weights would change the memory we are measuring; keep it off experts.
- Rejected siblings still get KV entries in layers ≤ L after a prune; the post-pass crop must use the accepted-prefix length, not the step-input length.
- Timing numbers stay out of RESULTS.md; the headline is `bytes_loaded` per accepted token.

## 10. References
- Limits of Speculation in MoE, arXiv 2609.22156 (verify-side cost model; early-exit rule left as future work)
- HiSpec, arXiv 2510.01336 (intermediate verification on dense early-exit models; cite and separate)
- Apple MTP, arXiv 2507.11851 (gated-LoRA mask path recipe)
- ESP, arXiv 2603.17942 (mask-token decoding layout)
