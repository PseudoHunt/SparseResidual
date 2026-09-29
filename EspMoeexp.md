# CODEX_TASK: ESP on MoE + mid-layer anchor probe (pre-calibrator prune)

This brief is self-contained. Assume zero prior context. Read all of it before writing code.

## 0. One-paragraph summary

Build a lossless self-speculative decoder based on ESP (Embedding-Space Probing, arXiv 2603.17942), first on a dense model, then on a small MoE model. ESP is training-free: it appends mask-token embeddings after the input, reads future-token guesses off the mask positions, and verifies them in the same forward pass. On an MoE, every mask token routes to experts, so mask tokens cost expert loads. We then add one small trained probe that reads mid-layer hidden states, predicts which candidate node will be the deepest accepted node ("anchor"), and drops every other node's mask tokens from layer L onward. Goal: same τ, fewer expert loads. Nothing in the base model is trained. Everything must stay lossless under greedy decoding (bit-identical output to plain autoregressive greedy).

## 1. Hard constraints

- Lossless only. Greedy decoding (temperature 0). Output tokens must match plain greedy AR decoding exactly. Add an assert-equality test.
- Batch size 1 everywhere.
- HF `transformers` only, raw model access. No vLLM, no SGLang, no llama.cpp.
- No training of base-model weights. No LoRA. The only trained module is the probe in Phase 3.
- Keep all shapes fixed-size where practical (fixed number of candidates, fixed number of masks, fixed-size gather at the prune layer). Dynamic shapes are allowed in this GPU POC, but write the code so a fixed-shape port is easy: no data-dependent Python control flow inside the layer loop except the one gather.
- GPU: one A100 80GB. bf16 for weights and activations.
- Everything reproducible: fixed seeds, pinned `transformers` version, results written as JSON.

## 2. Models

Read each model's `config.json` before writing code. Do not assume layer counts or expert counts.

Dense (sanity floor):
- `Qwen/Qwen3-1.7B` (use non-thinking mode: pass `enable_thinking=False` in the chat template, or use the plain prompt format).

MoE (main target):
- `allenai/OLMoE-1B-7B-0125-Instruct` — small, open, router logits exposed in HF.
- Fallback / second MoE: an IBM Granite 3.x MoE 3B instruct model (`ibm-granite/granite-3.1-3b-a800m-instruct` or newest 3.x). Check the HF class exposes router logits.
- Stretch (only if Phases 1–3 pass): `Qwen/Qwen3-30B-A3B` in bf16 (fits in 80GB).

For every MoE, confirm that `model(..., output_router_logits=True)` returns per-layer router logits. If a model does not expose them, register a forward hook on each MoE gate module to capture them. Record top-k expert indices per token per layer.

## 3. Data

- Primary: GSM8K test set, first 200 questions. Prompt: chat template with the question, `max_new_tokens=256`.
- Secondary (after primary works): 100 HumanEval or MBPP prompts (code), and 80 MT-Bench turn-1 prompts (open chat). Spec-Bench subsets are fine if easy to load.
- Stop at EOS or `max_new_tokens`.

## 4. ESP: the method you are implementing

Read arXiv 2603.17942 for exact details. Where this brief is narrower than the paper, follow this brief.

### 4.1 Mask embeddings

- Let `E` be the input embedding matrix. Prompt tokens `x_1..x_t`, embeddings `e_i = E[x_i]`.
- Initialize each of the `k` mask embeddings as the mean of the prompt embeddings: `m = (1/t) * sum_i e_i`. All `k` masks start with the same vector.
- After each generation step `s`, update: `m <- m + λ * (e_new - m)` where `e_new` is the embedding of the newest accepted token. Use `λ` from the paper. If unclear, sweep `λ ∈ {0.0, 0.05, 0.1, 0.3}` on 50 prompts and fix one value.
- Mask tokens are not vocab tokens. Build the step input with `inputs_embeds` (embed real tokens by hand with `model.get_input_embeddings()`), never with `input_ids`.

### 4.2 One decoding step (chain variant, implement first)

State: prefix of accepted tokens (in KV cache), last accepted token `a`.

Step input (appended after the cache): `[c_1, c_2, ..., c_γ, m_1, ..., m_k]` where `c_1..c_γ` are the draft candidates from the previous step (a chain of `γ` tokens), and the `k` masks are attached after the last candidate.

Forward pass with causal attention over `[cache, c_1..c_γ, m_1..m_k]`. Position ids continue from the cache length.

Verification (greedy):
- The logits at the last cached position predict `c_1`'s ground truth. In practice: keep the logits of `a` from the previous pass (or recompute; simplest is to include `a` as the first token of the step input and drop the cached copy — pick one and document it).
- Walk `i = 1..γ`: candidate `c_i` is accepted if `argmax(logits at position of c_{i-1}) == c_i`. Stop at the first mismatch.
- Bonus token `b = argmax(logits at position of the last accepted candidate)`. Always accepted. This is the anchor position.
- Accepted count this pass: `n_acc = (#accepted candidates) + 1` (the +1 is `b`). τ = mean of `n_acc` over passes.

Next-step drafts:
- Read `argmax` (or top-K, see 4.3) at mask positions `m_1..m_k`. Mask `m_j` sits at the position where the `j`-th token after the last candidate would sit. So `m_1`'s logits predict the token after `b`, `m_2` predicts the token after that, and so on. New chain: `c_1 = b`? No — `b` is already accepted and goes into the prefix. New chain: `c_1 = argmax(m_1 logits)`, `c_2 = argmax(m_2 logits)`, … up to `γ = k`.
- Important: the masks were placed after the *last candidate*, not after the *last accepted candidate*. If a candidate was rejected, the masks were conditioned on a wrong prefix. In the chain variant, when rejection happens at candidate `i < γ`, the mask guesses are stale. Simplest correct behavior: discard the mask guesses in that case and run one plain step (draft chain empty, `n_acc = 1` next pass). Log how often this happens. Section 4.4 fixes this properly.

KV cache: after verification, crop the cache to the accepted length (`DynamicCache.crop`). Drop all candidate/mask entries beyond the accepted prefix.

### 4.3 Tree variant (implement second)

ESP builds a small tree: top-K children from `m_1`'s logits, top-1 (or top-K') from `m_2`, pruned by cumulative probability. Implement:
- Width `W` at depth 1 (children of the root), depth-2 expansion top-1 per depth-1 node. Start with `W ∈ {1, 2, 4}`, `k = 2`.
- Tree attention: build a 4D additive attention mask (0 / -inf) so each node attends to the cache, its ancestors, and itself only. Position id = cache_len + depth. Pass as a 4D `attention_mask` (recent `transformers` supports float 4D masks for eager and sdpa). Verify by a unit test that a tree with `W=1` gives identical results to the chain code path.
- Verification: greedy path-walk from the root; accept the longest path whose tokens match the argmaxes along the way; bonus = argmax at the deepest accepted node.
- "Block complexity" `B` = number of tokens processed per model call (candidates + masks). Report it always; τ without `B` is meaningless.

### 4.4 Per-node masks (this is the ESP design and what the prune needs)

Attach `k` masks after **every** candidate node, not just the last one. Mask `m_{v,1}` after node `v` predicts the token after `b_v`, where `b_v` is the bonus that would be sampled if `v` turns out to be the deepest accepted node.

- Each mask set attends to: cache, the path from root to `v`, and its own masks in order. Not to other nodes or other nodes' masks.
- After verification with anchor node `v*` (deepest accepted), the next drafts are read from `m_{v*, 1..k}`. The other mask sets are discarded.
- Block complexity becomes `#nodes + #nodes * k`. This is the cost the prune attacks.

Unit test: with per-node masks, the accepted tokens and the bonus must be identical to the variant in 4.2/4.3 (masks never change verification, because candidates never attend to masks).

## 5. MoE instrumentation

For every pass on an MoE model, record:
- `topk_experts[l][pos]`: top-k expert indices at layer `l` for every position in the step input (candidates and masks), from router logits.
- `union[l]`: number of unique experts at layer `l` across all positions in the step input. Also `union_cand[l]` (candidates only) and `union_mask[l]` (masks only), and `union_total = sum_l union[l]`.
- Simulated expert cache: per-layer LRU with capacity `C` experts (sweep `C ∈ {8, 16, 32}` for OLMoE's 64). Before a pass, the experts needed by this pass hit or miss the cache. Count `misses` per pass. `bytes_loaded = misses * expert_bytes` where `expert_bytes` is computed from the config (three matrices of `hidden × intermediate` in bf16). Update the cache with this pass's experts after the pass. This is the on-device cost proxy.
- Routing overlap (Gate 2): for the anchor node's mask `m_{v*,1}` at layer `l`, compare its top-k expert set with the top-k expert set of the **true** token that later occupies that position (the next pass's first candidate if accepted, else the true token from plain AR). Report hit rate = `|A ∩ B| / k` per layer, averaged over passes. Do the same for `m_{v*,2}` vs the token two ahead.

Also record, for each pass, the hidden states at a chosen set of layers (`L ∈ {N/4, N/3, N/2, 2N/3}`) for every candidate node. Save to disk as fp16 with the anchor label. This is the probe training set (Phase 3). Keep this under ~20 GB; subsample passes if needed.

## 6. Phase 3: the anchor probe

Goal: from layer-`L` hidden states of the candidate nodes in the current pass, predict the anchor node index `v*` before the pass finishes.

- Input per node: `h_L(node)`. Optional extra features: `h_L(root)`, router logits of the node at layer `L`, node depth. Try `h_L(node)` alone first.
- Model: 2-layer MLP scorer `s(node) -> scalar`, softmax over the nodes of a pass, cross-entropy against the one-hot `v*`. Hidden width 512. Train with AdamW, lr 1e-3, 5 epochs, on 80% of cached passes; evaluate on 20% held out by prompt (not by pass).
- Report top-1 accuracy per `L`. Also report accuracy split by "all candidates accepted" vs "rejection happened" passes. Also report accuracy of two cheap baselines: (a) always predict the deepest node, (b) predict from the previous pass's acceptance count. The probe must beat both.
- The probe is a passive readout. It does not change any hidden state, so offline training on cached activations is valid.

## 7. Phase 4: mid-layer prune

Implement a manual layer loop over `model.model.layers` (do not monkeypatch `forward`; write your own loop that replicates the model's forward for one step, including the final norm and lm_head). At layer `L`:
1. Run the probe on `h_L` of the candidate nodes; get `v̂`.
2. Gather the positions to keep: all candidate nodes + the `k` masks of `v̂`. Drop all other mask positions from the hidden state tensor, the attention mask, and the position ids. This is one fixed-size gather (`#nodes + k` rows).
3. Continue layers `L+1..N` on the shorter sequence.
4. KV cache: entries for dropped positions in layers `≤ L` are harmless because the cache is cropped to the accepted prefix after every pass. Make sure the crop logic uses the accepted prefix length, not the step-input length.

Correctness: verification only reads candidate positions, which are never dropped, so accepted tokens and bonus are unchanged. Add the equality assert. If `v̂ ≠ v*`, the surviving masks are the wrong ones; discard them and take a plain step next (log this as a "miss").

Report for each `L` and each `C`:
- τ, block complexity before and after `L`, `union_total`, `misses`, `bytes_loaded`, probe miss rate, wall-clock tokens/sec (secondary; GPU timing is not the on-device cost, but log it).

## 8. Phases and pre-registered gates

Run in order. Stop and write up if a gate fails. Do not tune past a failed gate to make it pass; report it.

Phase 0 — ESP dense, `Qwen3-1.7B`, GSM8K-200, chain `k=2`, then tree `W∈{1,2,4}`.
- Gate 0: greedy-equality test passes on all 200 prompts; τ ≥ 1.5 at `B ≤ 8`. If τ < 1.5, check λ sweep and mask init before concluding.

Phase 1 — ESP on OLMoE, same settings, with MoE instrumentation on.
- Gate 1: τ_MoE ≥ 1.4 at `B ≤ 8`. Below that, mask routing is likely unstable; try the Granite model once, then stop.

Phase 2 — routing overlap and expert-union measurement (from Phase 1 logs).
- Gate 2: mask-vs-true top-k hit rate ≥ 0.6 averaged over the last third of layers for `m_{v*,1}`. Report the full per-layer curve either way. Also report `union_total` with per-node masks vs candidates-only; this is the cost the prune can save.

Phase 3 — probe.
- Gate 3: probe top-1 anchor accuracy ≥ 0.6 at `L = N/2` on held-out prompts, and above both cheap baselines by ≥ 0.05.

Phase 4 — prune.
- Gate 4: at the best `L`, `bytes_loaded` drops ≥ 20% versus unpruned per-node ESP, with τ loss ≤ 5% relative.

## 9. Deliverables

Repo layout:
```
esp_moe/
  esp/            # decoding: mask embeds, chain, tree, per-node masks, prune loop
  moe_instr/      # router capture, union, LRU cache sim, overlap
  probe/          # dataset dump, train, eval
  scripts/        # run_phase0.py ... run_phase4.py, one CLI flag set each
  tests/          # greedy-equality, chain==tree(W=1), per-node==single-set verification
  results/        # JSON per run: config + metrics; plots as PNG
  RESULTS.md      # tables for every gate, pass/fail, and what was observed
```
- Every run writes `results/<phase>_<model>_<config>.json` with the full config and all metrics in Section 7.
- `RESULTS.md` must include: τ vs B table per model; per-layer overlap curve plot; union_total and bytes_loaded vs L plot; probe accuracy vs L table with baselines; the gate table with pass/fail.
- Pin versions in `requirements.txt`. Record the exact `transformers` commit if you patch anything.
- Log every "discard masks" and "probe miss" event count.

## 10. Known pitfalls

- `inputs_embeds` and `input_ids` cannot both be passed to HF models; embed everything by hand.
- 4D attention masks: some `transformers` versions expect a float additive mask with `-inf`/`min` for blocked entries, some accept bool. Check the version's `_prepare_4d_causal_attention_mask` path and test with `W=1`.
- RoPE position ids for tree nodes: siblings share the same position id; masks after node `v` at depth `d` get positions `cache_len + d + 1, +2, ...`.
- Qwen3 chat template inserts thinking tags by default; disable thinking or τ numbers are not comparable.
- OLMoE router uses top-8 of 64 with softmax over all experts; check whether the HF class normalizes top-k weights and whether `output_router_logits` returns pre- or post-softmax logits. We only need the top-k indices.
- Do not measure GPU wall-clock as the headline. The headline on-device cost is `bytes_loaded` from the LRU simulation. Wall-clock is a secondary sanity check.
- The greedy-equality test is the most important test in the repo. Run it on every phase and every model before reporting any number.

## 11. Out of scope for this brief

- Sampling / temperature > 0 verification.
- Any training of base weights, LoRA, or sampler heads.
- Real expert offloading to CPU/flash (simulate only).
- NPU export.
- Any post-verifier drafter (EAGLE, DFlash, etc.).

## 12. References

- ESP: Goel, Gagrani, Lee, Lott. "Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing." arXiv 2603.17942 (ICML 2026).
- Apple MTP (mask-token design background): Samragh et al., arXiv 2507.11851.
- MoE-Spec (expert union growth under SD): arXiv 2602.16052.
- "The Limits of Speculation: Bounding Speculative Decoding in Mixture-of-Experts": arXiv 2609.22156.
