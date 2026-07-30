# Research Handoff — Lossless Self-Speculative Decoding on Mobile NPU

**Owner:** Sanjay, Jarvis AI Lab
**Date:** 30 July 2026
**Purpose:** Move this work to a new Claude account with zero context loss. This file is self-contained. Read it top to bottom once, then use it as a lookup table.

---

## 0. How to read this file

- **§1–2** = the box you must stay inside, and where things stand today.
- **§3–7** = the actual ideas. This is the part you asked for.
- **§8** = smaller live ideas that are parked but not dead.
- **§9** = dead ideas. Do not resurrect them. Each has a stated reason.
- **§10** = every number we have measured.
- **§11–14** = paper plan, working rules, tools, first actions.

A note on style: this doc is written plain on purpose. The jargon that stays (τ, bonus token, mask token, rank, marginal vs conditional) is load-bearing and cannot be swapped out without losing meaning.

---

## 1. Hard constraints — these never move

Every idea below must survive this list. If an idea breaks one, it is out of scope for the NPU track, no matter how good it is.

| Constraint | Meaning |
|---|---|
| Single mobile NPU | No second device. No concurrency. No overlap tricks. |
| Batch-1, latency-bound | Throughput tricks do not help. Wall-clock per token is the thing. |
| Lossless | Output distribution must match the base model exactly. Not "close." |
| Quantized weights | Fine-grained per-value math is expensive. |
| Static computation graph | No dynamic top-k, no per-node gathers, no priority queues, no data-dependent branching. |
| r ≈ 1 cost regime | Draft and verify share the same full model. A draft pass costs about the same as a verify pass. |

**The r ≈ 1 rule is the single most important line in this document.** It is why iterative refinement fails here (see §9, Denoise-BiTA). Any new module must add near-zero cost, or it must buy back more τ than it spends.

**Primary metric:** τ = mean tokens accepted per verify pass (also written TPI). When you report it, define it in the paper. Values around 7–8 at block 16 read like an acceptance *rate* to a careless reviewer. They are not.

**Shipping metric:** cost-adjusted tokens/sec. τ alone never ships.

---

## 2. Where things stand — the one-page version

**The drafter family.** BiTA/MTP-style in-path parallel drafters. Learnable mask tokens are appended to the prefix. One forward pass through the frozen model drafts several future tokens. The same model verifies them.

**The main live direction: pre-calibration.** Use a *predicted* bonus token from block N to condition block N+1's mask hidden states *before* the draft layers run. The whole field does post-calibration instead (fix the logits after the forward pass). That cell is empty as of the last literature check.

**What we have proved so far.**
- Distributed correction capacity beats concentrated capacity: **+0.26 to +0.31 τ** at matched parameter budget. This is a real, replicated, uncollided result. It is the backbone of the main paper.
- FlexDraft's calibration effect is **~5:1 concentrated at slot 1** versus deeper slots. Bonus conditioning saturates the block boundary. Deep-slot weakness is a *state-formation* problem, not an interface problem.
- Mask geometry (causal vs bidirectional inside a block) is a **null**: Δτ ≈ +0.015.
- The rank-1 rung of the capacity ladder is **empirically absent**. The best rank-1 direction is just the cell mean (0.2818 vs 0.2814).
- The slot axis is **collapsible**. Sharing parameters across slots keeps 0.400 of 0.434 at 1/15 the parameters.

**What is blocking.**
1. Rung-1 consistency-loss results are sitting on the Jarvis machine, unread. They gate two other decisions.
2. The marginalization floor has not been measured. It gates the recurrent-head and per-slot-parameter directions.

**Attribution note.** A colleague's team has a history of picking up ideas pitched to them without credit. Practical effect: keep repos private (`8BitSpacemanSpiff/VOLT` especially — it holds unpublished findings), name the three-way novelty combination explicitly in any writeup, and do a prior-art check before sharing anything externally.

---

## 3. IDEA 1 — Pre-calibration (the main paper)

### 3.1 The mechanism

Standard block-parallel drafting works like this:

1. Draft γ tokens in one pass.
2. Verifier checks them. It accepts a prefix and rejects at some slot.
3. At the rejection point, the verifier inserts its own token. **That is the bonus token.**
4. Next block starts from there.

The bonus token is the single most informative thing available at the block boundary. Everybody uses it. The question is *where*.

**Post-calibration (what everyone does).** FlexDraft reads the *true* bonus token after verification, then adds an MLP-produced bias to the current block's output logits. Apple MTP runs a small serial Markov sampler over the previous token after the fact. Both fix the output. Neither touches the *inputs*.

**Pre-calibration (ours).** Predict the bonus token from block N's early-exit signal. Inject it into block N+1's mask/forecast token hidden states *before* the draft layers run.

**Why the depth math works.** FlexDraft's draft branch only runs over the last ~10 layers. The bottom 26 layers are free. They can produce an early-exit estimate of the bonus token before the masks need it. So input-level injection is causally valid inside a single fused pass. This is the objection Claude raised first and got corrected on. Write it down so the next reviewer's objection lands pre-answered.

**The mechanistic claim (this is the paper's thesis sentence):**

> Post-calibration fixes slot-1 output logits but never influences deep-slot state formation. Pre-calibration injects the anchor before all draft layers, so slots 2 through γ attend to the corrected anchor from the start.

This claim is directly supported by our own hazard decomposition (5:1 slot-1 concentration). The boundary is already saturated by post-calibration. The remaining headroom is deep, and it is structural.

### 3.2 Literature position — verified, not assumed

| Method | What it owns | Collides with us? |
|---|---|---|
| **FlexDraft** | Post-forward logit bias from true bonus. Code-level confirmed: one static `mask_embedding` broadcast to all slots; bidirectional intra-block attention over content-free placeholders; bonus never touches next-block mask inputs. | **No.** Pre-calibration cell is empty. |
| **Apple MTP** (Samragh et al., 2507.11851) | Recurrent/serial sampler on the previous token, Markov-1, post-hoc. | **No** on the pre-calibration axis. |
| **Saguaro** (2603.03251) | Predict-then-pre-speculate on *separate hardware*. Spends the prediction as full pre-computed draft branches. | **Adjacent.** Must be cited up front. Also owns §4.1 (bonus predictability) and §4.2 (residual shaping). |
| **DDTree** (2604.12989) | Carries the *already-resolved* previous bonus token forward. | **No.** Different cell — resolved, not predicted. |
| **SpecBlock** (2605.07243) | Carries the drafter's own hidden state at block boundaries, because no target signal is available. | **No.** We supply a predicted *target* correction at exactly that boundary. |
| **DS2D** | BiTA applied on-device. Not a new mechanism. Its Limitations section explicitly names static forecast embeddings as a weakness — which is our gap. | **Helpful.** Cite it as motivation. |

**Novelty framing discipline.** Do not claim "nobody conditions on the bonus token." That is false and will die on an artifact check. Claim the **three-way combination**:

> predicted (not resolved) bonus token × injected into mask hidden states *before* the draft layers × single-pass, single-device.

Eight prior novelty claims in this program died because they were stated too broadly. Name Saguaro and FlexDraft in the first paragraph of the related-work section.

**Also cite as validating primitives, not as competitors:** logit lens, tuned lens, Future Lens (Pal et al. 2023). Future Lens hits >48% accuracy predicting tokens several steps ahead from a single hidden state with a linear probe. Our setting is strictly easier — real verifier trunk, one boundary position, rejection cases only — so our ceiling should be higher.

### 3.3 The 4×4 tree complication (resolved)

Mask tokens are not a flat list. They are **4 branches × 4 masks per branch**. Four branches because we do not know how many of the previous block's tokens will be accepted, so we hedge one branch per acceptance outcome.

Naive reading: 16 injection points. Wrong.

**The bonus token is a per-branch object.** Branch *b* means "tokens 1..b accepted, token b+1 rejected." Each branch cuts the previous block at a different position, so each has its own bonus token, readable from the verifier trunk at that position.

So: **4 anchors, not 16.** One per branch, broadcast to that branch's 4 masks through a shared conditioning module. Weight by branch survival probability, which in practice concentrates on 1–2 branches.

This is good news for the static-graph constraint. Four fixed injection sites, one shared module, no data-dependent structure.

### 3.4 Variants V1–V6

These came out of the Saguaro survey session. V1 is the base scheme.

**V1 — Base.** Predict bonus token from early-exit. Inject as an anchor into next block's masks.

**V2 — Sharpen the free measurement.** Before any training, measure how often the verifier's true choice sits in the drafter's top-k at rejection positions. Split it out by: top-1 vs top-k, binary-reject vs k-way-correct, and as a function of layer depth. Costs nothing. Ranks V1, repulsion, and soft-prior against each other. **Do this first.**

**V3 — Predict the outcome (k, t\*), not just t\*.** Right now the scheme assumes it knows *where* the block will break. It does not. A wrong guess about k means you condition the wrong slot. Add a small k-predictor — drafter confidence is well calibrated for this (D²SD shows it) — and make the conditioning slot-aware.
*Novelty defense, must be argued explicitly:* D²SD uses k to place a second pass. Saguaro uses (k, t\*) as a cache key. We use (k, t\*) as a conditioning signal in the same pass.

**V4 — Geometric conditioning budget.** Saguaro's Theorem 12: do not spread guesses evenly over slots; spend them geometrically where rejection is likely. Our hazard result says the same from the other side (5:1 at slot 1). Merge them: put conditioning capacity — parameters, or fan-out branches — on early slots, geometric decay after. Gives the tree experiment a principled budget shape instead of uniform k.

**V5 — Shape your own residual.** *Sampling regime only. Cheap kill test.*
The bonus token is drawn from the residual max(p − q, 0), where p = target probability and q = draft probability. You control q at sampling time. If you shave q on the tokens you plan to guess, the residual piles onto those exact tokens, so the bonus lands inside your guess set far more often. Still lossless — the accept rule min(1, p/q) self-corrects.
*Cost:* the drafter's guesses get worse, so τ drops.
*Honest prediction:* net negative for us. For Saguaro the payoff is huge because a hit erases all draft latency. For us the payoff is only better conditioning. But it needs zero training, so it is a one-day kill test.
*Attribution:* this is a **transfer of Saguaro §4.2, not an original contribution.** Cite it or it is a death sentence at review.

**V6 — Confidence-gated soft conditioning.** When the predictor is unsure, a wrong anchor can hurt more than a static embedding. So blend:

```
e_mask = e_static + g · Δ(t̂)
```

where `g ∈ [0,1]` comes from predictor confidence and `Δ(t̂)` is the conditioning vector from the predicted bonus token.

Why this is the right shape for us specifically:
- A smooth dial keeps the graph static. An on/off switch does not. Mobile NPU requirement.
- `g = 0` recovers stock BiTA exactly. Free, clean ablation baseline.
- Suggested `g`: a calibrated function of the predictor's margin or entropy — e.g. `g = σ(a · (logit₁ − logit₂) + b)` with `a, b` learned. Keep it monotone in confidence and bounded.

**Priority order:** V2 → V3's k-question folded into Stage A design → V4 shapes the pending sampling run → V6 built into the Stage A module → V5 as a one-day kill test.

### 3.5 Three alternatives that are genuinely different (not variants)

**A. Repulsion instead of attraction.**
V1 predicts the *right* token and pulls toward it. Flip the sign. On a rejection, the token block N proposed is *known* to be wrong — and you already have it, because it is an input to this pass. No prediction needed for the signal itself. You only need to predict one bit: will this get rejected?
That binary is much easier to read early than the k-way correct token. If yes, inject a "not-X" signal so the next block's masks form state biased *away* from the dead token.
*Why it might matter:* our hazard result says deep slots fail at state formation. A wrong attractor from the rejected token may be poisoning that formation. Removing a bad attractor is cheaper than installing a good one.
*Collision risk:* contrastive decoding, DoLa, anti-draft work. Check before claiming.
*Gate:* compare early-exit accuracy on binary-reject vs on k-way-correct. If reject-prediction is much more accurate early (likely), repulsion is the cheaper lever.

**B. Difficulty conditioning instead of content.**
Do not inject a token. Inject a scalar or short vector saying how *hard* the upcoming block is — predicted acceptance length, or verifier entropy at the boundary. Masks then form state conditioned on difficulty. Easy block → draft aggressively. Hard block → deep slots hedge.
*Why it targets the right thing:* acceptance dies by slot 6–8. A per-slot difficulty signal can tell slots 6–8 to stop trying on hard blocks and commit harder on easy ones.
*Big collision warning:* adaptive draft-length / early-stop is a crowded field (SpecDec++, adaptive-γ). The surviving twist is *conditioning mask states on difficulty*, not *stopping the draft at difficulty*. That distinction is the entire novelty. Check it hard.
*Gate:* does predicted acceptance length correlate with actual, read early? Free to measure.

**C. Soft distribution prior instead of one token.**
Early-exit top-1 will often be wrong even when top-k holds the right token. So do not commit to top-1. Inject the *whole* early-exit distribution as a soft prior into the masks. The masks hedge across live candidates.
This is the natural fix if the V2 measurement shows "true bonus in top-k but not top-1" — the most likely outcome.
It also directly spends the rank-2/rank-3 mass as steering, which is exactly what we want (see the value proposition below).
*Collision risk:* soft-target injection looks KD-adjacent. Confirm it is inference-time-distinct.

**Note:** A and C both live or die on the *same* free V2 measurement. Run V2 once and it ranks all three.

### 3.6 The value proposition, stated precisely

A predictor built on drafter states **cannot** rescue cases where the drafter had no signal at all (roughly 10% of rejections). It only harvests cases where the drafter held the correct token at rank 2 or 3 but committed to rank 1.

The pre-calibrator's job is to spend that rank-2 signal as **steering for the next block's masks**, rather than as a second draft candidate at the failed slot. Our own tree experiments showed the second-candidate route does not pay (see §10).

### 3.7 Training scheme

The oracle-on-frozen-weights cheat test **does not work here.** BiTA's forecast embeddings have no trained input port for a bonus token, and untrained base models never produce usable signal in this setup. Do not waste a week on it.

The corrected design has three stages:

- **Stage A — Oracle-fed conditioning.** Co-train a small conditioning module with the frozen backbone *in the loop*. Feed it the true bonus token. Pair with a **matched-capacity self-state control arm** so capacity is not confounded with information.
- **Stage B — Parallel predictor head.** Train the early-exit predictor that produces t̂ from the bottom-layer trunk.
- **Stage C — Scheduled sampling.** Bridge the exposure-bias gap between oracle bonus (Stage A) and predicted bonus (inference).

**Critical training rule (inherited from DVI, pre-empted failure mode):** KD warmup before any RL / acceptance-reward phase is **mandatory**. Pure acceptance reward causes catastrophic failure. This applies to every co-trained rung in this program, not just this one.

**Second critical rule:** do not train on cached activations. Interleaved corrections change the trajectory that downstream frozen layers process. Draft layers need to learn states that survive that downstream processing. Cached vanilla activations are the wrong artifact.

### 3.8 Phase 1 gate results — what passed and what did not

Setup: FlexDraft Qwen3-8B, GSM8K, 200 prompts, frozen weights.

| Gate | Question | Result |
|---|---|---|
| **G1** | Is the bonus predictable at all? | **PASS.** top1→top3 gap = 0.417 at T=1.0, 0.591 at T=0. Plenty of rank-2/3 signal to harvest. |
| **G2** | Does the state depend on restart position? | **FAIL.** Cosine 0.950 between buckets vs 0.523 within. |
| **G3** | Cross-slot second witness | **FAIL.** ~13% precision, killed by base-rate imbalance. |
| **G4** | Mixing objective/anchor | Technical PASS, **null payoff.** w = 1.00 optimal at all 15 slots. The anchor adds nothing on top. |
| **G5** | Error subspace low-rank? | **FAIL** both rungs. Confirmed independently by M3. |

**Incidental finding worth its own paragraph in the paper:** the shipped FlexDraft verifier uses **sample-and-match (Σp·q)**, not rejection sampling (Σmin(p,q)). Measured 0.8448 vs 0.8133 per slot. This is a correctness observation about a published, widely used checkpoint. Report it carefully and neutrally.

### 3.9 Phase 1b offline findings (M1/M2/M3)

- **M3** confirms G5. Worse, the headroom trend runs *against* the low-rank hypothesis: +0.121 → +0.021 as currency approaches acceptance.
- **M2:** the slot axis is collapsible. 0.931 cosine between slot means vs 0.523 within-cell. Sharing across slots keeps 0.400 of 0.434 at **1/15 the parameters.**
- **M1:** the apparent win (0.434 → 0.921) is a **metric artifact.** Setting a = 0 already achieves 0.918. Var(h_clean) is only 8% of the mask–clean gap.
- **Sharpest single finding:** the best rank-1 direction *is* the cell mean (0.2818 vs 0.2814). **The rank-1 rung is empirically absent.**

**Standing lesson:** always check degenerate baselines before claiming a result. M1 nearly shipped as a win.

---

## 4. IDEA 2 — A better loss (acceptance-gated self-conditioned consistency KD)

### 4.1 The problem it solves

Standard training for parallel drafters uses **marginal KD**: each mask slot is trained to match the target's distribution at that position, independently. That is a mismatch with how the drafter is actually used, because at inference the slots are decoded together and accepted as a chain.

The mask-geometry null (Δτ ≈ +0.015) is the evidence: two different intra-block attention geometries converge to the same τ. If geometry does not matter, the ceiling is not geometry. **The ceiling is the objective.**

### 4.2 The loss

Two pieces:

**Self-conditioning.** Run a second teacher pass through the frozen model, conditioned on the drafter's *own* sampled tokens rather than ground truth. Train the mask slots against that. This is token-level self-conditioning for parallel in-path slots.

**The chain gate.** Weight slot *j*'s loss by whether it would actually have been reached:

```
w_j = ∏_{i<j} accept_i
```

This is the greedy-longest-prefix-specific form of acceptance alignment. A slot that would never have been reached contributes nothing to the loss.

Anneal α from 1.0 → 0.5, with a pre-authorized 0.2 trigger.

### 4.3 Lineage — be honest about this

| Ancestor | What it owns |
|---|---|
| **HASS / CORAL** | The harmonization program: self-conditioning + acceptance-aligned objectives. Published, +8–20% over EAGLE-2. But operates on *sequential* EAGLE-family drafters at the *feature* level. |
| **DVI** | Acceptance-gating a loss. Reward-masked CE masks training loss by accept/reject outcome, done online. |
| **Scheduled sampling** | The general exposure-bias fix. |
| **Apple MTP** | Has its own consistency term pushing mask predictions toward autoregressive outputs. **This is the nearest possible neighbor.** |

**Our delta, precisely:** the harmonization program instantiated for **parallel in-path mask slots**, token-level via a second teacher pass through the frozen model, plus the hard chain gate.

**Outstanding check that decides how narrow the delta is:** read Apple MTP's loss section. If their consistency term is already *sampled-self-conditioned* rather than ground-truth-anchored, our contribution narrows to the gate plus the mask-interaction experiment. Do this before writing any novelty sentence.

**Reframing that survives review:** not "we invented a loss" (weak, and now false). Instead: *"the field's training-decoding harmonization principle, instantiated for parallel drafters — where the mask-geometry interaction, which sequential drafters cannot even express, becomes the open question."*

### 4.4 Measured results

- Qwen3-1.7B: τ lifted from ~2.578 to ~3.2 across two mask-geometry variants that converged identically.
- The gated consistency loss produced **+0.18 to +0.34 τ**.

### 4.5 Rung 1 — results exist but are UNREAD

The rung-1 brief was written 18 July. Both arms warm-started from converged checkpoints, acceptance-gated conditional KD, α annealed 1.0→0.5.

Standing rule for that thread was *results local, no metrics pushed*. So the numbers were never committed anywhere searchable. **They are on the Jarvis machine:** `results_cellpair.md`, the rung-1 results file, and `STATUS.log` per the brief's reporting rules.

**Pre-registered predictions to check them against:**
- Δτ positive
- Δτ larger under tree than under chain
- late-slot positional accuracy rises
- m₁ positional accuracy **unchanged** (any movement there is a flag, not a win)
- gated-depth curve climbing

**This is blocking item #1.** Two downstream decisions depend on it.

### 4.6 Rung 2 — the genuinely unclaimed cell

**Acceptance reward applied to bidirectional masks.** Nobody has run this in any drafter family. It is unclaimed.

Sequencing rule: KD warmup first, always (DVI lesson). Rung 2 currently competes with depth-adaptive placement for the next training slot. The gate decides which one runs.

### 4.7 The 2×2 that salvages the recurrent-head idea

Swapping the MLP calibration head for a GRU/LSTM is **not novel on its own** — it overlaps DSpark, Hydra, and Apple MTP.

It is salvageable only as an **interaction claim**:

|  | Marginal KD | Consistency loss |
|---|---|---|
| **MLP head** | baseline | cell 2 |
| **Recurrent head** | cell 3 | cell 4 |

**Prediction:** cross-slot capacity is *inert* under marginal KD and *activates* under the consistency loss. If that holds, the claim is "capacity and objective are coupled," which is a design law, not an architecture swap. That is publishable. The bare swap is not.

Gated on: rung-1 results + the marginalization floor measurement.

---

## 5. IDEA 3 — The unified design space (FlexDraft, Apple MTP, PLFE, our interleaved drafter)

### 5.1 The claim

These four look like different methods. They are four points in **one** space with three axes:

1. **Entry depth** — at which layer do the mask tokens join the model?
2. **Adaptation density** — of the layers the mask tokens travel through, how many are modified?
3. **Capacity form** — how big is each modification? Full matrix, low-rank matrix, or a vector?

### 5.2 The algebra that connects them

Let `W_f` be a frozen verifier weight matrix. Let `W_D` be the matching FlexDraft drafter matrix. FlexDraft was finetuned starting from the verifier, so:

```
W_D = W_f + Δ          where Δ = W_D − W_f
```

So when a mask token `M` passes through FlexDraft:

```
M · W_D  =  M · W_f  +  M · Δ
```

**Exact. Bit for bit.** FlexDraft is the frozen verifier plus a correction Δ applied only to mask rows. Same form as Apple MTP:

```
Apple MTP:   M · W_f  +  M · W_lora
```

The only difference is the *size* of the correction. FlexDraft's Δ is full-rank. Apple's is low-rank.

**Two facts from our checkpoint dissection of FlexDraft:**
- **Δ lives only in attention.** Their MLP and norm weights are bitwise copies of the verifier. Δ ≠ 0 only on q/k/v/o projections of the last 10 layers.
- **Δ is genuinely high-rank.** `o_proj` needs rank ≈ 1700 of 4096 to keep 90% of the energy. Cutting to rank 256 loses 35% of τ.
  *Pre-registered caveat:* truncating a finished solution is not the same as training under a rank limit from the start. That experiment is still open.

**One open code check:** does FlexDraft's draft branch reuse the verifier's K/V for prefix positions, or re-project them? If it re-projects, plain row-gating cannot express the merge exactly.

### 5.3 The capacity ladder

| Rung | Correction form | Method | Input-dependent? |
|---|---|---|---|
| Full rank | full Δ per matrix | FlexDraft | yes |
| Low rank | LoRA (rank r) | Apple MTP | yes |
| **Rank 1** | one outer product | **unclaimed rung** | magnitude only |
| Rank 0 | one fixed vector | **PLFE** | **no** |

**PLFE is rank-0, not rank-1.** A vector added to the state is a **bias**. It does not look at the hidden state at all — same correction for every input. Rank-1 would at least scale with the input.

**PLFE is not new as a mechanism.** One learned vector per layer injected into a frozen model is deep prompt tuning (P-Tuning v2, 2021). Bias-only tuning is BitFit. PLFE's real content is the *application*: draft slots, per-slot indexing, speculative decoding, on-device. Plus its number (+0.4 τ).

**What PLFE's result teaches us.** A constant vector can only fix errors that are the same for every input. Call these **role errors** — "this row is a mask token at slot 3, and slot-3 mask tokens always land in the wrong region." PLFE getting +0.4 τ means a large share of the off-distribution damage is role error. What a constant *cannot* fix is **content error** — corrections that depend on what the text actually says. **That gap is where everything above rank-0 has to earn its cost.**

**Important cross-reference:** our own M1/M2/M3 findings say the rank-1 rung is *empirically absent* on the error structure we measured (best rank-1 direction = the cell mean). So the ladder may be effectively rank-0 → low-rank, with nothing in between. That is itself a finding.

### 5.4 The placement axes

| Method | Entry depth | Density | Capacity |
|---|---|---|---|
| FlexDraft | layer 26 | 1.0 (all 10 traveled layers) | full rank |
| Apple MTP | layer 0 | 1.0 (all layers) | LoRA |
| **Our interleaved drafter** | layer 0 | ~0.17 (6 of 36 layers) | bottleneck modules |
| PLFE | layer 0 | 1.0 (a vector at every layer) | rank 0 |

Read these as bets about frozen computation:

- **FlexDraft bets** mask rows only benefit from layers retrained for them. So skip the first 26 layers entirely, retrain the tail densely.
- **We bet** the frozen layers do useful drafting work for free, and small corrections at a few depths keep the off-distribution rows on track.
- **Apple and PLFE** sit between: full-depth travel, correction at every layer, differing only in capacity per correction.

**Our evidence on this axis:**
- At matched parameter budget, capacity spread across 6 depths beat capacity concentrated at the top by **+0.26 to +0.31 τ**.
- Per-tap ablation of FlexDraft's own checkpoint: their **first tap (layer 26) carries 40% of τ**; middle taps are largely redundant. Their density-1.0 tail is partly decorative.

**Conversions are just walks along the axes:**
- FlexDraft → Apple MTP form: apply Δ = W_D − W_f as a row-gated additive correction. Exact, but not compressible (Δ is high-rank), and placement does not transfer — you get MTP arithmetic at FlexDraft geometry.
- Apple MTP → PLFE: keep entry 0 and density 1.0, drop capacity from LoRA to a constant vector.
- Any of them → ours: drop density below 1.0 and let mask rows travel frozen layers between corrections.

### 5.5 The unclaimed object

**The map nobody has published:** *capacity form × placement × per-slot vs shared*, at matched parameter budget **and** matched on-device cost.

We already own three cells:
- placement result (spread beats concentrated)
- PLFE's rank-0 number
- FlexDraft's dissected Δ

**The grid itself is unclaimed.** This is the strongest packaging option we have, because it turns a pile of ablations into a design law with an algebraic derivation behind it.

### 5.6 Packaging note

PLFE's publishability comes from naming the object after **what it does in the system**, not **where it sits on the capacity ladder.** "Per-layer forecast embedding" sells. "Rank-0 bias" does not. Apply the same rule to our own contributions.

### 5.7 External review flag

Mask-only residual branch modulation (LayerScale-style learned gains on attention and MLP branches) **collides** with LayerScale (CaiT 2021), IA³, and SSF. If we use it, **retain the bias alongside the modulation** rather than replacing it, and cite all three.

---

## 6. IDEA 4 — Improving DeLS-Spec (GPU track, OUT OF SCOPE for NPU)

**Scope warning up front.** This is a **GPU / datacenter-throughput** line. Tree expansion, dynamic top-k, per-node gathers, and best-first priority queues all violate the static-graph constraint. It is worth doing only if you want a *second, separate* paper in the datacenter regime. **Do not let it feed the pre-calibrator work.**

Status: design frozen, not yet run. Gated on one four-arm prototype.

### 6.1 What DeLS-Spec is today

It is a **logit-combination** method, not a tree method.

- Keeps DFlash frozen as a **long-context expert**.
- Adds a small **local head** as a **short-context expert**.
- Local head trained **alone**, plain next-token prediction. No joint training with the target model or with DFlash.
- At inference, mixes the two logit streams:

```
ℓ(x_i) = ℓ_L(x_i | y)  +  α · ℓ_S(x_i | z_i)  −  β · ℓ_P(x_i)
```

- `ℓ_L` = DFlash long-context logits. **Marginal.** Computed once per round, static across the block.
- `ℓ_S` = local head logits. **Depends on the drafted prefix `z_i` inside the block.** This is the only path-dependent term.
- `ℓ_P` = unigram prior. Static.

**Strength:** very low training cost, modular, portable. **Weakness:** raw quality. Lands ~4.8× average speedup. DominoTree reports up to 6.6× and ~10.7 tokens/round with a jointly-trained corrector.

> **Open verification item:** confirm the `−β·ℓ_P` unigram-prior term is actually in the DeLS-Spec paper. The two-expert fusion is confirmed. The prior term is not, and the normalization argument depends on it.

### 6.2 The observation that started this

DeLS-Spec has no tree. But it already contains the ingredient a tree needs: **two experts that can disagree.** Disagreement between a marginal long-context expert and a path-dependent local expert is a natural signal for *where to spend tree budget*.

### 6.3 The honest novelty position

The rename to "Divergence-Allocated Conditional Tree Drafting" is honest about lowering the headline from "DeLS→tree" to "adaptive allocation via decoupled experts." But **both halves are contested**:
- Adaptive allocation: crowded (OPT-Tree, EAGLE-2).
- Decoupled experts: DeLS owns it.

The novelty is strictly the **intersection** — using specifically *decoupled-expert* divergence as the allocation signal.

> **Open verification item, must pass before anything else:** confirm nobody has published divergence-guided branching for spec-decode trees. About an hour of reading.

### 6.4 The decisive experiment — four-arm prototype

Python prototype at matched `B` (branches) and `M` (masks). No GPU infrastructure. A few days. Genuinely decisive.

It is a 2×2: **{decoupled two-expert, coupled single-corrector} × {fixed allocation, divergence allocation}.**

| Arm | Scorer | Allocation |
|---|---|---|
| **C** | DeLS two-expert fusion | fixed / uniform tree |
| **D** | DeLS two-expert fusion | **divergence-allocated** |
| **B** | Coupled corrector (Domino-style) | fixed / uniform tree |
| **E** | Coupled corrector (Domino-style) | divergence-allocated (base-vs-corrected) |

*(Arm definitions reconstructed from the design discussion — confirm against the original proposal doc before running.)*

**The one number the whole paper rests on:**

```
(Arm D − Arm C)  must be materially larger than  (Arm E − Arm B)
```

That is the DeLS-**specificity** claim. If decoupled-expert divergence is a better allocation signal than the base-vs-corrected divergence any coupled method already has, there is a paper. If the gap is small, there is an ablation table where the DeLS-specific part lost.

**Arm E is the assassin, not a checkbox.** Go in expecting it to kill the headline. The modal outcome is "D beats C, but E beats B just as much" → the method is **general**, not DeLS-specific, and the headline dies.

### 6.5 Gates

- **Gate 1 (τ):** τ-gap ≤ 5–8%.
- **Gate 2 (builder latency):** **must be added.** The proposal says "three numbers decide this, not one" — τ, builder cost, per-node microseconds — but attaches a gate only to τ. On a throughput tree, builder cost and per-node microseconds are **co-equal deciders**. A slightly-lower-τ method with a cheaper scorer can win tokens/s. Without a threshold on Gate 2, "three numbers" is rhetoric.

### 6.6 Pre-committed kill — write this down before running

If Arm E ≈ Arm D's lift:
- either write the **general** adaptive-tree paper and drop DeLS to a footnote,
- or walk.

**Do not relabel a null on the DeLS-specificity claim as "still promising."** That exact relabeling turned eight prior novelty claims into late kills instead of early ones.

### 6.7 Clock risk

This is a two-week race against the DominoTree authors adding the same branching to their stronger scorer. If they ship disagreement-branching on Domino first, our Arm E becomes their paper and our Arm D is the ablation that lost. The clock should be the framing, not a footnote.

---

## 7. IDEA 5 — Off-argmax branch conditioning (the DominoTree gap)

This came out of the JetSpec/DominoTree literature survey and is the cleanest small target we found.

**The gap, verified from their own papers:**
- **DominoTree** documents its own weakness: its chain-trained GRU is miscalibrated on off-argmax branches. Φ = 10.73 predicted vs 9.28 realized — **1.16× over-credit.** They reuse a chain-trained head off-label on tree branches.
- **JetSpec** trains using **only ground-truth ancestors.** So off-argmax branches are out-of-distribution at inference. This was verified from their training section and closes the main collision risk.

**The direction:** train a conditioning module *specifically for off-argmax tree branches*. This is a **state-formation fix**, not a scoring fix. Neither DominoTree (reuses off-label) nor JetSpec (trains on ground truth only) does this.

**Related JetSpec claims we verified live, worth citing precisely:**
- Forward-KL beats reverse-KL by 36–46%. But forward-KL only beats plain SFT by ~3%.
- The budget-scaling advantage appears at **32+ nodes**, not at 16.
- Cumulative-logprob pruning beats entropy-only by ~42%.

**Constraint check:** this is compatible with the NPU track *only* if the branch structure stays static. Fixed branch count, fixed depth, shared module. That is the same shape as the 4×4 tree in §3.3, so it composes with the pre-calibrator rather than competing with it.

---

## 7A. IDEA 6 — Mask-row correction: measure the residual, then build (July 2026, post two review rounds)

*Added after v1 of this handoff. Numbered 7A so nothing below it shifts. This is now the **most current** thread and it supersedes the branch-modulation proposal as a standalone document.*

### 7A.1 Status in one paragraph

We have a measured result — the error field of mask rows in an in-path drafter is dominated by a **per-cell mean**, with a diffuse **high-rank residual** — and no method yet. Two rounds of external review killed both candidate methods we brought them (branch modulation, Soft-ID) as flagships. They converged instead on **one unresolved measurement** that decides whether our strongest claim survives at all. That measurement is the next thing to run. **Do not build any method before it returns.**

### 7A.2 Setting and the object

In-path drafter family (BiTA, Apple MTP, FlexDraft): mask tokens are appended to the sequence and travel through the verifier's **own frozen layers**. Drafting happens inside the verify forward pass. This is distinct from post-verifier drafters (EAGLE, DFlash, DeLS-Spec), where a separate model drafts after verification.

Binding constraints are the usual ones (§1). Metric is τ, judged on cost-adjusted tokens/sec.

**The problem.** Mask rows are off-distribution for frozen layers trained on real token embeddings. Define:

```
e = h_mask − h_clean          indexed by cell = (depth l, slot s)
```

Everything below is about the structure of `e`.

### 7A.3 What is measured

FlexDraft Qwen3-8B, GSM8K, 200 prompts, frozen weights, T=1.0 and T=0. Integrity verified: **0 reconstruction mismatches across 8288 blocks**; losslessness holds up to bf16 reproducibility. CIs bootstrap over prompts.

**a) The error is mean-dominated.** Uncentered held-out r=1 energy **0.282** vs centered PC1 **0.047**. Errors cluster around a large nonzero mean with diffuse scatter around it.

**b) The rank-1 rung is empirically absent.** Best rank-1 direction **0.2818**; cell mean direction **0.2814**. They are the same direction. Letting the coefficient float per sample buys **0.013**.

**c) Headroom above rank-0 shrinks as the currency approaches acceptance.**
+0.121 (hidden L2) → +0.075 (logit-space) → +0.061 (acceptance-weighted) → **+0.021 (norm-aware)**. The trend runs *against* the low-rank hypothesis.

**d) Depth carries structure; slot means do not.** Cosine between slot means **0.931** (within-cell baseline 0.523); between depth means **0.600**. Sharing across slots retains **0.400 of 0.434 at 1/15 the parameters**; sharing across depths collapses to **0.054**. This independently reproduces the earlier placement result (+0.26–0.31 τ, distributed over top-concentrated capacity at matched budget).

**e) Weight-space corroboration.** FlexDraft's trained delta needs rank ≈1700 of 4096 for 90% energy; r=256 truncation loses 35% of τ. *Caveat:* truncating a finished unconstrained solution ≠ training under a rank constraint.

**f) Measured dead.** Restart-length-indexed vectors (between-bucket cos 0.950 vs within 0.532); cross-slot re-ranking (real lift 0.448 vs 0.271 control, but ~13% precision on base rates); anchor-logit blending (optimal weight 0 at every slot); mask geometry (consistent nulls).

### 7A.4 Three caveats that gate everything

**a) The metric is mean-dominated and probably wrong.** `Var(h_clean)` within a cell is only **8%** of the mask–clean gap. So "energy captured" mostly measures the region-move a bias already performs, while τ is decided by the **within-cell variation** that distinguishes contexts. Demonstrated: a diagonal affine reached 0.921 energy captured, but `a = 0` — discarding the mask state entirely — already reached 0.918.

> Read (b) and (c) above as: *no useful rank-1 structure is visible under reconstruction metrics; whether it exists in the acceptance-relevant residual is **unresolved**.*

**b) Nothing is measured in τ.** No fitted correction has been applied at inference. Offline-fitted corrections have failed **twice** in this program from trajectory mismatch.

**c) The verifier is non-standard.** The released FlexDraft implementation uses **sample-and-match** (draw from target, accept iff equal), so acceptance is `Σ_v p(v)q(v)`, not the overlap `Σ_v min(p(v),q(v))`. Measured **0.8448 vs 0.8133**. The FlexDraft *paper* describes ratio-based speculative sampling, so paper and code disagree. **Settle this before choosing any objective** — the two targets are not interchangeable, and switching verifiers after training invalidates the run.

*Note on `Σpq`:* maximizing it is **mode-seeking**, and that is correct rather than defective. With `q = softmax(z)`, `∂A/∂z_j = q_j(p_j − A)`, so training pushes mass onto the verifier's high-probability tokens. Under sample-and-match that maximizes acceptance and stays lossless, because the verifier's own sample is committed on mismatch. It is simply **not distribution matching**, and must not be described as such.

### 7A.5 What the two review rounds killed

**a) Branch modulation → demoted to a diagnostic baseline.**
Proposed form: `h' + g^A ⊙ a + g^M ⊙ m + b`, gains initialized to 1, shared across slots, varying by depth.

Three fatal objections:

- **Mechanism collision.** This is **LayerScale** (Touvron et al., CaiT 2021). Nearby: **IA³**, **SSF**. Not novel as a mechanism.
- **The attention gain adds no function class.** FlexDraft already trains mask-specific Q/K/V/O projectors. A diagonal gain on attention output folds into the trainable output projection: `diag(g^A)·W_O`. So `g^A` is a reparameterization of tuning FlexDraft already does. The **MLP gain survives** — the FFN is shared and frozen, so mask-only scaling after it cannot be folded without touching clean rows — but that reduces the method to **mask-only SSF**.
- **A registered prediction was invalid, and is withdrawn.** We predicted `g^A` would move and `g^M` stay near 1, because FlexDraft's delta is attention-only. That inference does not hold: changing attention changes the FFN's *input*, so a frozen FFN can produce badly wrong output without its weights ever being trained. FlexDraft's choice shows attention tuning was *sufficient for their design*, not that the FFN contribution is already correct.

**Two technical corrections to carry forward regardless of what gets built:**

- **The displayed correction equation was wrong.** In a sequential block, `u = h + a`, `m = MLP(u)`. Gating attention changes the MLP's input, so `m' ≠ m` and `Δh ≠ (g^A − 1)⊙a + (g^M − 1)⊙m`. Either accept the sequential form and rewrite the analysis, or build a shadow residual path — which is no longer a standard block.
- **Bias and gains are not identifiable as written.** If `a` and `m` have nonzero means, then `b`, `d^A ⊙ E[a]`, and `d^M ⊙ E[m]` all explain the same mean correction, so "both beats bias-only" proves nothing. **Center the branch outputs:**

  ```
  Δh = b_l + d^A_l ⊙ (a − ā_l) + d^M_l ⊙ (m − m̄_l)
  ```

  Then the bias handles the region shift and modulation can only use **contextual deviation**.

**b) Soft-ID → one oracle experiment, not a pivot.**
Proposal: predict a posterior over token-content codes at an intermediate depth, convert to a soft content vector, reinject to condition later mask-only layers.

Prior-art status is better than first feared. LayerSkip and Draft&Verify *consume* intermediate predictions as drafter output; Soft-ID would *reinject* one as internal conditioning. That is a real distinction. *(Correction to an earlier note: DeLS-Spec is **not** an early-exit method; it is the two-expert long/short-context fusion design — see §6.1.)*

The stronger objection is mechanistic: the posterior is computed from `h_{l*}`, so what does decoding and re-encoding add that the remaining frozen layers could not extract directly from `h_{l*}`? The answer must be that **the token bottleneck is a useful inductive bias** — a claim, not a given. It also shares its core hypothesis with Denoise-BiTA, which this program already killed at r≈1 cost structure (§9). Moving the feedback from a second pass to later layers removes the *cost* objection but not the *information* objection.

**c) Do not sell the optimizer as novel.**
**DREAM-S** occupies hardware-aware NAS for drafter design. **KnapSpec** occupies knapsack-formulated module selection under a latency budget. Neither searches over *correction forms* for mask rows in a frozen in-path drafter, so neither kills the direction — but "we use hardware-aware search" and "we choose components under a cost budget" are both taken. Whether selection uses knapsack, DP, or enumeration is implementation detail.

### 7A.6 The decisive experiment — centered residual analysis

**The slot-collapsibility finding (7A.3d) is unproven where it matters.** Cosine 0.931 establishes that the **means** are near-parallel. It says nothing about whether the **acceptance-relevant residual** is slot-invariant. The capacity-form design law and its 1/15-parameter corollary both rest on this. If the residual is slot-dependent while the mean is not, the method corollary is gutted and the law weakens to a statement about the part that does not decide tokens.

Define `μ_{l,s} = E[e_{i,l,s}]` and `r_{i,l,s} = e_{i,l,s} − μ_{l,s}`. Evaluate every candidate correction **only on `r`**:

```
C_res = 1 − E‖r − r̂‖² / E‖r‖²
```

Four tests. Geometry alone is not sufficient.

- **Test A — residual spectrum by depth and slot.** Normalized singular-value spectrum of `r` per (l,s). **Do not pool slots before this measurement.** Reports whether residual capacity is low- or high-rank, and whether that varies by depth or slot.
- **Test B — residual-subspace similarity.** Mean cosine is inappropriate on zero-mean data. Compare principal subspaces: `sim_k(s,t) = (1/k)‖U_{l,s}^T U_{l,t}‖_F²`. Asks whether two slots vary along the same directions even when individual residuals do not align.
- **Test C — cross-slot transfer. This is the load-bearing test.** Fit a correction on slot `s`, apply it at slot `t`, **replay the remaining layers**, and measure the acceptance change. Compare `ΔA(s→s)` against `ΔA(s→t)`. The real slot-sharing claim is not "the residuals look similar" but "a correction learned at one slot retains most of its acceptance benefit at another."
- **Test D — shared vs slot-specific at matched parameter cost.** Otherwise the shared model looks worse merely for having fewer parameters, or better for seeing more training examples.

Also run: centered predictability of `r` from centered `h`, `a`, `m`; and acceptance-Jacobian-weighted residual energy.

**Cost.** A, B, D are offline on existing logs. **C requires replay through the remaining layers**, and separating `a` and `m` per mask row per depth requires a **new instrumented pass with hooks splitting the residual write**. That is a GPU session, not pure arithmetic.

**Pre-registered outcomes — write these down before running:**

| Outcome | What it means | What to build |
|---|---|---|
| **A** — residuals slot-shared, differ by depth | Best case | Depth-heterogeneous mask adaptation: at each depth pick the smallest adequate form from {none ⊂ bias ⊂ centered diagonal ⊂ low-rank ⊂ full}, one module shared across slots. Parameter-efficiency argument survives. |
| **B** — shared directions, slot-specific coefficients | `Δh_{l,s} = b_l + U_l c_{l,s}(x)`: shared depth basis, cheap slot-specific coordinates | Replaces "slot does not matter" with "slots share correction subspaces but need different coordinates within them." Possibly the **more interesting** result. **Do not pre-build it.** |
| **C** — residuals fully slot-specific | Parameter-efficiency corollary badly weakened | Savings must come from elsewhere: grouped slots, structured basis sharing. |
| **D** — no constrained rung improves replayed acceptance | Contextual residual is measurable but not usefully correctable under these constraints | Analysis result, not a method. |

### 7A.7 Plan, in order

1. **Centered residual analysis (7A.6).** A, B, D offline first; C needs the instrumented pass. This is the decisive step. **Nothing downstream is stable without it.**
2. **Artifact checks.** Mask-row LayerScale / SSF / IA³; intra-depth posterior reinjection; DREAM-S; KnapSpec; heterogeneous adapter/rank allocation *outside* speculative decoding. **Nine** novelty claims in this program have died on such checks.
3. **Verifier decision.** Choose sample-and-match or rejection sampling, fix the matching objective, reproduce baseline τ. Note: **switching to rejection sampling is itself a free τ gain** (`Σmin ≥ Σpq` always) and may exceed what any adapter delivers. That is a real possibility, not a joke — price it before spending a training slot.
4. **Four-rung centered intervention ladder:** static / centered diagonal / low-rank / dense, each **replayed through the remaining layers**, not scored offline. This separates four distinct worlds:
   - no rung beats static → residual unusable;
   - dense works but low-rank fails → genuinely high capacity needed;
   - low-rank works but diagonal fails → branch info helps, SSF-style scaling inadequate;
   - diagonal works → cheap modulation viable, but still weakly novel.
5. **Soft-ID oracle** — only if step 4 shows contextual correction is usable. Inject a content code from the **true** future token; compare against shuffled and random. **Stop immediately** if true identity does not separate from shuffled.
6. **Construct the method from the observed structure. Not before.**

**Training notes for whatever emerges:**
- Do **not** use mask–clean L2 as the main objective. Retain it as an auxiliary diagnostic only.
- Go **on-policy** via trajectory refresh rounds. Corrections change which examples survive to later slots. This is **intervention-induced distribution shift**, not ordinary selection bias.
- **KD warmup before any acceptance-reward stage is mandatory.** Pure acceptance reward has caused catastrophic failure in this program before (§3.7).

### 7A.8 OPEN DECISION — the hardware framing

**Status: unresolved. Settle after step 4, not before.** It affects how results are framed, not which experiments run, so deferring costs nothing.

**The situation.** The project's stated setting is a single mobile NPU, but there is currently **no documented target device, compiler, runtime, fused-kernel implementation, or benchmarking path**. All measurement to date is GPU-based.

**Two things must not be conflated:**

- **Dropping NPU *measurement*.** Report GPU τ and latency as measured; present operator counts, parameter counts, and activation traffic as clearly-labeled **analysis**. Never present estimated NPU speed as a measured outcome. **This part is settled:** we will not claim measured NPU throughput without an NPU.
- **Dropping the NPU *constraint*.** This is the open question, and it is expensive. The constraint currently does three things:
  1. It is **why rank-0 is interesting at all** — the cost ordering inverts on that hardware. On a GPU it largely does not, which turns a design law into a crowded parameter-efficiency claim.
  2. It is the justification for **"everything stays parallel,"** which is what keeps us out of the serial-cascade cell occupied by Apple MTP, Hydra, DSpark, Domino, and DeLS-Spec.
  3. It is **one of the three intended paper legs** (algorithm + systems + empirical, §11).
  It also sharpens the distinction from DREAM-S and KnapSpec, both of which are hardware-aware.

**Three possible claim levels, increasing in strength:**

1. **Estimated cost.** Operator counts and vendor-reported latency assumptions. Fine for deciding what to try. Not acceptable as a central claim.
2. **NPU operator microbenchmarks.** Implement the relevant ops on the target runtime and measure: mask-only bias, grouped diagonal scaling, low-rank projection, slot routing, and graph/memory-layout overhead. Gives a real cost table for architecture selection without full deployment. **Cheapest meaningful upgrade — pursue this if any NPU access exists at all.**
3. **End-to-end deployment.** Decode and verify latency, τ, tokens/sec, memory, power.

**Fallback framing if no NPU path materializes:** re-scope from "single mobile NPU method" to **"static-graph, batch-1 edge-accelerator method."** Keep the constraint as a *design* constraint that motivates the search space, report GPU results as the measured outcomes, and label hardware-cost analysis as analysis.

**Current working position:** keep the constraint as a design constraint, drop the measured claim, revisit after step 4 when we know which correction forms are in contention and therefore what a cost table would need to cover.

### 7A.9 The claim this is all aimed at

Not a particular adapter. The potential finding is:

> **Mask-row correction decomposes into a cheap shared region shift and a structured, depth-dependent acceptance residual whose minimum required function class can be measured and allocated.**

That is an **analysis result today.** It becomes a method paper only if the centered measurements reveal reproducible structure **and** the resulting heterogeneous architecture beats uniform alternatives at matched cost **in real τ**.

Standing methodological rule, learned expensively: *inventing the mechanism before measuring it* is what put this program into occupied or empirically dead cells nine times. Step 1 is a measurement.

### 7A.10 How this section updates the rest of the handoff

Read these as amendments. The original text in those sections is left intact on purpose, so the history is visible.

- **§3.9 / §10 (M1–M3 findings).** Same measurements, now with sharper error bars and a stronger caveat. The rank-1-absent and slot-collapsible results still stand *as reconstruction results*, but 7A.4a says reconstruction is the wrong currency. **Do not cite slot-collapsibility as settled until Test C returns.**
- **§3.8 / §8.6 (the sample-and-match finding).** Upgraded from "incidental finding" to a **blocking decision** (7A.7 step 3). Also: switching to rejection sampling may be a free τ win larger than any adapter we build.
- **§5.3 (capacity ladder).** The "rank-1 rung is absent" claim is now explicitly conditional on the metric. It may reappear in the acceptance-relevant residual.
- **§5.7 (LayerScale/IA³/SSF collision flag).** Now fully resolved: branch modulation is **demoted to a diagnostic baseline**, not a method. The centering fix (7A.5a) is the part worth keeping.
- **§8.2 (per-slot parameters as MoE).** Its expected payoff now depends entirely on Test C. Outcome B or C would revive it.
- **§14 (first actions).** Insert the centered residual analysis as the new item 0. It gates more than anything else on that list.
- **New dead/demoted entries for §9:** branch modulation (mechanism collision + no added function class on attention); restart-length-indexed vectors; cross-slot re-ranking; anchor-logit blending. Soft-ID is **not** dead — it is parked behind one oracle experiment.
- **New names to add to the reading list in §13:** LayerSkip, Draft&Verify, DREAM-S, KnapSpec.

---

## 8. Other live ideas (parked, not dead)

**8.1 Depth-adaptive placement.** The constructive successor to the placement result. Instead of uniform 6-of-36 spacing, place correction capacity where the measured deficit is. One gate away from being the second novel method, sitting directly on top of the first. Competing with rung-2 for the next training slot.

**8.2 Per-slot dedicated parameters as mixture-of-experts.** Equivalent to the depth-adaptive placement specialization axis already in the planned matrix. The geometry probe predicts **2 banks capture most of the gain over 8**. And M2 says the slot axis is collapsible anyway, so the expected payoff here is now low. Keep it as an ablation row, not a headline.

**8.3 The marginalization floor.** Measure the τ achievable by a perfectly-trained *marginal* drafter. This is the reference line. Everything above it is attributable to conditioning/coordination. It gates the recurrent-head and per-slot-parameter directions. **This is blocking item #2.**

**8.4 The conditional-hazard decomposition as a standalone paper.** The ~5:1 concentration of calibration effect at slot 1 vs deeper slots, on the common risk set. This may be the strongest standalone publishable object we have. It supports a clean design law: *bonus conditioning saturates the boundary; residual deep-slot weakness is a state-formation problem, not an interface problem.* Good fit for the ENLSP analysis-only paper.

**8.5 The information ladder.** "What is the minimum verifier information needed for recovery?" Ablate the conditioning signal in rungs: accepted length only → + bonus token → + verifier hidden state → + KV, with the serial oracle as the limit. **Guard, learned the hard way:** every rung must run at **matched module capacity**, or information content gets confounded with parameter count. That was the Denoise-BiTA-era error. The method is then "ship the cheapest rung that recovers the gap."

**8.6 The FlexDraft verifier finding.** Shipped FlexDraft uses sample-and-match (Σp·q) rather than rejection sampling (Σmin(p,q)). Also noted separately: FlexDraft's temperature path does not verify losslessly. Both are correctness observations about a widely used public checkpoint. Report neutrally; do not lead with them.

---

## 9. Dead ideas — do not resurrect

Each of these was killed for a stated reason. If a new idea looks like one of these, check the reason first.

| Idea | Why it died |
|---|---|
| **Denoise-BiTA** (iterative diffusion refinement) | Raises raw τ (2.33 → 2.93, +25.9%, losslessly) but **nets negative at r≈1**, because draft and verify share the same full model. The remask ablation was decisive: remask-OFF collapsed τ to 1.69, below baseline. **This is a publishable cost-structure principle, not just a failure.** |
| **Mask geometry** (causal vs bidirectional intra-block) | Null. Δτ ≈ +0.015. Replicated across chain, tree, and consistency objectives. |
| **VOLT premise** | Empirically false on FlexDraft Qwen3-8B. The frozen calibration head *already* uses bonus information correctly (D2 Δlogprob 1.5–4.1 nats, correct direction per KL). STOP-2 triggered. |
| **KV-repair** | Killed by architectural analysis. There is no cached bonus-position K/V in the parallel layout. |
| **Full VOLT latent reconstruction** | Killed by wall-clock arithmetic. |
| **Raw recurrent/serial head swap** | Overlaps DSpark, Hydra, Apple MTP. Salvageable *only* as the 2×2 interaction claim in §4.7. |
| **JetSpec-style tree attention as a bolt-on** | Not viable. Requires per-branch, per-depth conditioning on actual intra-block ancestor tokens. Incompatible with a marginal-latent drafter architecture. |
| **Bonus predictor as second draft candidate at the failed slot** | Tree experiments showed it does not pay. Spend the signal as *steering* instead (§3.6). |
| **Serial cascade** | Reopened briefly by the repaired boundary proxy (6.03% [5.06, 6.97]) and cascade-oracle ceiling (16.07% [12.1, 19.8]), then deprioritized on collision grounds — Hydra, Apple MTP, and DSpark occupy that cell. |
| **Adding diffusion to EAGLE-3** | Space is saturated as of mid-2026: DFlash, SpecDiff-2, DEER, FastEagle, D²SD, FailFast, DART. |
| **Steering DFlash** (SD²-style) | DFlash's KV injection already occupies the per-layer conditioning channel steering would target. Berdoz et al.'s own expressiveness results suggest dynamism is near its ceiling. |

**Naming note:** "SD²" is Berdoz et al., AAAI-26, *Steering Pretrained Drafters during Speculative Decoding*. It is **not** SpecDiff-2. These get confused constantly.

---

## 10. Results ledger — every number we have

### Qwen3-1.7B (earlier ablations)
| Result | Value |
|---|---|
| Baseline τ | ~2.578 |
| After consistency loss, two geometry variants | ~3.2 (both, converged identically) |
| Placement: distributed vs top-concentrated, matched budget | **+0.26 to +0.31 τ** (25M params across 6 depths) |
| Gated consistency loss | **+0.18 to +0.34 τ** |
| Mask geometry (causal vs bidirectional) | Δτ ≈ +0.015 (null) |

### Denoise-BiTA
| Result | Value |
|---|---|
| Baseline τ | 2.33 |
| v2 with self-conditioning + per-step mask conditioning + matched remask | 2.93 (+25.9%, lossless) |
| remask-OFF ablation | 1.69 (below baseline — proves the gain is from Mask-Predict, not bidirectional attention) |
| Cost-adjusted verdict | **Net negative at r≈1** |

### FlexDraft Qwen3-8B, GSM8K, block 16, tree budget 30, frozen weights
| Arm | Description | τ |
|---|---|---|
| **A** | Uncalibrated baseline | 7.853 |
| **B** | True bonus (post-calibration ceiling) | 8.749 |
| **C_k2** | Pre-emptive top-2 bonus guesses as branches | 7.639 |
| **C_k3** | Pre-emptive top-3 bonus guesses as branches | 6.675 |

Precondition divergence: frac = 0.480, KL = 3.141. All integrity checks passed. `nonargmax_bonus_accepted_frac = 0.000`.

**Interpretation, important:** the C < A < B ordering with monotonic degradation as k rises is a **structural consequence of greedy decoding**, not evidence the idea fails. Under greedy, non-argmax bonus tokens are rejected at their root by definition — confirmed by the 0.000 figure. Extra branches consume the matched budget without contributing accepted length. The hypothesis was always about the **sampling** regime, where non-argmax bonus tokens carry real target mass. **The sampling run is still pending.**

### Phase 1 gates (FlexDraft Qwen3-8B, GSM8K, 200 prompts)
See §3.8. G1 PASS; G2, G3, G5 FAIL; G4 technical PASS with null payoff.

### Phase 1b offline (M1/M2/M3)
See §3.9. Rank-1 rung absent; slot axis collapsible; M1 was a metric artifact.

### VOLT rung-0
| Item | Value |
|---|---|
| D2 Δlogprob (bonus sensitivity) | 1.5–4.1 nats, correct direction |
| D5 interaction (branch-specific info in H) | 4.65 nats |
| Repaired boundary proxy | 6.03% [5.06, 6.97] — up from the reported 2.634% |
| Cascade-oracle ceiling | 16.07% [12.1, 19.8] |
| Conditional-hazard concentration | **~5:1, slot 1 vs deeper slots** |

**Bugs found in external review, already fixed:** frozen RNG reseed in B1 (invalidated the estimator comparison), sign error in the carrier-gap formula, capacity confound in the additive probe arm. Reported G_max was not chain-consistent.

### FlexDraft checkpoint dissection
| Item | Value |
|---|---|
| Where Δ lives | Attention only (q/k/v/o, last 10 layers). MLP + norms are bitwise copies. |
| Rank of Δ (`o_proj`) | ~1700 of 4096 for 90% energy |
| Rank-256 truncation | −35% τ |
| First tap (layer 26) | carries 40% of τ |
| Middle taps | largely redundant |
| Verifier sampling | sample-and-match (Σp·q) = 0.8448 vs rejection (Σmin(p,q)) = 0.8133 per slot |

### PLFE (colleagues')
+0.4 τ. Rank-0 (a per-layer bias, per-slot indexed).

---

## 11. Publication plan

**Three-leg template for the main paper:**
1. **Algorithmic contribution** — pre-calibration, or the design-law framing.
2. **Systems contribution** — the NPU / static-graph builder constraint. **This is an asset, not a blocker.** Most of the field ignores static graphs entirely. Own that.
3. **Empirical findings** — the placement result, the hazard decomposition, the negative results.

**Near-term: ENLSP workshop at NeurIPS 2026.** Analysis-only, measurement-and-diagnostic framing. No training required. ~2-week paper. Best candidates for the core object:
- the conditional-hazard decomposition (§8.4), or
- the unified design-space map (§5).

**Full method paper: 3–4 months, targeting ~November 2026 submission.**

**Possible second paper (GPU track):** the divergence-allocated tree (§6), conditional on the Arm D vs Arm E result.

**Framing rules that keep coming up:**
- Failed experiments are findings. Denoise-BiTA is publishable as a cost-structure principle.
- Name the exact multi-way combination when claiming novelty. Never imply "nobody does X" without an artifact check.
- Package by what the object *does in the system*, not by where it sits on a technical ladder.

---

## 12. Working rules (carry these over)

**Communication.** Terse, mechanism-first, peer-level. Dense direct technical responses. Honest collision detection. Explicit self-correction when wrong. Early kill decisions. No hedging or encouragement framing over honest negative assessments. Plain-language rewrites on request — jargon density gets high and that is fine as long as a plain version is available.

**Experimental discipline.**
- Pre-registered diagnostic batteries with explicit **STOP criteria** written before the run.
- **Matched-budget** across arms, always. Matched parameters *and* matched on-device cost.
- Integrity checks before interpreting any result.
- **Check degenerate baselines before claiming a win** (the M1 lesson).
- Hand-transcribable output blocks required — the office GPU is air-gapped and results are transcribed manually. Design briefs to emit small structured result blocks.
- **Fail fast on cheap experiments before building expensive infrastructure.** Measure predictability ceilings (free, no training) before any Stage A training.

**Prior-art checking.**
- Literature surveys before committing resources.
- arXiv fetches to verify **specific internal claims** of cited papers, not just abstracts. Several claims in this program only held up or fell apart at that level of detail.

**Idea generation pipeline.** Multi-agent ChatGPT setup with an 11-operator taxonomy: Graft, Beat-the-block, Relax-an-assumption, Swap-the-objective, Re-axis, Transfer-across-surfaces, Unify-then-fill, Use-the-idle-signal, Push-the-regime, Hardware-invert, Sequential-to-joint. Includes a debate layer with a persistent objection ledger and saturation-based termination, plus a mathematical screening module using **reason-bearing tell detection, not forced categorization** — "None" is a first-class expected output, with a hard non-propagation rule preventing unverified tells from influencing ranking or novelty scores.

---

## 13. Tools, models, repos, hardware

**Compute**
- Jarvis AI Lab A100 80GB — primary.
- Office machine, RTX Pro 6000 (96GB Blackwell or 48GB Ada variant) — longer runs. **Air-gapped.** Results hand-transcribed. Only practical concern is CUDA kernel compatibility on Blackwell.

**Models**
- FlexDraft Qwen3-8B checkpoint — primary evaluation target.
- Qwen3-1.7B — earlier ablations.

**Evaluation**
- GSM8K, standard benchmark.
- τ / TPI primary. Cost-adjusted tokens/sec for shipping decisions.

**Repos**
- `8BitSpacemanSpiff/bita_diffusion` — **private.** Branch `placement-3b`. Holds the tree experiment code.
- `8BitSpacemanSpiff/VOLT` — branches `master`, `volt-rung0`. **Must stay private.** Contains unpublished findings: the amended STOP-2 verdict, the hazard-saturation result, the VID successor scope document. Live attribution exposure.

**Execution**
- Claude Code as autonomous experiment agent across fresh VM instances.
- Context bootstrap documents + `CLAUDE_TASK` briefs for session handoff.

**GitHub API pattern used for artifact checks**
- Unauthenticated `curl` against `api.github.com`, `recursive=1` tree traversal.
- Raw content via `raw.githubusercontent.com`.
- Visibility confirmed via `-w "%{http_code}"` (403 = private).

**Papers to have on hand (with IDs where known)**
FlexDraft · Apple MTP (2507.11851) · Saguaro (2603.03251) · DDTree (2604.12989) · JetSpec (2606.18394) · DominoTree (2607.08642) · Domino (2605.29707) · SpecBlock (2605.07243) · VSD (2602.05774) · PARD-2 (2605.08632) · D-PACE (2605.18810) · Flatter-Tokens (2601.18902) · CaDDTree (2606.01813) · DeLS-Spec (Zheng & Li, NUAA) · HASS · CORAL · DVI · POSS · DSpark · Hydra · EAGLE / EAGLE-2 · Medusa · Sequoia · D²SD · DFlash · SD² (Berdoz et al., AAAI-26) · Future Lens (Pal et al. 2023) · P-Tuning v2 · BitFit · LayerScale (CaiT 2021) · IA³ · SSF

---

## 14. First actions in the new account

Ordered. The first two unblock everything else.

1. **Pull the rung-1 numbers off the Jarvis machine.** Files: `results_cellpair.md`, the rung-1 results file, `STATUS.log`. Check against the five pre-registered predictions in §4.5. If Δτ landed and late-slot pos-acc moved as predicted, there is a two-rung positive result sitting on top of a designed null — that is already a paper skeleton.
2. **Measure the marginalization floor.** Gates the recurrent-head 2×2 and the per-slot-parameter direction.
3. **Run the V2 free measurement.** Early-exit hit rate at rejection positions: top-1 vs top-k, binary-reject vs k-way-correct, as a function of layer. One measurement ranks V1, repulsion, and soft-prior against each other. Costs nothing.
4. **Run the pending sampling-regime tree experiment.** Same conditions as the greedy run (Qwen3-8B, GSM8K, block 16, tree budget 30, frozen FlexDraft weights). The greedy result was structurally predetermined; sampling is the real test.
5. **Read Apple MTP's loss section.** Decides how narrowly the consistency-loss delta must be characterized. One hour.
6. **Confirm the two open code checks on FlexDraft:** (a) does the draft branch reuse verifier K/V for prefix positions or re-project them; (b) confirm the sample-and-match finding once more before publishing it.
7. **Decide the ENLSP object** — hazard decomposition vs design-space map — and start the 2-week analysis-only draft.
8. **Verify both repos are private.**

---

## 15. Standing principles (the short list)

- **r ≈ 1 is the decisive constraint.** Any method where draft and verify share the same full model cannot net positive through iterative refinement.
- **Fail fast on cheap experiments before building expensive infrastructure.**
- **Check degenerate baselines before claiming a result.**
- **Training on cached activations is wrong for interleaved co-training.** Cached vanilla activations are the wrong artifact.
- **KD warmup before RL is mandatory** for any co-trained rung.
- **Name the exact combination when claiming novelty.** Eight claims have died on artifact checks in this program.
- **Failed experiments are findings.** Framing matters for attribution and for morale.
- **Capacity-ladder empirics compress the design space.** The absent rank-1 rung and the collapsible slot axis together rule out a large set of variants without training any of them.
