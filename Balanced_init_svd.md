# Balanced Initialization and Conditioning Regularization for Factorized Layers

## 1. Problem

We factorize a dense weight matrix $W \in \mathbb{R}^{m \times n}$ using a rank-$r$ SVD:

$$W \approx U S V^\top, \qquad S = \mathrm{diag}(s_1, \dots, s_r), \quad s_1 \ge s_2 \ge \dots \ge s_r$$

The standard practice is to fold $S$ into one side:

$$A = U, \qquad B = S V^\top$$

Real spectra decay close to exponentially, $s_i \approx s_1 e^{-\lambda i}$. So the row norms of $B$ span several orders of magnitude:

$$\frac{\|B_{1,:}\|}{\|B_{r,:}\|} = \frac{s_1}{s_r} = \kappa$$

with $\kappa$ often $10^4$ to $10^6$. The result is a factor pair with a huge dynamic range. This is a conditioning problem in the parameterization itself, and it is present from step 0 of training.

## 2. Gauge freedom

The factorization is not unique. For any invertible $M \in \mathbb{R}^{r \times r}$:

$$A = U S^{1/2} M, \qquad B = M^{-1} S^{1/2} V^\top \;\;\Longrightarrow\;\; AB = U S V^\top$$

The layer computes the exact same function for every choice of $M$. The initial loss is unchanged. Only the conditioning of the two factors changes. So $M$ is free to choose, and it costs nothing to choose it well.

## 3. Proposed initialization

**Step 1 — split the spectrum.** Set $M = I$, giving $A = US^{1/2}$, $B = S^{1/2}V^\top$. Column norms of $A$ and row norms of $B$ are both $s_i^{1/2}$, so each factor now has dynamic range $\kappa^{1/2}$ instead of $\kappa$.

This is provably the best possible *diagonal* $M$. With $A = US^\alpha$ and $B = S^{1-\alpha}V^\top$, the worse of the two ranges is

$$\max\left(\kappa^{\alpha},\; \kappa^{1-\alpha}\right)$$

which is minimized at $\alpha = 1/2$. To do better we must leave the diagonal group.

**Step 2 — mix across the rank axis.** Set $M = Q$, an orthogonal matrix with entries of equal magnitude $|q_{ij}| = 1/\sqrt{r}$ (a Hadamard or DFT matrix). Column $j$ of $A$ becomes $\sum_i q_{ij}\sqrt{s_i}\,u_i$, with norm

$$\|A_{:,j}\| = \sqrt{\sum_i q_{ij}^2 s_i} = \sqrt{\frac{1}{r}\sum_i s_i} \quad \text{for all } j$$

Every column now carries identical energy, regardless of how fast $S$ decays. The initialization is therefore **energy-isotropic along the rank axis**: every latent factor starts with the same second-order energy. Since $Q$ is orthogonal, $M^{-1} = Q^\top$ and the inverse can never blow up.

**Why Hadamard and not a random orthogonal matrix.** A Haar-random orthogonal $Q$ only gives this in expectation, $\mathbb{E}[q_{ij}^2] = 1/r$, so column energies are equalized up to $O(1/\sqrt{r})$ fluctuation and vary run to run. A Hadamard matrix satisfies $|q_{ij}| = 1/\sqrt{r}$ *exactly*, so the equalization is exact, deterministic, reproducible across runs, and applicable in $O(r \log r)$ via the fast transform. Random signs on the diagonal can still be applied if we want to break any accidental alignment with the $u_i$.

Both steps are one-time, closed-form, and applied before training starts.

## 4. Training regularizers

Balance at init decays during training under Adam and weight decay. Two cheap penalties hold it:

**Balancedness.**

$$\mathcal{R}_{\text{bal}} = \lambda \left\| A^\top A - B B^\top \right\|_F^2$$

**Soft orthogonality.**

$$\mathcal{R}_{\text{orth}} = \lambda \left\| A^\top A - cI \right\|_F^2, \qquad c = \tfrac{1}{r}\|A\|_F^2$$

Both are $O(mr^2)$ per step, negligible next to the forward pass.

**Regularize the product, not the factors.** Standard weight decay on $\|A\|_F^2 + \|B\|_F^2$ is not gauge-invariant, so it penalizes the same function differently depending on how the factorization happens to be balanced. We adopt function-space rather than factor-space regularization:

$$\mathcal{R}_{\text{FD}} = \lambda \|AB\|_F^2$$

This is Frobenius decay from Khodak et al., adopted as-is. Likewise, the balancedness and soft-orthogonality penalties above are standard tools from the implicit-regularization literature. None of §4 is new; it exists to hold the §3 initialization in place during training.

## 5. Why balance helps the trained model

This is the part that stands on its own, with no reference to any downstream deployment step.

**Conserved imbalance.** Under gradient flow, the quantity $A^\top A - BB^\top$ is invariant:

$$\frac{d}{dt}\left(A^\top A - B B^\top\right) = 0$$

<cite index="21-1">Du, Hu and Lee prove that gradient flow keeps the differences between squared norms across layers invariant with no explicit regularization.</cite> This holds exactly only for gradient flow. Real training uses Adam, momentum, weight decay and finite step sizes, so the invariant is approximate.

The prediction is therefore: **balanced initialization has a lasting effect, because balancedness is approximately preserved during optimization, and correspondingly an imbalance present at init is only slowly corrected — if at all.** A factorization starting with $\kappa = 10^6$ of imbalance is expected to carry most of it for the run. That makes init the cheapest point of intervention, and §4 exists to absorb the drift that the finite-step-size gap introduces.

**Effective learning rate.** The gradients are $\nabla_A \mathcal{L} = G B^\top$ and $\nabla_B \mathcal{L} = A^\top G$, where $G = \partial\mathcal{L}/\partial(AB)$. When $\|B\| \gg \|A\|$, the two factors receive gradients of very different scale, so one factor moves and the other is effectively frozen. Balancing equalizes the step sizes.

**Floating-point headroom.** In BF16 (8 mantissa bits), a factor spanning $10^6$ in magnitude has increased susceptibility to rounding on its small entries. Those small entries hold the low-singular-value directions — exactly the fine structure the decomposition is meant to preserve. This is measurable directly (gradient variance, underflow/overflow counts, optimizer second-moment statistics) rather than assumed.

**Landscape geometry (hypothesis).** We hypothesize that balanced factors improve optimization conditioning by reducing anisotropy in the factor parameterization, and that this shows up as faster and more stable convergence. This is stated as a hypothesis to be verified, not a theoretical guarantee.

**No loss of capacity.** The reachable function class is still exactly rank-$r$. We change the optimization path, not the hypothesis space. The change is upside-only.

## 6. Literature

| Reference | Relevance |
|---|---|
| Khodak, Tenenholtz, Mackey, Fusi. *Initialization and Regularization of Factorized Neural Layers.* ICLR 2021. arXiv:2105.01029 | Closest prior work. <cite index="20-1">Studies spectral initialization, which initializes factors by SVD so their product approximates the target matrix, and Frobenius decay, which regularizes the product of the matrices rather than the individual factors.</cite> <cite index="12-1">They show that proper SVD initialization and regularizing the matrix product make factorized networks competitive with full-rank counterparts.</cite> Code: `microsoft/fnl_paper` |
| Du, Hu, Lee. *Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are Automatically Balanced.* NeurIPS 2018. arXiv:1806.00900 | The conservation law in §5. <cite index="21-1">Proves gradient flow keeps cross-layer squared-norm differences invariant, and analyzes gradient descent for asymmetric low-rank matrix factorization.</cite> |
| Arora, Cohen, Hu, Luo. *Implicit Regularization in Deep Matrix Factorization.* NeurIPS 2019. | Balancedness and implicit bias in the factorized setting. |
| Arora, Cohen, Hazan. *On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization.* ICML 2018. | Overparameterized factorization as a preconditioner; supports the effective-learning-rate argument. |
| Bansal, Chen, Wang. *Can We Gain More from Orthogonality Regularizations in Training Deep Networks?* NeurIPS 2018. | Empirical evidence that soft-orthogonality penalties improve accuracy and convergence. |

## 7. Validation plan

Small enough to run on one model, and it separates what is proven from what is claimed.

1. **Initialization only, no regularizers.** Compare three inits — $(U,\,SV^\top)$, square-root split, square-root + Hadamard — on identical training runs. Report loss curves and final metric. This isolates the §3 contribution.
2. **Numerical behaviour under BF16.** Log gradient variance, over/underflow counts, and Adam second-moment spread per factor. Tests the precision claim directly.
3. **Rank sweep.** $r \in \{16, 32, 64, 128\}$. The gap should widen with $r$, since $\kappa$ grows with the retained spectrum.
4. **Balance drift.** Track $\|A^\top A - BB^\top\|_F$ over training, with and without §4. Quantifies how far real optimizers depart from the gradient-flow invariant.

## 8. Ownership note

A downstream numerical-robustness fix operates on already-trained weights. Its best case is to add zero error — it can never produce a better model than training did. A conditioning fix at initialization changes the optimum the model converges to, so its expected outcome is better-or-equal, and the well-conditioned factors are a free by-product.

The two also compose: fixing conditioning here does not remove any downstream option, it only reduces the work required there. And conditioning of the factorization is a property of this model, so it belongs on this side regardless of what any consumer does with it.
