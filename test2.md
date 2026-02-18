Hey Babsey, here to help you!

Cool — let’s do the basic DARTS-style LoRA rank search, exactly as you described: multiple rank branches, one \alpha per branch, sigmoid gating, plus an entropy regularizer so it collapses to one branch by the end.

I’ll write it cleanly and in a way that’s implementable.

⸻

1) Multi-branch LoRA for a single layer

Base linear:
y = Wx

Create K LoRA “rank branches”, each with its own (A_k,B_k) and rank r_k:
\Delta W_k = B_k A_k,
\quad A_k \in \mathbb{R}^{r_k \times d_{in}},\; B_k \in \mathbb{R}^{d_{out} \times r_k}

Gate each branch with a sigmoid on one scalar \alpha_k:
g_k = \sigma(\alpha_k)

Then:
y = Wx + \sum_{k=1}^{K} g_k \, (B_k(A_k x))

So you’re searching “which rank branch to use” via \alpha.

⸻

2) Make it behave like “choose exactly one branch”

If you only use independent sigmoids, the model can keep multiple branches partially on.

So you add a regularizer that encourages a one-hot selection.

(A) Normalize to a distribution

Turn gates into a probability distribution:
p_k = \frac{g_k}{\sum_{j=1}^K g_j + \epsilon}
Now p sums to 1, so entropy makes sense.

(B) Entropy regularization (collapse to one)

Entropy:
H(p) = -\sum_{k=1}^{K} p_k \log(p_k + \epsilon)

To encourage one branch, you want low entropy, so you add entropy with a negative sign (or subtract it):
\mathcal{L} = \mathcal{L}_{task} + \lambda H(p)
…but careful: minimizing \mathcal{L} with +\lambda H would push entropy down or up?
	•	If you minimize and you add +\lambda H, you’ll try to minimize entropy only if \lambda is positive and you minimize? Actually entropy is ≥ 0, so minimizing +\lambda H drives it down → good.
So:
\boxed{\mathcal{L} = \mathcal{L}_{task} + \lambda H(p)}\quad (\lambda>0)
works to push toward one-hot.

(C) Add “exactly one active” constraint (optional but helps a lot)

Entropy alone can still shrink all g_k toward 0. So add:
\mathcal{L}_{count} = \mu \left(\sum_{k} g_k - 1\right)^2
This softly enforces “one branch on”.

Final:
\boxed{\mathcal{L}=\mathcal{L}_{task}+\lambda H(p)+\mu(\sum_k g_k-1)^2}

⸻

3) DARTS bilevel optimization (basic form)

You typically split data into train/val:
	•	LoRA weights \theta = \{A_k,B_k\} updated on training loss
	•	Architecture \alpha = \{\alpha_k\} updated on validation loss + regularizers

Updates
	1.	Weight step:
\theta \leftarrow \theta - \eta_\theta \nabla_\theta \mathcal{L}_{train}(\theta,\alpha)
	2.	Architecture step:
\alpha \leftarrow \alpha - \eta_\alpha \nabla_\alpha \Big(\mathcal{L}_{val}(\theta,\alpha)+\lambda H(p)+\mu(\sum g_k-1)^2\Big)

You can use first-order DARTS (ignore second-order terms) to keep it simple.

⸻

4) How you “choose one alpha” at the end

After search training, per layer:
	•	compute g_k=\sigma(\alpha_k)
	•	choose branch:
k^* = \arg\max_k g_k
Then drop the other branches and keep only LoRA of rank r_{k^*}. Usually you do a short retrain/finalize pass.

⸻

5) Practical training schedule (so it actually collapses)

This works best with a schedule:
	•	Warm-up: freeze \alpha for N steps so \theta learns something
	•	Anneal: increase \lambda (entropy) and \mu (sum-to-1) over time
	•	early: explore (small \lambda,\mu)
	•	late: collapse (large \lambda,\mu)

Also keep \eta_\alpha smaller than \eta_\theta.

⸻

6) One important note: sigmoid vs softmax

Your setup (sigmoid per branch) is fine, but if you ever get instability, the softmax version is the classic “exactly-one” relaxation:

p_k=\text{softmax}(\alpha/\tau)_k,\quad
y=Wx+\sum_k p_k \Delta W_k x

Then entropy regularization is even more natural. But we can stay with sigmoid + normalization exactly as you asked.

⸻

If you want, I’ll write the full PyTorch module for one linear layer with K LoRA branches + the exact loss terms H(p) and (\sum g-1)^2, and the alternating DARTS update loop.