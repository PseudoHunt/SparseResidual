import torch
import tensorly as tl
from tensorly.decomposition import tucker

# Set tensorly backend to PyTorch
tl.set_backend('pytorch')

def soft_thresholding(X, tau):
    """Applies soft-thresholding for L1 sparsity (GPU-compatible)."""
    return torch.sign(X) * torch.relu(torch.abs(X) - tau)

def elastic_net_thresholding(X, lambda1, lambda2):
    """Elastic Net thresholding (L1 + L2 regularization, GPU-compatible)."""
    return torch.sign(X) * torch.relu(torch.abs(X) - lambda1) / (1 + lambda2)

def sparse_tucker_fista_gpu(original_tensor, ranks, lambda1, lambda2, 
                            lr=0.01, lr_decay=0.95, max_iter=100, tol=1e-4, device="cuda"):
    """
    GPU-accelerated FISTA-based Sparse Tucker Decomposition with Adaptive Learning.

    Parameters:
    - original_tensor: Input tensor (PyTorch tensor, moved to GPU if available)
    - ranks: Low-rank Tucker decomposition dimensions
    - lambda1: L1 sparsity regularization parameter
    - lambda2: L2 stability regularization parameter
    - lr: Initial learning rate
    - lr_decay: Learning rate decay factor
    - max_iter: Maximum iterations
    - tol: Convergence threshold
    - device: "cuda" or "cpu"

    Returns:
    - core: Low-rank core tensor (on GPU)
    - factors: Factor matrices (on GPU)
    """

    # Move tensor to GPU
    original_tensor = original_tensor.to(device)

    # Initial Tucker decomposition (GPU-accelerated)
    core, factors = tucker(original_tensor, ranks, init='random', tol=tol, n_iter_max=100)

    # Move tensors to GPU
    core = core.to(device)
    factors = [f.to(device) for f in factors]

    # Initialize FISTA variables
    Y = core.clone()  # Momentum variable
    t = 1  # Momentum term
    learning_rate = lr  # Adaptive learning rate

    for _ in range(max_iter):
        # Compute gradient step (GPU)
        grad = tl.tucker_to_tensor(Y, factors) - original_tensor
        Y_new = Y - learning_rate * grad  # Gradient update

        # Apply Elastic Net thresholding (L1 + L2 regularization) on GPU
        core_new = elastic_net_thresholding(Y_new, lambda1, lambda2)

        # Update momentum term (FISTA acceleration)
        t_new = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        Y = core_new + ((t - 1) / t_new) * (core_new - core)

        # Adaptive learning rate decay
        learning_rate *= lr_decay

        # Convergence check (GPU computation)
        error = torch.norm(core_new - core) / torch.norm(original_tensor)
        if error < tol:
            break

        # Update variables
        core = core_new
        t = t_new

    return core, factors
import tensorly.random as tlr

# Create a random tensor (move to GPU)
original_tensor = tlr.random_tensor((100, 100, 100)).to("cuda")

# Define Tucker ranks
ranks = [50, 50, 50]

# Set regularization parameters
lambda1 = 0.1  # L1 sparsity strength
lambda2 = 0.01  # L2 stability strength

# Run GPU-accelerated sparse Tucker decomposition
core, factors = sparse_tucker_fista_gpu(original_tensor, ranks, lambda1, lambda2)

print("Core Tensor Shape:", core.shape)
for i, factor in enumerate(factors):
    print(f"Factor {i} Shape:", factor.shape)
