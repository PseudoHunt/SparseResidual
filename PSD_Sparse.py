import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np

def soft_thresholding(X, tau):
    """Applies soft thresholding to enforce sparsity."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def sparse_tucker_fista(original_tensor, ranks, tau, lr=0.01, max_iter=100, tol=1e-4):
    """
    FISTA-based sparse Tucker decomposition.
    
    Parameters:
    - original_tensor: Input tensor
    - ranks: Low-rank Tucker decomposition dimensions
    - tau: Sparsity regularization parameter
    - lr: Learning rate for gradient descent
    - max_iter: Maximum iterations
    - tol: Convergence threshold

    Returns:
    - core: Low-rank core tensor
    - factors: Factor matrices
    """
    # Initial Tucker decomposition
    core, factors = tucker(original_tensor, ranks)
    
    # Initialize FISTA variables
    Y = core.copy()  # Momentum variable
    t = 1  # FISTA momentum

    for _ in range(max_iter):
        # Compute gradient step
        grad = tl.tucker_to_tensor(Y, factors) - original_tensor
        Y_new = Y - lr * grad  # Gradient update

        # Apply soft-thresholding for sparsity
        core_new = soft_thresholding(Y_new, tau)

        # Update momentum term (FISTA acceleration)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        Y = core_new + ((t - 1) / t_new) * (core_new - core)

        # Convergence check
        if np.linalg.norm(core_new - core) / np.linalg.norm(original_tensor) < tol:
            break

        # Update variables
        core = core_new
        t = t_new

    return core, factors

import tensorly.random as tlr

# Create a random tensor
original_tensor = tlr.random_tensor((10, 10, 10))

# Define Tucker ranks
ranks = [5, 5, 5]

# Set sparsity parameter tau
tau = 0.1

# Run FISTA-based sparse Tucker decomposition
core, factors = sparse_tucker_fista(original_tensor, ranks, tau)

print("Core Tensor Shape:", core.shape)
for i, factor in enumerate(factors):
    print(f"Factor {i} Shape:", factor.shape)
