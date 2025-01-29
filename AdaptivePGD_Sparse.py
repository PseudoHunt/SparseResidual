import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np

def soft_thresholding(X, tau):
    """Soft-thresholding for sparsity (L1 regularization)."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def elastic_net_thresholding(X, lambda1, lambda2):
    """Elastic Net thresholding (L1 + L2 regularization)."""
    return np.sign(X) * np.maximum(np.abs(X) - lambda1, 0) / (1 + lambda2)

def sparse_tucker_fista_optimized(original_tensor, ranks, lambda1, lambda2, 
                                  lr=0.01, lr_decay=0.95, max_iter=100, tol=1e-4):
    """
    Optimized FISTA-based Sparse Tucker Decomposition with Adaptive Learning and Elastic Net.

    Parameters:
    - original_tensor: Input tensor
    - ranks: Low-rank Tucker decomposition dimensions
    - lambda1: L1 regularization strength (sparsity)
    - lambda2: L2 regularization strength (stability)
    - lr: Initial learning rate
    - lr_decay: Factor for learning rate decay
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
    t = 1  # Momentum term
    learning_rate = lr  # Adaptive learning rate

    for _ in range(max_iter):
        # Compute gradient step
        grad = tl.tucker_to_tensor(Y, factors) - original_tensor
        Y_new = Y - learning_rate * grad  # Gradient update

        # Apply Elastic Net thresholding (L1 + L2 regularization)
        core_new = elastic_net_thresholding(Y_new, lambda1, lambda2)

        # Update momentum term (FISTA acceleration)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        Y = core_new + ((t - 1) / t_new) * (core_new - core)

        # Learning rate decay
        learning_rate *= lr_decay

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

# Set regularization parameters
lambda1 = 0.1  # L1 sparsity strength
lambda2 = 0.01  # L2 stability strength

# Run optimized FISTA-based sparse Tucker decomposition
core, factors = sparse_tucker_fista_optimized(original_tensor, ranks, lambda1, lambda2)

print("Core Tensor Shape:", core.shape)
for i, factor in enumerate(factors):
    print(f"Factor {i} Shape:", factor.shape)
