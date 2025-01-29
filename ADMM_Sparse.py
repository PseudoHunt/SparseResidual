import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np

def soft_thresholding(X, tau):
    """Applies soft thresholding to promote sparsity."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def sparse_tucker_admm(original_tensor, ranks, tau, rho=1.0, max_iter=50, tol=1e-4):
    """
    ADMM-based sparse Tucker decomposition.
    
    Parameters:
    - original_tensor: Input tensor
    - ranks: Low-rank Tucker decomposition dimensions
    - tau: Regularization strength for sparsity
    - rho: ADMM penalty parameter
    - max_iter: Maximum ADMM iterations
    - tol: Convergence threshold

    Returns:
    - core: Low-rank core tensor
    - factors: Factor matrices
    - sparse_residual: Sparsified residual tensor
    """
    # Initialize with a standard Tucker decomposition
    core, factors = tucker(original_tensor, ranks)
    
    # Initialize auxiliary variables for ADMM
    Z = np.zeros_like(original_tensor)  # Sparsified residual
    U = np.zeros_like(original_tensor)  # Lagrange multiplier

    for _ in range(max_iter):
        # Step 1: Update the Tucker decomposition (solve min ||original - (core, factors) - Z + U||)
        X_tilde = original_tensor - Z + U
        core, factors = tucker(X_tilde, ranks, tol=tol, init='random', n_iter_max=100)

        # Step 2: Update Z using soft thresholding for sparsity
        residual = original_tensor - tl.tucker_to_tensor(core, factors) + U
        Z = soft_thresholding(residual, tau)

        # Step 3: Update Lagrange multiplier U
        U += residual - Z

        # Convergence check
        error = np.linalg.norm(residual - Z) / np.linalg.norm(original_tensor)
        if error < tol:
            break

    return core, factors, Z

import tensorly.random as tlr

# Create a random tensor
original_tensor = tlr.random_tensor((10, 10, 10))

# Define Tucker ranks (low-rank approximation)
ranks = [5, 5, 5]

# Set sparsity parameter tau
tau = 0.1

# Run ADMM-based sparse Tucker decomposition
core, factors, sparse_residual = sparse_tucker_admm(original_tensor, ranks, tau)

print("Core Tensor Shape:", core.shape)
for i, factor in enumerate(factors):
    print(f"Factor {i} Shape:", factor.shape)

print("Sparse Residual Shape:", sparse_residual.shape)
