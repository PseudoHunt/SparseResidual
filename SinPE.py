import numpy as np
import matplotlib.pyplot as plt

# Function to generate 2D sinusoidal positional encodings
def generate_2d_sinusoidal_pe(grid_size, d, base_freq=0.2, freq_decay=0.9):
    rows, cols = grid_size
    num_positions = rows * cols
    coords = np.array([(i, j) for i in range(rows) for j in range(cols)])  # (num_positions, 2)

    frequencies = base_freq * (freq_decay ** np.arange(0, d // 4))  # (d/4,)

    pe = np.zeros((num_positions, d))

    # x part
    pe[:, 0::4] = np.sin(np.outer(coords[:, 0], frequencies))  # sin(x * w_k)
    pe[:, 1::4] = np.cos(np.outer(coords[:, 0], frequencies))  # cos(x * w_k)
    # y part
    pe[:, 2::4] = np.sin(np.outer(coords[:, 1], frequencies))  # sin(y * w_k)
    pe[:, 3::4] = np.cos(np.outer(coords[:, 1], frequencies))  # cos(y * w_k)

    return pe, coords

# Function to create exponential distance kernel on top of PE
def exponential_kernel_over_pe(pe, lambda_kernel=1.0):
    num_positions = pe.shape[0]
    pe_dist_matrix = np.zeros((num_positions, num_positions))

    # Compute squared Euclidean distances between PEs
    for i in range(num_positions):
        for j in range(num_positions):
            pe_dist_matrix[i, j] = np.sum((pe[i] - pe[j]) ** 2)

    # Apply exponential kernel
    exp_kernel = np.exp(-lambda_kernel * pe_dist_matrix)
    return exp_kernel

# Parameters
grid_size = (4, 4)  # 4x4 grid
d = 32  # Embedding dimension
lambda_kernel = 1.0  # Kernel sharpness
alpha_manhattan = 1.0  # Decay rate for real Manhattan decay

# Step 1: Generate 2D sinusoidal PEs
pe, coords = generate_2d_sinusoidal_pe(grid_size, d)

# Step 2: Generate exponential kernel over PE
pe_exp_kernel = exponential_kernel_over_pe(pe, lambda_kernel=lambda_kernel)

# Step 3: Compute real Manhattan decay matrix for comparison
num_patches = grid_size[0] * grid_size[1]
manhattan_decay_matrix = np.zeros((num_patches, num_patches))
for i in range(num_patches):
    for j in range(num_patches):
        dist = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
        manhattan_decay_matrix[i, j] = np.exp(-alpha_manhattan * dist)

# Step 4: Visualization
plt.figure(figsize=(18, 6))

# Exponential kernel over PE
plt.subplot(1, 3, 1)
plt.imshow(pe_exp_kernel, cmap='viridis')
plt.colorbar(label='Value')
plt.title('Exponential Kernel over Sinusoidal PE')

# Error Matrix
plt.subplot(1, 3, 2)
plt.imshow(np.abs(manhattan_decay_matrix - pe_exp_kernel), cmap='hot')
plt.colorbar(label='Absolute Error')
plt.title('Absolute Error vs. Manhattan Decay')

# Manhattan Decay Matrix (Reference)
plt.subplot(1, 3, 3)
plt.imshow(manhattan_decay_matrix, cmap='viridis')
plt.colorbar(label='Decay Value (γ_ij)')
plt.title('Manhattan Decay Matrix (Reference)')

plt.tight_layout()
plt.show()

# Step 5: Return relative error for inspection
relative_error = np.linalg.norm(manhattan_decay_matrix - pe_exp_kernel) / np.linalg.norm(manhattan_decay_matrix)
relative_error
