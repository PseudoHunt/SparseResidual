# Generate multi-head Fourier PE with different directional projections per head
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib.pyplot
def generate_multihead_rotated_pe(grid_size, d, angles_deg, base_freq=0.2, freq_decay=-0.9):
    rows, cols = grid_size
    coords = np.array([(i, j) for i in range(rows) for j in range(cols)])
    x, y = coords[:, 0], coords[:, 1]

    num_heads = len(angles_deg)
    d_per_head = d // num_heads
    pe_all_heads = []

    for theta_deg in angles_deg:
        theta = np.deg2rad(theta_deg)
        z = x * np.cos(theta) + y * np.sin(theta)

        num_freqs = d_per_head // 2
        frequencies = base_freq * (freq_decay ** np.arange(num_freqs))
        pe = np.zeros((len(coords), d_per_head))
        for i, freq in enumerate(frequencies):
            pe[:, 2 * i]     = np.sin(freq * z)
            pe[:, 2 * i + 1] = np.cos(freq * z)

        pe_all_heads.append(pe)

    return pe_all_heads, coords

# Define directional angles per head
angles_deg = [0, 45, 90, 135]  # horizontal, \ diagonal, vertical, / diagonal
d = 32
grid_size = (7, 7)
center_pos = (grid_size[0] // 2, grid_size[1] // 2)
# Generate PE for all heads
pe_heads, coords_mh = generate_multihead_rotated_pe(grid_size, d, angles_deg)
center_idx_mh = [tuple(coord) for coord in coords_mh].index(center_pos)

# Plot similarity from center pixel for each head
fig, axes = plt.subplots(1, len(angles_deg), figsize=(16, 4))

for i, pe in enumerate(pe_heads):
    sim = pe @ pe.T
    sim_grid = sim[center_idx_mh].reshape(grid_size)
    axes[i].imshow(sim_grid, cmap='viridis')
    axes[i].set_title(f"Head {i+1}\nθ = {angles_deg[i]}°")
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.suptitle("Multi-head Rotated Fourier PE (Similarity from Center Pixel)", fontsize=14)
plt.tight_layout()
plt.show()
