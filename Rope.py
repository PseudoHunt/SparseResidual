def generate_2d_sinusoidal_pe(self, H, W, d, base_freq=0.2, freq_decay=0.9):
    coords = torch.tensor([(i, j) for i in range(H) for j in range(W)], dtype=torch.float, device=self.angle.device)
    x, y = coords[:, 0], coords[:, 1]

    freqs = base_freq * (freq_decay ** torch.arange(d // 4, device=self.angle.device))
    pe = torch.zeros((H * W, d), device=self.angle.device)
    pe[:, 0::4] = torch.sin(torch.outer(x, freqs))
    pe[:, 1::4] = torch.cos(torch.outer(x, freqs))
    pe[:, 2::4] = torch.sin(torch.outer(y, freqs))
    pe[:, 3::4] = torch.cos(torch.outer(y, freqs))
    return pe
#replace this section 
mask = self.generate_2d_decay(slen[0], slen[1])
retention_rel_pos = ((sin, cos), mask)

#with this
# Sinusoidal PE as similarity mask
pe = self.generate_2d_sinusoidal_pe(slen[0], slen[1], d=self.angle.shape[0] * 2)
pe_mask = pe @ pe.T  # (L, L) similarity matrix
pe_mask = pe_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)  # (n, L, L)
retention_rel_pos = ((sin, cos), pe_mask)

#normalize
pe_mask = pe_mask / (pe_mask.abs().max() + 1e-6)
