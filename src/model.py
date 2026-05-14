# -*- coding: utf-8 -*-
"""
model.py
---------
Core model components shared across all experiments (real & synthetic).

Implements:
    • PerRegionGRU       : temporal encoder per region
    • TimePositional     : time embedding
    • PhaseGraphs        : phase-specific signed adjacency matrices
    • GraphProjector     : message-passing with spectral normalization
    • ARHead             : autoregressive temporal decoder with attention
    • BACE               : full Behavior-Adaptive Connectivity Estimator

The architecture is identical across real and synthetic data.
For real data, each region has 10 channels (C=10);
for synthetic data, each region has 1 channel (C=1).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ==============================================================
#  Per-region temporal encoder (GRU)
# ==============================================================

class PerRegionGRU(nn.Module):
    """
    One GRU per region to encode local temporal dynamics.
    Input:  [B, N, C, T]
    Output: [B, N, T, d_hidden]
    """
    def __init__(self, N: int, C: int, d_hidden: int):
        super().__init__()
        self.N = N
        self.grus = nn.ModuleList([
            nn.GRU(input_size=C, hidden_size=d_hidden, batch_first=False)
            for _ in range(N)
        ])
        self.ln = nn.LayerNorm(d_hidden)

    def forward(self, x):
        B, N, C, T = x.shape
        outs = []
        for r in range(N):
            xr = x[:, r].permute(2, 0, 1)    # [T, B, C]
            y, _ = self.grus[r](xr)          # [T, B, d_hidden]
            y = self.ln(y)
            outs.append(y.permute(1, 0, 2))  # [B, T, d_hidden]
        return torch.stack(outs, dim=1)      # [B, N, T, d_hidden]


# ==============================================================
#  Temporal positional embedding
# ==============================================================

class TimePositional(nn.Module):
    """Adds a simple time embedding (learned linear projection of normalized t)."""
    def __init__(self, d_time: int, T_in: int):
        super().__init__()
        self.emb = nn.Linear(1, d_time)
        self.T_in = T_in

    def forward(self, B: int, T: int, device=None):
        t = torch.linspace(0., 1., steps=T, device=device).view(1, T, 1)
        t = t.repeat(B, 1, 1)
        return torch.tanh(self.emb(t))  # [B, T, d_time]


# ==============================================================
#  Phase-specific graph module
# ==============================================================

class PhaseGraphs(nn.Module):
    """
    Learns a signed adjacency per behavioral phase.
    Each row normalized to L1 = 1 (preserving directionality).
    Optional row-gain reweights outgoing influence per node.
    """
    def __init__(self, N: int, P: int, eps: float = 1e-6, use_row_gain: bool = True):
        super().__init__()
        self.N, self.P, self.eps = N, P, eps
        self.S = nn.Parameter(torch.zeros(P, N, N))   # signed pattern
        self.register_buffer("I", torch.eye(N))
        self.use_row_gain = use_row_gain
        if use_row_gain:
            self.G = nn.Parameter(torch.zeros(P, N))  # per-phase, per-row gains

    def _zero_diag(self, X):
        return X * (1.0 - self.I)

    def _row_norm_l1_signed(self, S):
        S = self._zero_diag(S)
        denom = S.abs().sum(-1, keepdim=True).clamp_min(self.eps)
        return S / denom

    def forward(self, phases: torch.Tensor):
        A_tilde = self._row_norm_l1_signed(self.S)  # [P,N,N]
        A = A_tilde[phases]
        if self.use_row_gain:
            g_all = F.softplus(self.G) + 1e-6
            g_all = g_all * (self.N / g_all.sum(-1, keepdim=True).clamp_min(self.eps))
            g = g_all[phases].unsqueeze(-1)         # [B,N,1]
            A = A * g
        return A

    @torch.no_grad()
    def export_pattern(self):
        return self._row_norm_l1_signed(self.S).cpu().numpy()

    @torch.no_grad()
    def export_eff(self):
        A = self._row_norm_l1_signed(self.S)
        if self.use_row_gain:
            g = F.softplus(self.G) + 1e-6
            g = g * (self.N / g.sum(-1, keepdim=True).clamp_min(self.eps))
            A = A * g[..., None]
        return A.detach().cpu().numpy()

    @torch.no_grad()
    def init_from_correlation(self, C_list):
        """Initialize S from per-phase correlation matrices (train set only)."""
        for p in range(self.P):
            M = torch.as_tensor(C_list[p], dtype=torch.float32)
            M.fill_diagonal_(0.0)
            self.S[p].copy_(M)
        if self.use_row_gain:
            self.G.zero_()


# ==============================================================
#  Graph projector (message passing)
# ==============================================================

class GraphProjector(nn.Module):
    """
    Linear self + neighbor aggregation with spectral normalization.
    H_seq: [B, N, T, d_in]
    A_batch: [B, N, N]
    Returns: [B, N, T, d_out]
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_self = nn.Linear(d_in, d_out, bias=False)
        self.W_neigh = spectral_norm(nn.Linear(d_in, d_out, bias=False))

    def forward(self, H_seq, A_batch):
        Hs = self.W_self(H_seq)
        Hn = self.W_neigh(H_seq)
        Zn = torch.einsum("bij,bjtd->bitd", A_batch, Hn)
        return F.leaky_relu(Hs + Zn, 0.1)


# ==============================================================
#  Autoregressive decoder with attention
# ==============================================================

class ARHead(nn.Module):
    """
    Decodes T_out future steps autoregressively.
    Uses attention over encoder memory + optional neighbor feedback.
    """
    def __init__(self, d_in: int, C: int, T_out: int, h_dim=None, use_kv=True, attn_p=0.1):
        super().__init__()
        h_dim = d_in if h_dim is None else h_dim
        self.cell = nn.GRUCell(d_in + C, h_dim)
        self.read = nn.Linear(h_dim, C)
        self.T_out = T_out
        self.C = C
        self.use_kv = use_kv
        if use_kv:
            self.k = nn.Linear(d_in, d_in, bias=False)
            self.v = nn.Linear(d_in, d_in, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(d_in))))
        self.attn_drop = nn.Dropout(p=attn_p) if attn_p > 0 else nn.Identity()

    def _attend(self, q, K, V):
        scale = torch.exp(self.log_tau)
        scores = torch.bmm(K, q.unsqueeze(-1)).squeeze(-1) * scale
        alpha = torch.softmax(scores, dim=-1)
        alpha = self.attn_drop(alpha)
        ctx = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)
        return ctx

    def forward(self, Z_seq, x_last=None, teacher=None, sched_sampling_p=0.0, A_batch=None):
        B, N, T, d = Z_seq.shape
        BN = B * N
        M = Z_seq.view(BN, T, d)
        K = self.k(M) if self.use_kv else M
        V = self.v(M) if self.use_kv else M
        h = M[:, -1, :]
        y_prev = torch.zeros(BN, self.C, device=Z_seq.device) if x_last is None else x_last.view(BN, self.C)

        outs = []
        for _ in range(self.T_out):
            if A_batch is not None:
                y_prev_bnc = y_prev.view(B, N, self.C)
                alpha_g = 0.3
                y_prev_bnc = y_prev_bnc + alpha_g * torch.einsum("bij,bjc->bic", A_batch, y_prev_bnc)
                y_prev = y_prev_bnc.view(BN, self.C)
            ctx = self._attend(h, K, V)
            inp = torch.cat([ctx, y_prev], dim=-1)
            h = self.cell(inp, h)
            y_t = self.read(h)
            y_t = y_t + y_prev
            outs.append(y_t.view(B, N, self.C, 1))

            if self.training and (teacher is not None) and (sched_sampling_p > 0):
                use_pred = (torch.rand(BN, 1, device=Z_seq.device) < sched_sampling_p).float()
                teach_t = teacher[..., outs[-1].shape[-1]-1].view(BN, self.C)
                y_prev = use_pred * y_t + (1 - use_pred) * teach_t
            else:
                y_prev = y_t
        return torch.cat(outs, dim=-1)  # [B,N,C,T_out]


# ==============================================================
#  Full Behavior-Adaptive Connectivity Estimator (BACE)
# ==============================================================

class BACE(nn.Module):
    """
    Full Behavior-Adaptive Connectivity Estimator.

    Components:
        encoder  : PerRegionGRU
        tpos     : TimePositional
        graphs   : PhaseGraphs
        projector: GraphProjector
        head     : ARHead
    """
    def __init__(self, N, C, cfg):
        super().__init__()
        self.N, self.C = N, C
        self.enc = PerRegionGRU(N, C, cfg.d_hidden)
        self.tpos = TimePositional(cfg.d_timectx, cfg.T_in)
        self.graphs = PhaseGraphs(N, P=4, use_row_gain=cfg.use_row_gain)
        self.proj = GraphProjector(cfg.d_hidden + cfg.d_timectx, cfg.d_proj)
        self.head = ARHead(cfg.d_proj, C, cfg.T_out, use_kv=True, attn_p=0.1)

    def forward(self, X_in, phases, teacher=None, sched_p=0.0, use_neigh=True):
        B, N, C, T = X_in.shape
        H = self.enc(X_in)
        u = self.tpos(B, T, device=X_in.device).unsqueeze(1).repeat(1, N, 1, 1)
        Hc = torch.cat([H, u], dim=-1)
        A = self.graphs(phases)
        Z = self.proj(Hc, A)
        x_last = X_in[:, :, :, -1]
        Y_delta = self.head(Z, x_last=None, teacher=teacher,
                            sched_sampling_p=sched_p,
                            A_batch=A if use_neigh else None)
        Y_hat = x_last.unsqueeze(-1) + Y_delta
        return Y_hat
