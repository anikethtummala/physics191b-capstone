"""
PyTorch port of the SNN transformer in main.c (Izhikevich / Spiking Manifesto style).

Matches the C layout and forward pass:
  - Token embedding (vocab x embedding_dim)
  - NUM_LAYERS x (NUM_HEADS causal LUT attention + LUT FFN), residual on z
  - LUT unembedder to vocab logits

Constants default to those in main.c. Training in the C code uses in-place SGD on
lookup tables with a custom derivative Up(); here we use torch.autograd.Function
so standard optimizers can update S and positional encodings.

See main.c: CONTEXT_SIZE, VOCAB_SIZE, EMBEDDING_DIM, POSITIONAL_DIM, NUM_LAYERS,
NUM_HEADS, N_T, N_C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sign_t(x: torch.Tensor) -> torch.Tensor:
    """C sign(): +1 for x>0, -1 otherwise (including 0)."""
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))


def up_u(u: torch.Tensor) -> torch.Tensor:
    """C Up(x): derivative of U for LUT backward through the min-|u| comparison."""
    s = sign_t(u)
    ax = u.abs().clamp(min=1e-12)
    return -0.5 * s / (ax * ax)


@dataclass(frozen=True)
class SNNTransformerConfig:
    context_size: int = 32
    vocab_size: int = 256
    embedding_dim: int = 32
    positional_dim: int = 4
    num_layers: int = 6
    num_heads: int = 4
    n_t: int = 16  # tables per LUT
    n_c: int = 6  # comparisons per table (embedding / FFN / unembed)

    @staticmethod
    def small_for_tests() -> "SNNTransformerConfig":
        """
        Smaller topology so forward/backward fits in memory during development.

        Full defaults match main.c; attention V LUT then has 2^(2*N_C+P_D) = 2^16 rows
        per table (large). This config uses N_C=3, P_D=2 -> 2^8 rows per V table.
        """
        return SNNTransformerConfig(
            context_size=8,
            vocab_size=64,
            embedding_dim=16,
            positional_dim=2,
            num_layers=2,
            num_heads=2,
            n_t=8,
            n_c=3,
        )


class _LUTSpikeFn(torch.autograd.Function):
    """
    One LUT bank: y = sum_i S[i, j_i, :] where j_i from comparisons on x[a]-x[b].

    Backward matches main.c LUT_backward / BACKWARD_UPDATE (gradient through the
    argmin comparison channel only).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        S: torch.Tensor,
        anchor_a: torch.Tensor,
        anchor_b: torch.Tensor,
    ) -> torch.Tensor:
        # x: (D,), S: (N_T, 2**nc, y_dim), anchors: (N_T, nc) long
        n_t, table_size, y_dim = S.shape
        nc = int(math.log2(table_size))
        device, dtype = x.device, x.dtype

        j = torch.zeros(n_t, dtype=torch.long, device=device)
        r_min = torch.zeros(n_t, dtype=torch.long, device=device)
        u_min = torch.full((n_t,), float("inf"), device=device, dtype=torch.float32)

        xa = x[anchor_a]
        xb = x[anchor_b]
        u = xa - xb
        for r in range(nc):
            j = j | ((u[:, r] > 0).long() << r)
            absu = u[:, r].abs()
            closer = absu < u_min.abs()
            u_min = torch.where(closer, u[:, r], u_min)
            r_min = torch.where(closer, torch.full_like(r_min, r), r_min)

        y = torch.zeros(y_dim, device=device, dtype=dtype)
        for i in range(n_t):
            y = y + S[i, j[i]]

        ctx.save_for_backward(x, S, anchor_a, anchor_b)
        ctx.j = j
        ctx.r_min = r_min
        ctx.u_min = u_min
        ctx.nc = nc
        ctx.n_t = n_t
        ctx.y_dim = y_dim
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, S, anchor_a, anchor_b = ctx.saved_tensors
        j, r_min, u_min = ctx.j, ctx.r_min, ctx.u_min
        nc, n_t, y_dim = ctx.nc, ctx.n_t, ctx.y_dim
        grad_x = torch.zeros_like(x)
        grad_S = torch.zeros_like(S)

        gy = grad_y.to(torch.float32)

        for i in range(n_t):
            ji = int(j[i].item())
            rmi = int(r_min[i].item())
            jbar = ji ^ (1 << rmi)

            row_j = S[i, ji]
            row_jbar = S[i, jbar]
            gi = (gy * (row_jbar - row_j)).sum()
            v = gi * up_u(u_min[i])
            ai = int(anchor_a[i, rmi].item())
            bi = int(anchor_b[i, rmi].item())
            grad_x[ai] = grad_x[ai] + v.to(grad_x.dtype)
            grad_x[bi] = grad_x[bi] - v.to(grad_x.dtype)

            grad_S[i, ji] = grad_S[i, ji] + grad_y

        return grad_x, grad_S, None, None


class _ConcatLUTSpikeFn2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xq: torch.Tensor,
        xk: torch.Tensor,
        pe_row: torch.Tensor,
        S: torch.Tensor,
        anchor_a_q: torch.Tensor,
        anchor_b_q: torch.Tensor,
        anchor_a_k: torch.Tensor,
        anchor_b_k: torch.Tensor,
    ) -> torch.Tensor:
        n_t, table_size, y_dim = S.shape
        n_c = anchor_a_q.shape[1]
        p_d = pe_row.shape[1]
        nc = n_c + n_c + p_d
        device, dtype = xq.device, xq.dtype

        def cache_bits(x: torch.Tensor, aa: torch.Tensor, bb: torch.Tensor, nbits: int):
            j = torch.zeros(n_t, dtype=torch.long, device=device)
            r_min = torch.zeros(n_t, dtype=torch.long, device=device)
            u_min = torch.full((n_t,), float("inf"), device=device, dtype=torch.float32)
            xa = x[aa]
            xb = x[bb]
            u = xa - xb
            for r in range(nbits):
                j = j | ((u[:, r] > 0).long() << r)
                absu = u[:, r].abs()
                closer = absu < u_min.abs()
                u_min = torch.where(closer, u[:, r], u_min)
                r_min = torch.where(closer, torch.full_like(r_min, r), r_min)
            return j, r_min, u_min, u

        j_q, r_q, u_q, _ = cache_bits(xq, anchor_a_q, anchor_b_q, n_c)
        j_k, r_k, u_k, _ = cache_bits(xk, anchor_a_k, anchor_b_k, n_c)
        j_pe = torch.zeros(n_t, dtype=torch.long, device=device)
        r_pe = torch.zeros(n_t, dtype=torch.long, device=device)
        u_pe = torch.full((n_t,), float("inf"), device=device, dtype=torch.float32)
        for r in range(p_d):
            j_pe = j_pe | ((pe_row[:, r] > 0).long() << r)
            absu = pe_row[:, r].abs()
            closer = absu < u_pe.abs()
            u_pe = torch.where(closer, pe_row[:, r], u_pe)
            r_pe = torch.where(closer, torch.full_like(r_pe, r), r_pe)

        shift_k = p_d
        shift_q = n_c + p_d
        j_cat = (j_q << shift_q) | (j_k << shift_k) | j_pe

        y = torch.zeros(y_dim, device=device, dtype=dtype)
        for i in range(n_t):
            y = y + S[i, j_cat[i]]

        ctx.save_for_backward(xq, xk, pe_row, S, anchor_a_q, anchor_b_q, anchor_a_k, anchor_b_k)
        ctx.j_cat = j_cat
        ctx.j_q, ctx.j_k, ctx.j_pe = j_q, j_k, j_pe
        ctx.r_q, ctx.r_k, ctx.r_pe = r_q, r_k, r_pe
        ctx.u_q, ctx.u_k, ctx.u_pe = u_q, u_k, u_pe
        ctx.n_c, ctx.p_d, ctx.n_t, ctx.y_dim = n_c, p_d, n_t, y_dim
        ctx.shift_q, ctx.shift_k = shift_q, shift_k
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        xq, xk, pe_row, S, aa_q, ab_q, aa_k, ab_k = ctx.saved_tensors
        j_cat = ctx.j_cat
        j_q, j_k, j_pe = ctx.j_q, ctx.j_k, ctx.j_pe
        r_q, r_k, r_pe = ctx.r_q, ctx.r_k, ctx.r_pe
        u_q, u_k, u_pe = ctx.u_q, ctx.u_k, ctx.u_pe
        n_c, p_d, n_t, y_dim = ctx.n_c, ctx.p_d, ctx.n_t, ctx.y_dim
        shift_q, shift_k = ctx.shift_q, ctx.shift_k

        grad_xq = torch.zeros_like(xq)
        grad_xk = torch.zeros_like(xk)
        grad_pe = torch.zeros_like(pe_row)
        grad_S = torch.zeros_like(S)
        gy = grad_y.to(torch.float32)

        for i in range(n_t):
            j = int(j_cat[i].item())
            row_j = S[i, j]
            grad_S[i, j] = grad_S[i, j] + grad_y

            abs_q, abs_k, abs_pe = u_q[i].abs(), u_k[i].abs(), u_pe[i].abs()
            jbar_q = int(j_q[i].item()) ^ (1 << int(r_q[i].item()))
            jbar_k = int(j_k[i].item()) ^ (1 << int(r_k[i].item()))
            jbar_pe = int(j_pe[i].item()) ^ (1 << int(r_pe[i].item()))
            jbar_q_full = (jbar_q << shift_q) | (int(j_k[i].item()) << shift_k) | int(j_pe[i].item())
            jbar_k_full = (int(j_q[i].item()) << shift_q) | (jbar_k << shift_k) | int(j_pe[i].item())
            jbar_pe_full = (int(j_q[i].item()) << shift_q) | (int(j_k[i].item()) << shift_k) | jbar_pe

            if abs_q < abs_k:
                row_jb = S[i, jbar_q_full]
                gi_q = (gy * (row_jb - row_j)).sum()
                v = gi_q * up_u(u_q[i])
                rq = int(r_q[i].item())
                grad_xq[int(aa_q[i, rq].item())] += v.to(grad_xq.dtype)
                grad_xq[int(ab_q[i, rq].item())] -= v.to(grad_xq.dtype)
            else:
                row_jb = S[i, jbar_k_full]
                gi_k = (gy * (row_jb - row_j)).sum()
                v = gi_k * up_u(u_k[i])
                rk = int(r_k[i].item())
                grad_xk[int(aa_k[i, rk].item())] += v.to(grad_xk.dtype)
                grad_xk[int(ab_k[i, rk].item())] -= v.to(grad_xk.dtype)

            if abs_pe < abs_q and abs_pe < abs_k:
                row_jb = S[i, jbar_pe_full]
                gi_pe = (gy * (row_jb - row_j)).sum()
                delta = gi_pe * up_u(u_pe[i])
                rp = int(r_pe[i].item())
                grad_pe[i, rp] = grad_pe[i, rp] + delta.to(grad_pe.dtype)

        return grad_xq, grad_xk, grad_pe, grad_S, None, None, None, None


def _init_anchors(n_t: int, n_c: int, dim: int, device: torch.device, generator: torch.Generator | None):
    """Match C: random a, b in [0, dim), b != a per row."""
    aa = torch.randint(0, dim, (n_t, n_c), device=device)
    bb = torch.randint(0, dim, (n_t, n_c), device=device)
    for i in range(n_t):
        for r in range(n_c):
            while bb[i, r] == aa[i, r]:
                bb[i, r] = torch.randint(0, dim, (1,), device=device, generator=generator).item()
    return aa, bb


class SpikingLUT(nn.Module):
    """N_T lookup tables, each 2**nc rows of y_dim (FFN / unembed style)."""

    def __init__(self, n_t: int, n_c: int, y_dim: int, embedding_dim: int, generator: torch.Generator | None = None):
        super().__init__()
        self.n_t = n_t
        self.n_c = n_c
        self.y_dim = y_dim
        self.embedding_dim = embedding_dim
        t = 1 << n_c
        self.S = nn.Parameter(torch.zeros(n_t, t, y_dim))
        dev = self.S.device
        aa, bb = _init_anchors(n_t, n_c, embedding_dim, dev, generator)
        self.register_buffer("anchor_a", aa)
        self.register_buffer("anchor_b", bb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (D,) -> (y_dim,)"""
        return _LUTSpikeFn.apply(x, self.S, self.anchor_a, self.anchor_b)


class SpikingConcatLUT(nn.Module):
    """Attention V LUT: concat Q, K, PE bit indices, total nc = 2*N_C + P_D."""

    def __init__(
        self,
        n_t: int,
        n_c: int,
        p_d: int,
        y_dim: int,
        embedding_dim: int,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.n_t = n_t
        self.n_c = n_c
        self.p_d = p_d
        self.y_dim = y_dim
        nc = 2 * n_c + p_d
        t = 1 << nc
        self.S = nn.Parameter(torch.zeros(n_t, t, y_dim))
        dev = self.S.device
        aa_q, bb_q = _init_anchors(n_t, n_c, embedding_dim, dev, generator)
        aa_k, bb_k = _init_anchors(n_t, n_c, embedding_dim, dev, generator)
        self.register_buffer("anchor_a_q", aa_q)
        self.register_buffer("anchor_b_q", bb_q)
        self.register_buffer("anchor_a_k", aa_k)
        self.register_buffer("anchor_b_k", bb_k)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """xq, xk: (D,), pe: (N_T, P_D) — one row of relative PE for this pair."""
        return _ConcatLUTSpikeFn2.apply(
            xq, xk, pe, self.S,
            self.anchor_a_q, self.anchor_b_q, self.anchor_a_k, self.anchor_b_k,
        )


class SNNAttentionHead(nn.Module):
    """C AttentionHead: causal updates to position pos from pos1 < pos."""

    def __init__(self, cfg: SNNTransformerConfig, generator: torch.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        self.pe = nn.Parameter(torch.zeros(cfg.context_size, cfg.n_t, cfg.positional_dim))
        nn.init.uniform_(self.pe, -1.0, 1.0)
        self.v_lut = SpikingConcatLUT(
            cfg.n_t, cfg.n_c, cfg.positional_dim, cfg.embedding_dim, cfg.embedding_dim, generator
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (L, D) layer input (same role as frozen x in C's attention_forward).

        Returns a residual delta of shape (L, D) to add to z (C adds in-place into z).
        """
        cfg = self.cfg
        L, D = x.shape
        delta = torch.zeros(L, D, device=x.device, dtype=x.dtype)
        for pos in range(1, L):
            for pos1 in range(pos):
                rel = pos - pos1
                pe_slice = self.pe[rel]
                delta[pos] = delta[pos] + self.v_lut(x[pos], x[pos1], pe_slice)
        return delta


class SNNTransformerBlock(nn.Module):
    def __init__(self, cfg: SNNTransformerConfig, generator: torch.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        self.heads = nn.ModuleList([SNNAttentionHead(cfg, generator) for _ in range(cfg.num_heads)])
        self.ffn = SpikingLUT(cfg.n_t, cfg.n_c, cfg.embedding_dim, cfg.embedding_dim, generator)

    def forward(self, x_layer: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Residual attention then residual FFN (same order as C); no in-place on z for autograd.
        for h in self.heads:
            z = z + h(x_layer)
        ffn_delta = torch.stack([self.ffn(z[pos]) for pos in range(z.shape[0])], dim=0)
        return z + ffn_delta


class SNNTransformer(nn.Module):
    """
    Character-level causal transformer from main.c.

    Forward: token_ids (B, L) with L == context_size -> logits (B, L, vocab).

    Note: default hyperparameters match main.c; the attention V LUT has shape
    (N_T, 2^(2*N_C+POSITIONAL_DIM), EMBEDDING_DIM) which is memory-heavy. Use
    ``SNNTransformerConfig.small_for_tests()`` while iterating.
    """

    def __init__(self, cfg: SNNTransformerConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = cfg or SNNTransformerConfig()
        c = self.cfg
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        self.token_embed = nn.Parameter(torch.empty(c.vocab_size, c.embedding_dim))
        nn.init.uniform_(self.token_embed, -1.0, 1.0)

        self.blocks = nn.ModuleList([SNNTransformerBlock(c, gen) for _ in range(c.num_layers)])
        self.unembed = SpikingLUT(c.n_t, c.n_c, c.vocab_size, c.embedding_dim, gen)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, L) int64, values in [0, vocab_size).
        Returns logits (B, L, vocab_size).
        """
        c = self.cfg
        emb = F.embedding(token_ids, self.token_embed)
        B, L, D = emb.shape
        if L != c.context_size:
            raise ValueError(f"expected sequence length {c.context_size}, got {L}")

        out = []
        for b in range(B):
            z = emb[b].clone()
            x0 = emb[b].clone()
            for blk in self.blocks:
                z = blk(x0, z)
            logits = torch.stack([self.unembed(z[pos]) for pos in range(L)], dim=0)
            out.append(logits)
        return torch.stack(out, dim=0)


__all__ = [
    "SNNTransformer",
    "SNNTransformerConfig",
    "SpikingLUT",
    "SpikingConcatLUT",
    "SNNAttentionHead",
    "SNNTransformerBlock",
]


if __name__ == "__main__":
    cfg = SNNTransformerConfig.small_for_tests()
    m = SNNTransformer(cfg, seed=0)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_size))
    y = m(x)
    assert y.shape == (2, cfg.context_size, cfg.vocab_size)
    loss = y.sum()
    loss.backward()
    print("ok", y.shape, "grad token_embed", m.token_embed.grad is not None)
