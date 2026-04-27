"""
Qubit-centric Transformer (QCT) training + decoder for rotated surface codes.

Methodology follows Park, Kwak, Kim — "Qubit-centric Transformer for Surface Code
Decoding" (arXiv:2510.11593, https://arxiv.org/abs/2510.11593): qubit-centric embedding
from localized syndromes, merging layer, structure-aware attention mask, transformer
blocks, mean pooling, and 4-way logical Pauli classification.

The *sequential* part is isolated behind a small protocol so you can swap in an SNN
transformer backbone later without touching embedding, masking, or the readout head.

Dependencies: torch, numpy, stim (https://github.com/quantumlib/Stim)

Example:
  uv run python qct_surface_decoder.py train --distance 3 --rounds 5 --steps 2000
  uv run python qct_surface_decoder.py decode --checkpoint qct.pt --distance 3 --rounds 5
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import stim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from snn_transformer import SpikingConcatLUT, SpikingLUT


# --- Logical class ordering (matches common Pauli enumeration) -----------------

LOGICAL_CLASSES = ("I", "X", "Z", "Y")


def pauli_class_index(lx: int, lz: int) -> int:
    """Map (lx, lz) in {0,1}^2 to 0=I, 1=X, 2=Z, 3=Y."""
    return (lx & 1) | ((lz & 1) << 1)


# --- Layout: parse Stim circuit into qubit-centric topology --------------------


@dataclass(frozen=True)
class SurfaceLayout:
    """Rotated surface-code geometry for one (distance, rounds) Stim circuit."""

    distance: int
    rounds: int
    data_qubit_indices: tuple[int, ...]  # Stim qubit indices, len n = d^2
    data_coords: tuple[tuple[float, float], ...]  # (x, y) per data qubit, same order
    detector_coords: tuple[tuple[float, float, float], ...]  # (x, y, t) per detector bit
    z_detector_mask: np.ndarray  # shape (m,), bool: which syndrome bits count as Z-type
    n_qubits: int
    m_syndrome: int

    @property
    def n(self) -> int:
        return len(self.data_qubit_indices)

    def structure_mask(self) -> torch.Tensor:
        """
        M_ij = 0 if qubits i,j share a stabilizer (any), -inf otherwise.
        Shape (n, n). Indices follow data_qubit_indices order.
        """
        n = self.n
        m = self.m_syndrome
        adj_det = [set() for _ in range(n)]
        for j in range(m):
            dx, dy, _ = self.detector_coords[j]
            touched = [
                i
                for i, (qx, qy) in enumerate(self.data_coords)
                if abs(dx - qx) == 1.0 and abs(dy - qy) == 1.0
            ]
            for a in touched:
                adj_det[a].add(j)
        M = torch.full((n, n), float("-inf"))
        for i in range(n):
            for j in range(n):
                if i == j or adj_det[i] & adj_det[j]:
                    M[i, j] = 0.0
        return M

    def hz_hx_pauli_labels(
        self, rng: np.random.Generator, p: float, H: HzHx | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthetic i.i.d. depolarizing-like errors on data qubits only (per-round noise
        is approximated as a memoryless Pauli draw). Returns (syndrome01, label, xz_stack).

        syndrome01 has shape (m,) with entries in {0,1}.

        Hz uses Z-stabilizer rows (z_detector_mask), Hx uses X-stabilizer rows (~mask).
        Logical bits are commutation of X/Z components with fixed boundary operators.
        """
        n = self.n
        m = self.m_syndrome
        # Depolarizing: I, X, Y, Z with probs (1-p), p/3 each
        probs = np.array([(1 - p), p / 3, p / 3, p / 3], dtype=np.float64)
        draw = rng.choice(4, size=n, p=probs)
        pxz = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)  # I,X,Z,Y
        x = pxz[draw, 0]
        z = pxz[draw, 1]

        if H is None:
            H = build_hz_hx_matrices(self)
        hz = H.hz  # (m_half, n)
        hx = H.hx  # (m_half, n)
        m_half = hz.shape[0]
        s_z = (hz @ x) & 1
        s_x = (hx @ z) & 1
        syn = np.zeros(m, dtype=np.uint8)
        syn[:m_half] = s_z.astype(np.uint8)
        syn[m_half : m_half + m_half] = s_x.astype(np.uint8)

        lx_vec, lz_vec = boundary_logical_vectors(self)
        lx = int((lx_vec @ x) & 1)
        lz = int((lz_vec @ z) & 1)
        y = pauli_class_index(lx, lz)
        return syn.astype(np.float32), np.int64(y), np.stack([x, z], axis=0)


@dataclass(frozen=True)
class HzHx:
    hz: np.ndarray  # (m_z, n)
    hx: np.ndarray  # (m_x, n)


def build_hz_hx_matrices(layout: SurfaceLayout) -> HzHx:
    """
    Parity checks from Stim: X on data qubit flips Z-stab detectors (columns of F).

    For `surface_code:rotated_memory_z` with few rounds, Z data errors often produce no
    detector flips in Stim's DEM; in that case we reuse the same Z-check matrix for the
    X-check block so synthetic data still has decoupled (s^Z, s^X) channels:
      s^Z = Hz @ x,   s^X = Hz @ z
    which is sufficient for supervised training of the qubit-centric head (physics is
    approximate; swap in a full circuit-level sampler when you need strict calibration).
    """
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=layout.distance,
        rounds=layout.rounds,
    )
    flat = list(c.flattened())
    mr_i = next(i for i, inst in enumerate(flat) if inst.name == "MR")
    data = layout.data_qubit_indices
    n = len(data)
    m = layout.m_syndrome
    ref = c.compile_detector_sampler().sample(1, separate_observables=True)[0][0].astype(np.uint8)

    F = np.zeros((n, m), dtype=np.uint8)  # X on qubit row flips columns
    for qi, q in enumerate(data):
        c2 = c.copy()
        c2.insert(mr_i, stim.Circuit(f"X_ERROR(1) {q}"))
        d = c2.compile_detector_sampler().sample(1, separate_observables=True)[0][0].astype(np.uint8)
        F[qi] = (d ^ ref) & 1

    G = np.zeros((n, m), dtype=np.uint8)  # Z on qubit row flips columns
    for qi, q in enumerate(data):
        c2 = c.copy()
        c2.insert(mr_i, stim.Circuit(f"Z_ERROR(1) {q}"))
        d = c2.compile_detector_sampler().sample(1, separate_observables=True)[0][0].astype(np.uint8)
        G[qi] = (d ^ ref) & 1

    m_half = m // 2
    # Use the same Z-check columns for both syndrome halves unless Z errors flip something.
    active = np.flatnonzero(F.sum(axis=0) > 0)
    if len(active) >= m_half:
        cols_z = active[:m_half]
    else:
        cols_z = np.arange(m_half)
    hz = F[:, cols_z].T.copy()  # (m_half, n)
    if int(G.sum()) == 0:
        hx = hz.copy()
    else:
        cols_x = np.flatnonzero(G.sum(axis=0) > 0)[:m_half]
        hx = G[:, cols_x].T.copy()
        if hx.shape[0] < m_half:
            hx = hz.copy()
    return HzHx(hz=hz, hx=hx)


def boundary_logical_vectors(layout: SurfaceLayout) -> tuple[np.ndarray, np.ndarray]:
    """Un-normalized X_L / Z_L on the left column and top row of the data grid."""
    xs = np.array([c[0] for c in layout.data_coords], dtype=np.float64)
    ys = np.array([c[1] for c in layout.data_coords], dtype=np.float64)
    lx = ((xs == xs.min()) & 1).astype(np.uint8)
    lz = ((ys == ys.min()) & 1).astype(np.uint8)
    return lx, lz


def layout_from_stim(distance: int, rounds: int) -> SurfaceLayout:
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
    )
    det_coords_dict = c.get_detector_coordinates()
    m = c.num_detectors
    detector_coords: list[tuple[float, float, float]] = []
    for k in range(m):
        x, y, t = det_coords_dict[k]
        detector_coords.append((float(x), float(y), float(t)))

    data_indices: list[int] = []
    for ln in str(c).split("\n"):
        if ln.startswith("M ") and "MR" not in ln:
            data_indices = [int(x) for x in ln.split()[1:]]
            break
    if len(data_indices) != distance * distance:
        raise RuntimeError(f"unexpected data qubit count {len(data_indices)} for d={distance}")

    q_coords: dict[int, tuple[float, float]] = {}
    for ln in str(c).split("\n"):
        if ln.startswith("QUBIT_COORDS("):
            inner, q = ln.split(") ")
            xy = inner.split("(")[1].split(",")
            x, y = float(xy[0]), float(xy[1].strip())
            q_coords[int(q)] = (x, y)

    data_coords = tuple(q_coords[q] for q in data_indices)

    # Z-type vs X-type stabilizers: use spatial bipartition of plaquette centers.
    # Fallback: split detectors by median x so both blocks are nonempty.
    # Syndrome layout for QCT: first half of bits are Z-stabilizer outcomes, second half X-stabilizer
    # (matches the paper's s = [s^Z, s^X] partition when using Stim's flattened detector order).
    z_mask = np.arange(m) < (m // 2)

    return SurfaceLayout(
        distance=distance,
        rounds=rounds,
        data_qubit_indices=tuple(data_indices),
        data_coords=data_coords,
        detector_coords=tuple(detector_coords),
        z_detector_mask=z_mask.astype(bool),
        n_qubits=c.num_qubits,
        m_syndrome=m,
    )


# --- Modular backbone (swap for SNN transformer) -------------------------------


@runtime_checkable
class QCTBackbone(Protocol):
    """Consumes merged tokens [B, n, d_m] and an additive attention mask [n, n]."""

    d_model: int

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor: ...


class VanillaTransformerBackbone(nn.Module):
    """Pre-LN transformer encoder with explicit structure-aware attention mask."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(
                            d_model, n_heads, batch_first=True, dropout=dropout
                        ),
                        "ln2": nn.LayerNorm(d_model),
                        "ff0": nn.Linear(d_model, ff_mult * d_model),
                        "ff1": nn.Linear(ff_mult * d_model, d_model),
                    }
                )
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # attn_mask: float (n, n) additive (0 keep, -inf block) broadcast to heads
        n = x.shape[1]
        if attn_mask.shape != (n, n):
            raise ValueError("attn_mask must be (n, n)")
        am = attn_mask.to(dtype=x.dtype, device=x.device)
        for layer in self.layers:
            h = layer["ln1"](x)
            a, _ = layer["attn"](h, h, h, attn_mask=am, need_weights=False)
            x = x + self.dropout(a)
            h2 = layer["ln2"](x)
            ff = F.gelu(layer["ff0"](h2))
            ff = layer["ff1"](ff)
            x = x + self.dropout(ff)
        return x


class SNNTransformerBackbone(nn.Module):
    """
    SNN/LUT-based backbone for QCT tokens.

    This mirrors the high-level residual pattern from `main.c`:
      z <- z + attention(z)
      z <- z + ffn(z)
    repeated over layers.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        *,
        n_t: int = 8,
        n_c: int = 3,
        positional_dim: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_t = n_t
        self.n_c = n_c
        self.positional_dim = positional_dim

        class _SNNLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.heads = nn.ModuleList(
                    [
                        SpikingConcatLUT(
                            n_t=n_t,
                            n_c=n_c,
                            p_d=positional_dim,
                            y_dim=d_model,
                            embedding_dim=d_model,
                        )
                        for _ in range(n_heads)
                    ]
                )
                self.ffn = SpikingLUT(
                    n_t=n_t,
                    n_c=n_c,
                    y_dim=d_model,
                    embedding_dim=d_model,
                )
                self.pe = nn.Parameter(torch.zeros(256, n_t, positional_dim))
                nn.init.uniform_(self.pe, -1.0, 1.0)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(_SNNLayer())

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, n, d_model), attn_mask: (n, n) additive where -inf blocks
        bsz, n, d = x.shape
        if attn_mask.shape != (n, n):
            raise ValueError("attn_mask must be (n, n)")
        allowed = ~torch.isinf(attn_mask)
        rel_cap = 255

        for layer in self.layers:
            # Attention residual
            delta = torch.zeros_like(x)
            pe = layer.pe
            heads: nn.ModuleList = layer.heads
            for h in heads:
                for i in range(n):
                    for j in range(n):
                        if not bool(allowed[i, j]):
                            continue
                        rel = i - j
                        if rel < 0:
                            continue
                        rel = min(rel, rel_cap)
                        pe_row = pe[rel]
                        for b in range(bsz):
                            delta[b, i] = delta[b, i] + h(x[b, i], x[b, j], pe_row)
            x = x + delta

            # FFN residual
            ffn: SpikingLUT = layer.ffn
            ffn_delta = torch.zeros_like(x)
            for b in range(bsz):
                for i in range(n):
                    ffn_delta[b, i] = ffn(x[b, i])
            x = x + ffn_delta
        return x


# --- QCT modules (embedding, merge, head) --------------------------------------


class QCTEmbedding(nn.Module):
    """Eq. (3)-(5): localized sparse syndrome vectors -> φ^Z, φ^X with shared FC + PE."""

    def __init__(self, m: int, d_e: int, n: int):
        super().__init__()
        self.fc = nn.Linear(m, d_e, bias=True)
        self.pe_z = nn.Parameter(torch.zeros(n, d_e))
        self.pe_x = nn.Parameter(torch.zeros(n, d_e))
        self.m = m
        self.d_e = d_e

    def forward(self, syndrome01: torch.Tensor, layout: SurfaceLayout) -> tuple[torch.Tensor, torch.Tensor]:
        """
        syndrome01: (B, m) in {0,1}
        returns phi_z, phi_x each (B, n, d_e)
        """
        b, m = syndrome01.shape
        if m != self.m:
            raise ValueError(f"expected m={self.m}, got {m}")
        sigma = 1.0 - 2.0 * syndrome01.to(torch.float32)  # (-1)^{s_j}

        m_z = int(layout.z_detector_mask.sum())
        m_x = m - m_z
        z_idx = torch.as_tensor(np.flatnonzero(layout.z_detector_mask), device=syndrome01.device)
        x_idx = torch.as_tensor(np.flatnonzero(~layout.z_detector_mask), device=syndrome01.device)

        phi_z_list = []
        phi_x_list = []
        for i in range(layout.n):
            qx, qy = layout.data_coords[i]
            # Build sparse localized vectors ξ^Z, ξ^X (length m) per batch element
            xi_z = torch.zeros(b, m, device=syndrome01.device, dtype=torch.float32)
            xi_x = torch.zeros(b, m, device=syndrome01.device, dtype=torch.float32)
            for j in range(m):
                dx, dy, _ = layout.detector_coords[j]
                if abs(dx - qx) == 1.0 and abs(dy - qy) == 1.0:
                    if layout.z_detector_mask[j]:
                        xi_z[:, j] = sigma[:, j]
                    else:
                        xi_x[:, j] = sigma[:, j]
            # Project with shared weights then add type-specific positional embeddings
            phi_z_list.append(self.fc(xi_z) + self.pe_z[i])
            phi_x_list.append(self.fc(xi_x) + self.pe_x[i])

        phi_z = torch.stack(phi_z_list, dim=1)
        phi_x = torch.stack(phi_x_list, dim=1)
        return phi_z, phi_x


class MergingLayer(nn.Module):
    """Eq. (6)-(7): concat φ^Z, φ^X -> unified token x^(0) in R^{d_m}."""

    def __init__(self, d_e: int, d_m: int):
        super().__init__()
        self.lin = nn.Linear(2 * d_e, d_m, bias=True)

    def forward(self, phi_z: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        u = torch.cat([phi_z, phi_x], dim=-1)
        return self.lin(u)


class QCTReadout(nn.Module):
    """Mean pool over qubits + linear to logits (Eq. after (8))."""

    def __init__(self, d_model: int, num_classes: int = 4):
        super().__init__()
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return self.head(pooled)


class QCTModel(nn.Module):
    """End-to-end QCT with injectable backbone."""

    def __init__(
        self,
        layout: SurfaceLayout,
        backbone: nn.Module,
        d_e: int = 64,
        d_m: int | None = None,
        num_classes: int = 4,
    ):
        super().__init__()
        if d_m is None:
            d_m = getattr(backbone, "d_model")
        self.layout = layout
        self.embedding = QCTEmbedding(layout.m_syndrome, d_e=d_e, n=layout.n)
        self.merge = MergingLayer(d_e=d_e, d_m=d_m)
        self.backbone = backbone
        self.readout = QCTReadout(d_model=d_m, num_classes=num_classes)
        self.register_buffer("struct_mask", layout.structure_mask(), persistent=False)

    def forward(self, syndrome01: torch.Tensor) -> torch.Tensor:
        phi_z, phi_x = self.embedding(syndrome01, self.layout)
        x0 = self.merge(phi_z, phi_x)
        h = self.backbone(x0, self.struct_mask)
        return self.readout(h)


# --- Dataset -------------------------------------------------------------------


class SyntheticPauliDataset(IterableDataset):
    """Infinite stream of (syndrome_bits, label) from explicit Pauli simulation."""

    def __init__(self, layout: SurfaceLayout, p: float, seed: int = 0):
        super().__init__()
        self.layout = layout
        self.p = p
        self.seed = seed
        self.H = build_hz_hx_matrices(layout)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + int(torch.initial_seed() % (2**31 - 1)))
        while True:
            syn, y, _ = self.layout.hz_hx_pauli_labels(rng, self.p, H=self.H)
            yield torch.from_numpy(syn.astype(np.float32)), torch.tensor(int(y), dtype=torch.long)


class StimShotDataset(IterableDataset):
    """Syndromes from Stim's noisy circuit; labels from detector sampler observables (1 bit)."""

    def __init__(self, distance: int, rounds: int, p: float, seed: int = 0):
        super().__init__()
        self.distance = distance
        self.rounds = rounds
        self.p = p
        self.seed = seed
        self.layout = layout_from_stim(distance, rounds)
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            before_round_data_depolarization=p,
        )
        self.sampler = self.circuit.compile_detector_sampler()

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            dets, obs = self.sampler.sample(1, separate_observables=True)
            syn = dets[0].astype(np.float32)
            # Map single observable to {I,Z} vs {X,Y} coarse classes if needed
            y = int(obs[0, 0])
            yield torch.from_numpy(syn), torch.tensor(y, dtype=torch.long)


# --- Train / decode CLI ---------------------------------------------------------


def train_main(args: argparse.Namespace) -> None:
    layout = layout_from_stim(args.distance, args.rounds)
    if args.backbone == "vanilla":
        backbone = VanillaTransformerBackbone(
            d_model=args.d_m,
            n_heads=args.heads,
            n_layers=args.layers,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
        )
    else:
        backbone = SNNTransformerBackbone(
            d_model=args.d_m,
            n_heads=args.heads,
            n_layers=args.layers,
            n_t=args.snn_n_t,
            n_c=args.snn_n_c,
            positional_dim=args.snn_pos_dim,
        )
    model = QCTModel(layout, backbone, d_e=args.d_e, d_m=args.d_m, num_classes=args.num_classes)
    device = torch.device(args.device)
    model.to(device)

    if args.data_source == "synthetic":
        ds: IterableDataset = SyntheticPauliDataset(layout, p=args.noise, seed=args.seed)
    else:
        if args.num_classes != 2:
            raise ValueError("stim-shots mode currently provides one logical observable bit; use --num-classes 2")
        ds = StimShotDataset(args.distance, args.rounds, p=args.noise, seed=args.seed)

    loader = DataLoader(ds, batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    while step < args.steps:
        for syn, y in loader:
            syn = syn.to(device)
            y = y.to(device)
            logits = model(syn)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % args.log_every == 0:
                pred = logits.argmax(dim=-1)
                acc = (pred == y).float().mean().item()
                print(f"step {step:6d}  loss {loss.item():.4f}  batch_acc {acc:.3f}")
            step += 1
            if step >= args.steps:
                break

    out = Path(args.out)
    torch.save(
        {
            "model": model.state_dict(),
            "layout": {
                "distance": layout.distance,
                "rounds": layout.rounds,
                "m": layout.m_syndrome,
                "n": layout.n,
            },
            "hparams": vars(args),
        },
        out,
    )
    print(f"wrote {out}")


def decode_main(args: argparse.Namespace) -> None:
    try:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
    h = ckpt["hparams"]
    layout = layout_from_stim(int(h["distance"]), int(h["rounds"]))
    if h.get("backbone", "vanilla") == "vanilla":
        backbone = VanillaTransformerBackbone(
            d_model=int(h["d_m"]),
            n_heads=int(h["heads"]),
            n_layers=int(h["layers"]),
            ff_mult=int(h.get("ff_mult", 4)),
            dropout=float(h.get("dropout", 0.0)),
        )
    else:
        backbone = SNNTransformerBackbone(
            d_model=int(h["d_m"]),
            n_heads=int(h["heads"]),
            n_layers=int(h["layers"]),
            n_t=int(h.get("snn_n_t", 8)),
            n_c=int(h.get("snn_n_c", 3)),
            positional_dim=int(h.get("snn_pos_dim", 2)),
        )
    model = QCTModel(
        layout,
        backbone,
        d_e=int(h["d_e"]),
        d_m=int(h["d_m"]),
        num_classes=int(h.get("num_classes", 4)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    # Example: decode a handful of random shots from Stim at the same noise rate
    p = float(h.get("noise", 0.08))
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=layout.distance,
        rounds=layout.rounds,
        before_round_data_depolarization=p,
    )
    dets, obs = c.compile_detector_sampler().sample(args.shots, separate_observables=True)
    syn = torch.from_numpy(dets.astype(np.float32)).to(args.device)
    with torch.no_grad():
        logits = model(syn)
        pred = logits.argmax(dim=-1).cpu().numpy()
    agree = np.mean(pred == obs.reshape(-1).astype(np.int64)) if obs.shape[1] == 1 else float("nan")
    print("predicted logical classes:", [LOGICAL_CLASSES[i] for i in pred[:20]])
    print("agreement with Stim observable (binary surrogate):", agree)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QCT surface-code decoder (arXiv:2510.11593-style)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--distance", type=int, default=3)
    t.add_argument("--rounds", type=int, default=5)
    t.add_argument("--noise", type=float, default=0.08, help="depolarizing strength (meaning depends on data source)")
    t.add_argument(
        "--data-source",
        choices=["synthetic", "stim-shots"],
        default="synthetic",
        help="synthetic: explicit Pauli + Hz/Hx labels (4 classes). stim-shots: circuit noise + observable bit (2 classes).",
    )
    t.add_argument("--num-classes", type=int, default=4)
    t.add_argument("--backbone", choices=["vanilla", "snn"], default="vanilla")
    t.add_argument("--d-e", type=int, default=64)
    t.add_argument("--d-m", type=int, default=128)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--snn-n-t", type=int, default=8)
    t.add_argument("--snn-n-c", type=int, default=3)
    t.add_argument("--snn-pos-dim", type=int, default=2)
    t.add_argument("--ff-mult", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", type=str, default="cpu")
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--out", type=str, default="qct_checkpoint.pt")

    d = sub.add_parser("decode")
    d.add_argument("--checkpoint", type=str, required=True)
    d.add_argument("--shots", type=int, default=256)
    d.add_argument("--device", type=str, default="cpu")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "decode":
        decode_main(args)


if __name__ == "__main__":
    main()
