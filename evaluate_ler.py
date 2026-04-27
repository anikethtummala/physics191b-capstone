"""
Evaluate logical error rate (LER) for a trained QCT checkpoint.

Supports:
  - synthetic mode: uses the same synthetic Pauli label generator as training
  - stim-shots mode: uses Stim detector samples + observable bit labels

Examples:
  uv run python evaluate_ler.py --checkpoint qct_checkpoint.pt --data-source synthetic --shots 20000
  uv run python evaluate_ler.py --checkpoint qct_checkpoint.pt --data-source stim-shots --noise-sweep 0.04 0.06 0.08 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import stim
import torch

from qct_surface_decoder import (
    LOGICAL_CLASSES,
    QCTModel,
    SNNTransformerBackbone,
    SyntheticPauliDataset,
    VanillaTransformerBackbone,
    layout_from_stim,
)


def load_checkpoint(path: str, device: str) -> dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def build_model_from_checkpoint(ckpt: dict, device: str) -> tuple[QCTModel, dict]:
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
        layout=layout,
        backbone=backbone,
        d_e=int(h["d_e"]),
        d_m=int(h["d_m"]),
        num_classes=int(h.get("num_classes", 4)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, h


@torch.no_grad()
def evaluate_synthetic(
    model: QCTModel,
    *,
    shots: int,
    batch_size: int,
    noise: float,
    seed: int,
    device: str,
) -> tuple[float, float]:
    ds = SyntheticPauliDataset(model.layout, p=noise, seed=seed)
    it = iter(ds)
    correct = 0
    seen = 0
    while seen < shots:
        b = min(batch_size, shots - seen)
        syn = []
        y = []
        for _ in range(b):
            s, t = next(it)
            syn.append(s.numpy())
            y.append(int(t.item()))
        syn_t = torch.from_numpy(np.stack(syn).astype(np.float32)).to(device)
        y_t = torch.tensor(y, device=device, dtype=torch.long)
        logits = model(syn_t)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y_t).sum().item())
        seen += b
    acc = correct / shots
    ler = 1.0 - acc
    return ler, acc


@torch.no_grad()
def evaluate_stim_shots(
    model: QCTModel,
    *,
    shots: int,
    batch_size: int,
    noise: float,
    distance: int,
    rounds: int,
    device: str,
) -> tuple[float, float]:
    if model.readout.head.out_features != 2:
        raise ValueError(
            "stim-shots evaluation expects a 2-class model (single observable bit). "
            "Use a checkpoint trained with --data-source stim-shots --num-classes 2."
        )

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=noise,
    )
    sampler = circuit.compile_detector_sampler()

    correct = 0
    seen = 0
    while seen < shots:
        b = min(batch_size, shots - seen)
        dets, obs = sampler.sample(b, separate_observables=True)
        syn_t = torch.from_numpy(dets.astype(np.float32)).to(device)
        y_t = torch.from_numpy(obs.reshape(-1).astype(np.int64)).to(device)
        logits = model(syn_t)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y_t).sum().item())
        seen += b

    acc = correct / shots
    ler = 1.0 - acc
    return ler, acc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate QCT checkpoint logical error rate.")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--data-source", choices=["synthetic", "stim-shots"], default="synthetic")
    p.add_argument("--shots", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--noise-sweep", type=float, nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ckpt = load_checkpoint(args.checkpoint, args.device)
    model, h = build_model_from_checkpoint(ckpt, args.device)
    distance = int(h["distance"])
    rounds = int(h["rounds"])

    noise_values = args.noise_sweep if args.noise_sweep else [args.noise]
    rows = []
    for noise in noise_values:
        if args.data_source == "synthetic":
            ler, acc = evaluate_synthetic(
                model,
                shots=args.shots,
                batch_size=args.batch_size,
                noise=float(noise),
                seed=args.seed,
                device=args.device,
            )
        else:
            ler, acc = evaluate_stim_shots(
                model,
                shots=args.shots,
                batch_size=args.batch_size,
                noise=float(noise),
                distance=distance,
                rounds=rounds,
                device=args.device,
            )
        rows.append((float(noise), ler, acc))
        print(
            f"noise={noise}  shots={args.shots}  LER={ler}  ACC={acc}  "
            f"source={args.data_source}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("noise,ler,acc,data_source,shots,checkpoint\n")
            for noise, ler, acc in rows:
                f.write(
                    f"{noise},{ler},{acc},{args.data_source},{args.shots},{args.checkpoint}\n"
                )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

