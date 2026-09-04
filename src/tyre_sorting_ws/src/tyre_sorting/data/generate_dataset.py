"""
T2 — Multi-Modal Synthetic Fragment Dataset generator.

Produces:
  dataset/meshes/<id>.obj              -- non-uniform rigid-body fragment mesh
  dataset/spectra.npz                  -- {ids, spectra[N,bands,channels]}
  dataset/metadata.csv                 -- one row per fragment (labels + geometry)

Run:
    python -m tyre_sorting.data.generate_dataset --n 320 --out dataset --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os
import numpy as np

from tyre_sorting.data.mesh_gen import generate_fragment
from tyre_sorting.data.spectroscopy import (
    TYRE_COMPOSITIONS, SpectroscopyConfig, synthesize_signature,
)


def build_dataset(n: int, out_dir: str, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    tyre_types = list(TYRE_COMPOSITIONS.keys())
    surface_classes = ["tread", "sidewall"]
    spec_cfg = SpectroscopyConfig(seed=seed)

    specs = []
    spectra = np.zeros((n, spec_cfg.n_bands, spec_cfg.n_channels), dtype=np.float32)

    for i in range(n):
        tyre_type = tyre_types[rng.integers(0, len(tyre_types))]
        surface_class = surface_classes[rng.integers(0, len(surface_classes))]
        frag_id = f"frag_{i:04d}"

        spec = generate_fragment(frag_id, surface_class, tyre_type, rng, mesh_dir)
        specs.append(spec)

        composition = TYRE_COMPOSITIONS[tyre_type]
        spectra[i] = synthesize_signature(composition, spec_cfg, rng, surface_class=surface_class)

    # Write metadata table
    meta_path = os.path.join(out_dir, "metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fragment_id", "surface_class", "tyre_type",
                          "extent_x", "extent_y", "extent_z", "roughness", "mesh_path"])
        for s in specs:
            writer.writerow([s.fragment_id, s.surface_class, s.tyre_type,
                              *[f"{v:.5f}" for v in s.bbox_extent],
                              f"{s.roughness:.4f}", s.mesh_path])

    # Write spectra bundle
    np.savez_compressed(
        os.path.join(out_dir, "spectra.npz"),
        ids=np.array([s.fragment_id for s in specs]),
        spectra=spectra,
    )

    print(f"Generated {n} fragments -> {out_dir}")
    print(f"  meshes:   {mesh_dir}/*.obj")
    print(f"  metadata: {meta_path}")
    print(f"  spectra:  {os.path.join(out_dir, 'spectra.npz')}  shape={spectra.shape}")


def build_dataset_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--out", type=str, default="dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_dataset(args.n, args.out, args.seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--out", type=str, default="dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_dataset(args.n, args.out, args.seed)
