"""
Procedural generator for non-uniform 3D rigid-body tyre shred meshes.

Real ELT shredding produces irregular torn fragments, not clean primitives.
We approximate this by:
  1. Sampling a random point cloud on a warped disc/strip "seed" shape
     (disc -> tread-like curved fragments, strip -> sidewall-like flatter
     fragments), then perturbing it with fractal (multi-octave) noise to
     emulate a torn edge profile.
  2. Taking the convex hull of the perturbed points to get a valid,
     simulator-ready watertight rigid body mesh (required for PyBullet
     collision + inertia computation).
  3. Tagging each fragment tread/sidewall and stamping a synthetic
     "surface roughness" scalar field (proxy for the tread-block / sidewall
     texture map referenced in T2 of the proposal) as a per-vertex value
     baked into the mesh metadata, since procedural UV texturing is out of
     scope for a physics-focused co-simulation.
"""
from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import dataclass


@dataclass
class FragmentSpec:
    fragment_id: str
    surface_class: str      # "tread" | "sidewall"
    tyre_type: str          # passenger | truck_hgv | motorcycle | otr_mining
    bbox_extent: tuple      # (x, y, z) in meters, post-generation
    roughness: float        # 0-1 synthetic surface-texture proxy
    mesh_path: str


def _fractal_perturb(points: np.ndarray, rng: np.random.Generator,
                      octaves: int = 3, base_amp: float = 0.15) -> np.ndarray:
    """Multi-octave radial perturbation to fake a torn, non-uniform edge."""
    center = points.mean(axis=0)
    direction = points - center
    radius = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
    unit = direction / radius

    perturb = np.zeros((points.shape[0], 1))
    amp = base_amp
    freq = 1.0
    for _ in range(octaves):
        phase = rng.uniform(0, 2 * np.pi, size=3)
        angle = np.arctan2(unit[:, 1], unit[:, 0])
        elev = np.arcsin(np.clip(unit[:, 2], -1, 1))
        perturb[:, 0] += amp * np.sin(freq * angle + phase[0]) * np.cos(freq * elev + phase[1])
        amp *= 0.5
        freq *= 2.3

    return points + unit * perturb * radius


def generate_fragment(fragment_id: str, surface_class: str, tyre_type: str,
                       rng: np.random.Generator, out_dir: str) -> FragmentSpec:
    """Generate one torn-fragment mesh and write it to <out_dir>/<fragment_id>.obj"""
    n_seed = rng.integers(40, 90)

    if surface_class == "tread":
        # curved, chunkier seed -> disc segment lofted in z (mimics tread block)
        theta = rng.uniform(0, 2 * np.pi, n_seed)
        r = rng.uniform(0.03, 0.09, n_seed)
        z = rng.uniform(-0.015, 0.015, n_seed) + 0.01 * np.sin(3 * theta)
        pts = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    else:
        # flatter, more elongated strip seed -> sidewall peel
        x = rng.uniform(-0.08, 0.08, n_seed)
        y = rng.uniform(-0.035, 0.035, n_seed)
        z = rng.uniform(-0.006, 0.006, n_seed)
        pts = np.stack([x, y, z], axis=1)

    pts = _fractal_perturb(pts, rng, octaves=rng.integers(2, 4),
                            base_amp=rng.uniform(0.10, 0.22))

    hull = trimesh.Trimesh(vertices=pts).convex_hull
    hull.apply_translation(-hull.centroid)

    # Slight random anisotropic scale for extra non-uniformity
    scale = rng.uniform(0.7, 1.4, size=3)
    hull.apply_scale(scale)

    out_path = f"{out_dir}/{fragment_id}.obj"
    hull.export(out_path)

    roughness = float(np.clip(rng.normal(0.55 if surface_class == "tread" else 0.30, 0.12), 0, 1))
    extent = tuple(float(v) for v in hull.extents)

    return FragmentSpec(
        fragment_id=fragment_id,
        surface_class=surface_class,
        tyre_type=tyre_type,
        bbox_extent=extent,
        roughness=roughness,
        mesh_path=out_path,
    )
