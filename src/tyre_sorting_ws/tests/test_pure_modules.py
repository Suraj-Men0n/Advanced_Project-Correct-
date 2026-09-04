"""
Tests for everything that does NOT require PyBullet/ROS 2 -- these run
anywhere, including CI, without a robotics stack installed.
Run: PYTHONPATH=src pytest tests/ -v
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tyre_sorting.data.mesh_gen import generate_fragment
from tyre_sorting.data.spectroscopy import TYRE_COMPOSITIONS, SpectroscopyConfig, synthesize_signature
from tyre_sorting.perception.segmentation_node import segment_depth_frame
from tyre_sorting.control.pbvs_controller import pbvs_step, PBVSGains


def test_mesh_generation_watertight(tmp_path):
    rng = np.random.default_rng(1)
    spec = generate_fragment("t0", "tread", "passenger", rng, str(tmp_path))
    import trimesh
    m = trimesh.load(spec.mesh_path)
    assert m.is_watertight
    assert len(m.vertices) > 3


def test_spectroscopy_shape_and_composition_separability():
    rng = np.random.default_rng(2)
    cfg = SpectroscopyConfig(noise_std=0.01)  # low noise for a clean separability check
    s_nr = synthesize_signature(TYRE_COMPOSITIONS["otr_mining"], cfg, rng, surface_class="tread")  # NR-rich
    s_sbr = synthesize_signature(TYRE_COMPOSITIONS["passenger"], cfg, rng, surface_class="sidewall")  # SBR-rich
    assert s_nr.shape == (cfg.n_bands, cfg.n_channels)
    # NR-rich (otr_mining) should show a stronger primary NR peak than passenger
    assert s_nr[0].max() > s_sbr[0].max()


def test_segmentation_detects_isolated_fragments():
    H, W = 120, 160
    belt_depth = 0.75
    depth = np.full((H, W), belt_depth, dtype=np.float32)
    # three well-separated blobs
    for (cx, cy) in [(30, 30), (100, 40), (70, 90)]:
        yy, xx = np.mgrid[0:H, 0:W]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= 15 ** 2
        depth[mask] = belt_depth - 0.03
    _, dets = segment_depth_frame(depth, belt_depth)
    assert len(dets) == 3


def test_pbvs_converges_within_velocity_cap():
    gains = PBVSGains(max_lin_vel=0.35)
    ee_pos = np.array([0.0, 0.0, 0.3])
    ee_quat = np.array([1.0, 0, 0, 0])
    target_pos = np.array([0.1, 0.0, 0.0])
    target_quat = np.array([1.0, 0, 0, 0])
    belt_vel = np.array([0.1, 0, 0])  # well under the velocity cap
    dt = 1 / 60
    for _ in range(600):  # 10s
        lin, ang, err = pbvs_step(ee_pos, ee_quat, target_pos, target_quat, belt_vel, gains)
        ee_pos = ee_pos + lin * dt
        target_pos = target_pos + belt_vel * dt
    assert np.linalg.norm(err) < 0.005


def test_pbvs_diverges_when_target_faster_than_cap():
    gains = PBVSGains(max_lin_vel=0.2)
    ee_pos = np.array([0.0, 0.0, 0.3])
    ee_quat = np.array([1.0, 0, 0, 0])
    target_pos = np.array([0.1, 0.0, 0.0])
    target_quat = np.array([1.0, 0, 0, 0])
    belt_vel = np.array([0.5, 0, 0])  # faster than the arm can ever move
    dt = 1 / 60
    for _ in range(600):
        lin, ang, err = pbvs_step(ee_pos, ee_quat, target_pos, target_quat, belt_vel, gains)
        ee_pos = ee_pos + lin * dt
        target_pos = target_pos + belt_vel * dt
    assert np.linalg.norm(err) > 0.1  # confirms the physically-expected failure mode

def test_segmentation_exposes_bbox_and_surface_prior():
    H, W = 120, 160
    belt_depth = 0.75
    depth = np.full((H, W), belt_depth, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((xx - 50) / 22) ** 2 + ((yy - 60) / 8) ** 2 <= 1
    depth[mask] = belt_depth - 0.03
    _, dets = segment_depth_frame(depth, belt_depth)
    assert len(dets) == 1
    d = dets[0]
    assert d.bbox[2] >= d.bbox[0] and d.bbox[3] >= d.bbox[1]
    assert d.vision_surface_class in {'sidewall', 'unknown'}
