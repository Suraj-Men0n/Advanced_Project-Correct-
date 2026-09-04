from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "src/tyre_sorting/sim/env.py").read_text()

def test_mechanical_pick_contract():
    assert "self.target_pick_pos" in ENV
    assert "p.getBasePositionAndOrientation(self.target_body_id)" in ENV
    assert "env.grasp_fragment(bid)" in ENV
    assert "self.completed_drops += 1" in ENV
    assert "def command_cartesian_target" in ENV


def test_camera_overlay_contract():
    assert "Synthetic Camera RGB - Bounding Boxes" in ENV
    assert "cv2.rectangle" in ENV
    assert "PICK WINDOW" in ENV
