"""
Regression test for the camera-frame fix in sim_bridge.py / segmentation_node.py.

Verifies that world_to_camera() (used by the spectral/sim-bridge stream) and
backproject() (used by the geometry/segmentation stream) agree on the same
convention, so fusion_node's nearest-neighbour association is comparing
like with like. Uses the *corrected* intrinsics for the actual 320x240,
60deg-vertical-FOV simulated camera (fx=fy=207.85, cx=160, cy=120) -- not
the old stock-Kinect placeholder values (554.3, 554.3, 320, 240).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tyre_sorting.sim.camera_geom import world_to_camera, camera_to_world, CAMERA_TARGET, CAMERA_EYE
from tyre_sorting.perception.segmentation_node import backproject

FX = FY = 207.85
CX, CY = 160.0, 120.0


def _forward_project(x_cam, y_cam, z_cam):
    """Inverse of backproject(): camera-frame -> pixel (u, v)."""
    return CX + FX * x_cam / z_cam, CY + FY * y_cam / z_cam


def test_target_point_lands_at_image_centre():
    x, y, z = world_to_camera(CAMERA_TARGET)
    assert abs(x) < 1e-9 and abs(y) < 1e-9
    assert z == CAMERA_EYE[2] - CAMERA_TARGET[2]
    u, v = _forward_project(x, y, z)
    assert (round(u, 3), round(v, 3)) == (CX, CY)


def test_up_direction_moves_image_upward():
    # +world_x is CAMERA_UP; it must move the point to a SMALLER v
    # (upward in an OpenCV/pinhole y-down image).
    _, y0, _ = world_to_camera((0.0, 0.0, 0.0))
    _, y1, _ = world_to_camera((0.1, 0.0, 0.0))
    assert y1 < y0


def test_world_to_camera_is_exact_inverse_of_backproject():
    intr = {"fx": FX, "fy": FY, "cx": CX, "cy": CY}
    for world_pt in [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0),
                      (0.05, -0.03, 0.2), (0.3, 0.15, 0.02), (-0.2, 0.08, -0.01)]:
        x_cam, y_cam, z_cam = world_to_camera(world_pt)
        u, v = _forward_project(x_cam, y_cam, z_cam)
        rx, ry, rz = backproject((u, v), z_cam, intr)
        assert abs(rx - x_cam) < 1e-9
        assert abs(ry - y_cam) < 1e-9
        assert abs(rz - z_cam) < 1e-9


def test_corrected_principal_point_is_inside_the_actual_image():
    # The old defaults (cx=320, cy=240) sat outside a 320x240 image entirely.
    width, height = 320, 240
    assert 0 <= CX < width
    assert 0 <= CY < height


def test_camera_to_world_is_exact_inverse_of_world_to_camera():
    # This is the property pbvs_controller.py's frame-mismatch fix depends
    # on: /ee_pose is world-frame, fragment centroids are camera-frame --
    # camera_to_world() must genuinely round-trip or the "fix" just moves
    # the bug rather than removing it.
    for world_pt in [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0),
                      (0.05, -0.03, 0.2), (0.3, 0.15, 0.02), (-0.2, 0.08, -0.01),
                      (-0.45, 0.65, 0.25)]:  # includes a real workspace point
        cam_pt = world_to_camera(world_pt)
        rx, ry, rz = camera_to_world(cam_pt)
        assert abs(rx - world_pt[0]) < 1e-9
        assert abs(ry - world_pt[1]) < 1e-9
        assert abs(rz - world_pt[2]) < 1e-9
