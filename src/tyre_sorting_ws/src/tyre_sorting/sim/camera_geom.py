"""
Fixed top-down camera pose and the world<->camera-frame transform.

Kept separate from sim/env.py (which imports pybullet at module scope) so
this pure geometry can be unit-tested without pybullet installed, per the
project's "dependency-light" test layer.
"""
from __future__ import annotations

# Fixed top-down camera pose. env.py's get_rgbd() must use these SAME values
# when calling p.computeViewMatrix, or this transform silently goes stale.
#
# INSPECTION POSE covering BOTH the upstream identification area and the
# pick point. Centred at x=-0.65, 0.98m above the belt surface: the frame
# spans x in [-1.22, -0.08], which includes the spawn point (x=-1.1) AND
# the pick point (x=-0.30) with room to spare, so the arm and end effector
# are clearly visible in RGB during a pick while fragments are still
# identified well upstream. Verified at this pose: the bins, arm base and
# both conveyor side rails all project OUTSIDE the belt-interior ROI used
# by env.py's get_rgbd(), so none of them can be reported as fragments.
#
# History: an earlier attempt at z=2.3 aimed at the belt's middle put the
# ROBOT dead centre in frame and shrank the belt to a thin strip -- the arm
# and bins were then segmented as "fragments" and labelled tread/sidewall.
# The belt-interior ROI filter in env.py's get_rgbd() is what prevents that
# now, which is why the camera can safely be raised for a better view.
CAMERA_EYE = (-0.65, 0.0, 1.05)
CAMERA_TARGET = (-0.65, 0.0, 0.0)
CAMERA_UP = (1.0, 0.0, 0.0)


def world_to_camera(world_xyz, eye=CAMERA_EYE):
    """Convert a PyBullet WORLD-frame point to this camera's OpenCV-style
    optical frame (x=right, y=down, z=depth-away-from-camera) -- the SAME
    convention segmentation_node.backproject() uses. This lets
    sim_bridge.py report spectral-sample centroids in a frame that
    fusion_node.py can directly nearest-neighbour match against the
    geometry stream's backprojected centroids.

    Derived by hand for the FIXED top-down pose above (camera forward =
    world -Z, camera "up" = world +X): re-derive this if eye/target/up
    ever change, or replace with a general view-matrix inverse.

        x_cam = -(world_y - eye_y)
        y_cam = -(world_x - eye_x)
        z_cam =  (eye_z - world_z)

    Sanity checks (see tests/test_camera_geom.py): a point at CAMERA_TARGET
    gives x_cam=y_cam=0 and z_cam = the eye-to-target distance; a point
    offset in +world_x (the "up" direction) gives negative y_cam (upward in
    an OpenCV y-down image), matching CAMERA_UP; and this transform composed
    with the forward pinhole projection is the exact inverse of
    segmentation_node.backproject().
    """
    wx, wy, wz = world_xyz
    ex, ey, ez = eye
    return (-(wy - ey), -(wx - ex), ez - wz)


def camera_to_world(cam_xyz, eye=CAMERA_EYE):
    """Inverse of world_to_camera above -- converts a point in this
    camera's optical frame back to PyBullet WORLD frame.

    Needed because /ee_pose (sim_bridge.py's _publish_pose, frame_id=
    'world') and fragment centroids (camera-optical frame, per
    world_to_camera's docstring) are NOT the same frame -- a real bug
    confirmed live in pbvs_controller.py, which was subtracting them
    directly (pos_error = target_pos - ee_pos) with no conversion.

    Solving world_to_camera's three equations for world_x/y/z:
        x_cam = -(world_y - eye_y)  =>  world_y = eye_y - x_cam
        y_cam = -(world_x - eye_x)  =>  world_x = eye_x - y_cam
        z_cam =  (eye_z - world_z)  =>  world_z = eye_z - z_cam
    """
    xc, yc, zc = cam_xyz
    ex, ey, ez = eye
    return (ex - yc, ey - xc, ez - zc)
