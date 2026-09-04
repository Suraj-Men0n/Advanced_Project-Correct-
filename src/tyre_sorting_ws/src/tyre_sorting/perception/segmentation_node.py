"""
T3 -- Geometric Segmentation & Dynamic Tracking.

Pure image-processing core (framework-agnostic) so it can be unit-tested
without a live RGB-D stream, plus a thin ROS 2 wrapper node at the bottom
that subscribes to the camera plugin's depth topic and republishes fragment
detections.

Pipeline:
  depth image -> foreground mask (belt-plane subtraction)
  -> distance transform + watershed (splits touching/overlapping fragments)
  -> per-fragment centroid, bbox, pixel area
  -> (with camera intrinsics) back-projected 3D centroid for the PBVS node
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class FragmentDetection:
    label: int
    centroid_px: tuple          # (u, v)
    centroid_3d: tuple | None   # (x, y, z) in camera/world frame, if intrinsics given
    bbox: tuple                 # (u0, v0, u1, v1)
    pixel_area: int
    vision_surface_class: str = "unknown"
    vision_surface_confidence: float = 0.0


def segment_depth_frame(depth: np.ndarray, belt_plane_depth: float,
                         depth_tol: float = 0.01, min_area_px: int = 25,
                         ) -> tuple[np.ndarray, list[FragmentDetection]]:
    """
    depth: HxW float32 depth image (meters), as would be published by the
           RGB-D camera plugin in T1.
    belt_plane_depth: known depth of the empty conveyor surface, used to
           subtract the background (fragments sit above the belt, i.e. at
           a smaller depth value from a top-down camera).
    Returns (label_map, detections).
    """
    fg_mask = ((belt_plane_depth - depth) > depth_tol).astype(np.uint8) * 255

    # Clean small noise, then compute distance transform for watershed seeds
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)

    if dist.max() <= 0:
        return np.zeros(depth.shape, dtype=np.int32), []

    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(sure_fg)

    sure_bg = cv2.dilate(fg_mask, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed needs a 3-channel image
    color_stub = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_stub, markers)

    detections: list[FragmentDetection] = []
    for lbl in range(2, n_markers + 1):  # label 1 is background, -1 is boundary
        ys, xs = np.where(markers == lbl)
        if xs.size < min_area_px:
            continue
        u0, u1, v0, v1 = xs.min(), xs.max(), ys.min(), ys.max()
        centroid_px = (float(xs.mean()), float(ys.mean()))
        bbox_w = max(1, int(u1 - u0 + 1))
        bbox_h = max(1, int(v1 - v0 + 1))
        aspect = min(bbox_w, bbox_h) / max(bbox_w, bbox_h)
        # Simulator baseline: tread chunks are generated compact/rounder,
        # sidewall chunks as flatter elongated strips.  This is intentionally
        # a lightweight geometry/vision prior, not a learned real-camera model.
        if aspect >= 0.68:
            vision_surface_class = "tread"
        elif aspect <= 0.55:
            vision_surface_class = "sidewall"
        else:
            vision_surface_class = "unknown"
        vision_surface_confidence = float(min(1.0, abs(aspect - 0.615) / 0.30))

        detections.append(FragmentDetection(
            label=lbl, centroid_px=centroid_px, centroid_3d=None,
            bbox=(int(u0), int(v0), int(u1), int(v1)), pixel_area=int(xs.size),
            vision_surface_class=vision_surface_class,
            vision_surface_confidence=vision_surface_confidence,
        ))

    return markers, detections


def backproject(centroid_px: tuple, depth_value: float, intrinsics: dict) -> tuple:
    """Pinhole back-projection: pixel (u,v) + depth -> camera-frame (x,y,z)."""
    u, v = centroid_px
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    z = depth_value
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (float(x), float(y), float(z))


# --------------------------------------------------------------------------
# ROS 2 wrapper (uncomment / run inside your ROS 2 + PyBullet environment).
# Kept import-guarded so this module stays importable/testable without ROS 2
# installed, per the sandbox note in the project README.
# --------------------------------------------------------------------------
def build_ros2_node():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from tyre_sorting_msgs.msg import FragmentArray, Fragment  # custom msgs, see msg/

    class SegmentationNode(Node):
        def __init__(self):
            super().__init__("geometry_segmentation_node")
            self.bridge = CvBridge()
            # 0.83 = CAMERA_EYE.z (0.9) - belt top surface z (0.07).
            # Was 0.85, which is BELOW the true value: since the foreground
            # test is (belt_plane_depth - depth) > depth_tol, the empty belt
            # itself (depth 0.83) passed as foreground and segmentation
            # returned one huge blob covering the belt, arm and bins instead
            # of individual fragments. Confirmed live in the standalone demo.
            self.declare_parameter("belt_plane_depth", 0.83)
            # NOTE: these must match the simulated camera in sim/env.py::get_rgbd
            # (320x240 render, 60deg VERTICAL FOV -> fx=fy=height/(2*tan(fov/2)),
            # cx=width/2, cy=height/2). The previous defaults (554.3, 554.3, 320.0,
            # 240.0) were stock Kinect-at-640x480 intrinsics that don't match this
            # camera at all -- cx=320/cy=240 fall outside a 320x240 image entirely,
            # and fx/fy were ~2.7x too large. If you change width/height/fov in
            # get_rgbd(), recompute these to match.
            self.declare_parameter("fx", 207.85)
            self.declare_parameter("fy", 207.85)
            self.declare_parameter("cx", 160.0)
            self.declare_parameter("cy", 120.0)

            self.sub = self.create_subscription(
                Image, "/camera/depth/image_raw", self.on_depth, 10)
            self.pub = self.create_publisher(FragmentArray, "/perception/fragments", 10)

        def on_depth(self, msg: Image):
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            belt_depth = self.get_parameter("belt_plane_depth").value
            _, dets = segment_depth_frame(depth, belt_depth)

            intr = dict(
                fx=self.get_parameter("fx").value, fy=self.get_parameter("fy").value,
                cx=self.get_parameter("cx").value, cy=self.get_parameter("cy").value,
            )
            out = FragmentArray()
            out.header = msg.header
            for d in dets:
                u, v = int(d.centroid_px[0]), int(d.centroid_px[1])
                z = float(depth[v, u])
                x, y, z = backproject(d.centroid_px, z, intr)
                frag = Fragment()
                frag.id = d.label
                frag.centroid.x, frag.centroid.y, frag.centroid.z = x, y, z
                frag.pixel_area = d.pixel_area
                frag.bbox_xmin, frag.bbox_ymin = d.bbox[0], d.bbox[1]
                frag.bbox_xmax, frag.bbox_ymax = d.bbox[2], d.bbox[3]
                frag.vision_surface_class = d.vision_surface_class
                frag.vision_surface_confidence = float(d.vision_surface_confidence)
                out.fragments.append(frag)
            self.pub.publish(out)

    return SegmentationNode


def main():
    import rclpy
    rclpy.init()
    NodeCls = build_ros2_node()
    node = NodeCls()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
