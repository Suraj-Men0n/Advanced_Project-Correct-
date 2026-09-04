"""
T5 -- Closed-Loop ROS 2 Visual Servoing Node.

Framework-agnostic PBVS control law: given the current end-effector pose and
a (possibly moving) target fragment pose, compute a Cartesian velocity
command that drives the tracking error to zero, with feed-forward
compensation for the target's known belt velocity so the controller doesn't
perpetually lag a moving target (pure proportional control on a moving
target has non-zero steady-state error -- feed-forward removes it).
"""
from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass


@dataclass
class PBVSGains:
    kp_lin: float = 4.0      # proportional gain, linear
    kp_ang: float = 3.0      # proportional gain, angular
    max_lin_vel: float = 0.35   # m/s, end-effector speed cap
    max_ang_vel: float = 2.0    # rad/s


# Verified-reachable belt x-range for the iiwa base at (0.1, 0.45, 0) --
# see find_action_zone.py, which swept x in [-0.55, 0.60] at pick height
# and confirmed low, comfortable joint_load across that whole span. The
# camera's field of view extends much further upstream (world x down to
# -1.22, see camera_geom.py) than the arm can actually reach, so without
# this filter the very first fragment of a run -- spawned at x=-1.1 (1.28m
# from the base, camera-visible, classification-eligible) -- gets locked
# as the PBVS target and the arm sits saturated against its reach limit
# for the ~20s it takes to drift into range, producing IK-clamp
# discontinuities that look identical to target-identity oscillation.
REACHABLE_X_MIN = -0.55
REACHABLE_X_MAX = 0.55

# Fixed camera-frame-independent approach height for a PBVS target. The
# camera-derived z (via camera_to_world) reports the fragment's actual
# resting height on the belt surface (~0.07m) -- driving the end-effector
# to that z drives it into the belt itself (only FRAGMENT collisions are
# masked out for the arm, see sim/env.py's COLLISION_GROUP_* comment), so
# pos_error's z-component never converges. Constant vertical position
# error saturates the linear velocity clamp in pbvs_step() (kp_lin=4.0 *
# ~0.18m error alone exceeds max_lin_vel=0.35), which starves the x/y
# correction that's actually doing the visual servoing and looks exactly
# like oscillation. Holding z at a fixed clearance above the belt
# (matching find_action_zone.py's own pick_z=0.10, and PickPlaceController's
# grasp height) lets x/y -- the axes the camera signal actually informs --
# converge, while z is handled the same deliberate way the state machine
# already treats it.
PBVS_APPROACH_Z = 0.10


def pbvs_step(ee_pos: np.ndarray, ee_quat: np.ndarray,
              target_pos: np.ndarray, target_quat: np.ndarray,
              target_lin_vel: np.ndarray, gains: PBVSGains,
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One control step of position-based visual servoing.

    Returns (linear_vel_cmd[3], angular_vel_cmd[3], pos_error[3]) in the
    world/base frame. Angular part uses a simplified small-angle quaternion
    error (adequate for the gripper-alignment tolerances in this task;
    swap in a full SO(3) log map if you need large-angle reorientation).
    """
    pos_error = target_pos - ee_pos
    lin_cmd = gains.kp_lin * pos_error + target_lin_vel  # feed-forward tracks belt motion

    speed = np.linalg.norm(lin_cmd)
    if speed > gains.max_lin_vel:
        lin_cmd = lin_cmd / speed * gains.max_lin_vel

    # Quaternion error (vector part ~ proportional to rotation error for small angles)
    q_err = _quat_mul(target_quat, _quat_conj(ee_quat))
    ang_cmd = gains.kp_ang * 2.0 * q_err[1:] * np.sign(q_err[0] + 1e-9)
    ang_speed = np.linalg.norm(ang_cmd)
    if ang_speed > gains.max_ang_vel:
        ang_cmd = ang_cmd / ang_speed * gains.max_ang_vel

    return lin_cmd, ang_cmd, pos_error


def select_locked_target(eligible_world: list[tuple[object, np.ndarray]],
                          locked_pos: np.ndarray | None,
                          locked_wall: float | None,
                          belt_velocity: np.ndarray,
                          now_wall: float,
                          gate_dist: float = 0.10,
                          ) -> tuple[object, np.ndarray, np.ndarray, float]:
    """
    Pick which fragment to track this cycle, preferring continuity with
    whatever was already locked over freshly re-ranking every candidate.

    eligible_world: list of (fragment_msg, world_pos) pairs already passed
    classification/surface filtering.
    locked_pos/locked_wall: the previously tracked target's last known
    world position and the wall-clock time it was last confirmed, or
    (None, None) if nothing is currently locked.

    Returns (target_fragment, target_world_pos, new_locked_pos, new_locked_wall).
    """
    if locked_pos is not None:
        dt_lock = now_wall - locked_wall
        predicted = locked_pos + belt_velocity * dt_lock
        nearest_f, nearest_pos = min(
            eligible_world, key=lambda fp: np.linalg.norm(fp[1] - predicted))
        if np.linalg.norm(nearest_pos - predicted) <= gate_dist:
            return nearest_f, nearest_pos, nearest_pos, now_wall
        # Nothing near where the locked target should be -- it's gone
        # (picked, fell off, or dropped out of classification for a
        # frame). Fall through to acquiring a fresh target below.

    # No active lock (or it was just lost) -- acquire one. Prefer the
    # piece furthest along +x (smallest y_cam), because belt motion is
    # +world_x.
    target, target_pos = min(eligible_world, key=lambda fp: fp[0].centroid.y)
    return target, target_pos, target_pos, now_wall


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


# --------------------------------------------------------------------------
# ROS 2 node wrapper -- subscribes to /perception/fragments (T3) and the
# arm's current pose, publishes Cartesian velocity commands to the
# controller_manager. Import-guarded; see README for the sandbox note.
# --------------------------------------------------------------------------
def build_ros2_node():
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import TwistStamped, PoseStamped
    from tyre_sorting_msgs.msg import FragmentArray
    from tyre_sorting.sim.camera_geom import camera_to_world

    class PBVSNode(Node):
        def __init__(self):
            super().__init__("pbvs_controller_node")
            self.gains = PBVSGains()
            self.ee_pose = None
            # Was hardcoded to 0.20, while sim_bridge.py's actual belt_speed
            # default is 0.05 -- a 4x feed-forward mismatch. Now declared as
            # a parameter matching sim_bridge's own parameter name/default,
            # so a launch file can set both consistently in one place.
            self.belt_velocity = np.array([
                float(self.declare_parameter('belt_speed', 0.05).value), 0.0, 0.0])

            # Target lock: segmentation_node assigns ids from
            # cv2.connectedComponents, which are fresh per-frame labels
            # (numbered by pixel scan order), not persistent identities --
            # matching by .id across messages is meaningless. Track the
            # locked target's own last WORLD position instead, predict
            # where it should be now via belt_velocity, and only accept a
            # new detection as "the same fragment" if it falls within
            # this gate distance of that prediction. Without this, picking
            # the nearest eligible fragment fresh on every single message
            # let the selected target's identity flip whenever two
            # fragments were simultaneously visible and their ordering
            # shifted -- reversing the commanded velocity direction every
            # time and producing a visibly oscillating, non-converging
            # arm (reported live as the arm "DJ-ing" over the belt,
            # picking nothing).
            self.locked_target_pos = None
            self.locked_target_wall = None
            self.gate_dist = 0.10  # m; belt moves ~0.005m per 100ms at 0.05 m/s

            # Stuck-lock guard: confirmed live (dissertation Section 4.5/5)
            # that select_locked_target()'s coordinate-range reachability
            # filter is a coarse gate, not an IK-solvability check -- a
            # fragment can sit at an x within REACHABLE_X_MIN/MAX while
            # still being unreachable at that exact y/z, so the lock never
            # naturally breaks (predicted position keeps tracking the real
            # one at belt_velocity, so the gate_dist check never fails) and
            # the controller camps on it. Observed episode lengths were
            # 18.5-55.2s across 6 occurrences in 5 live runs. This bounds
            # the damage: if the SAME continuous lock has been held this
            # long without ever being released (which only happens via
            # sim_bridge.py transitioning to 'carry' after a successful
            # grasp, which clears and reacquires the lock naturally), force
            # a reset so eligible_world is re-evaluated from scratch rather
            # than camping indefinitely. 10s is chosen with margin above
            # normal convergence time (offline worst case 6.72s at the
            # highest tested belt speed; live pbvs runs use the 0.05 m/s
            # default, where convergence is under 2s) so it should not
            # false-trigger on a target that is genuinely still converging.
            self.lock_acquired_wall = None
            self.stuck_lock_timeout_s = 10.0

            self.create_subscription(PoseStamped, "/ee_pose", self.on_ee_pose, 10)
            # Was /perception/fragments (raw geometry, no classification) --
            # /perception/fused_fragments (fusion_node.py) carries the same
            # Fragment/FragmentArray types, just with material_class/
            # classification_confidence populated, so target selection can
            # use classification once needed without any message changes.
            self.create_subscription(FragmentArray, "/perception/fused_fragments",
                                      self.on_fragments, 10)
            self.cmd_pub = self.create_publisher(TwistStamped, "/arm/cartesian_velocity_cmd", 10)

        def on_ee_pose(self, msg: PoseStamped):
            p, o = msg.pose.position, msg.pose.orientation
            self.ee_pose = (np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z]))

        def on_fragments(self, msg: FragmentArray):
            if self.ee_pose is None or not msg.fragments:
                return
            # Select only a positively classified, routable fragment.
            #
            # Was also gated on surface_class in {'tread', 'sidewall'} --
            # removed. Section 6 of the project notes already documents
            # surface_class as near-chance (~50-54%) because the synthetic
            # spectral signature never encoded a real tread/sidewall
            # signal, and fusion_node.py's own confidence arbitration
            # falls back to 'unknown' whenever neither modality clears
            # 0.60 -- which happens on a large fraction of messages (the
            # giant-bbox log excerpt showed confidences sitting at
            # 0.43-0.51). Every reseed of that noise (sim_bridge.py's
            # _publish_spectra() reseeds ~10x/sec per fragment) could flip
            # this gate, dropping the LOCKED target out of `eligible` for
            # a frame even though it's still visible and still the right
            # target -- which forces a fall-through to fresh acquisition
            # (or, if nothing else is eligible, a hard reset to no lock)
            # regardless of how correct select_locked_target() itself is.
            # This is why the target-lock fix alone didn't resolve the
            # live oscillation: the lock was working, it just kept being
            # handed an empty list to work with. routing_node/decision
            # logic never used surface_class for bin decisions either
            # (see decision/routing.py), so it has no business gating
            # arm control.
            eligible = [
                f for f in msg.fragments
                if getattr(f, 'material_class', '') in {
                    'passenger', 'truck_hgv', 'motorcycle', 'otr_mining'}
                and float(getattr(f, 'classification_confidence', 0.0)) >= 0.65
            ]

            # Reachability + approach-height filter -- see REACHABLE_X_MIN/
            # MAX and PBVS_APPROACH_Z above for why both are necessary.
            eligible_world = []
            for f in eligible:
                wx, wy, wz = camera_to_world(
                    np.array([f.centroid.x, f.centroid.y, f.centroid.z]))
                if not (REACHABLE_X_MIN <= wx <= REACHABLE_X_MAX):
                    continue
                eligible_world.append((f, np.array([wx, wy, PBVS_APPROACH_Z])))

            now_wall = time.time()

            # Stuck-lock guard -- see the attribute comments in __init__.
            # Force a reset if the current continuous lock has run past
            # the timeout without ever being cleared by a successful grasp.
            was_locked = self.locked_target_pos is not None
            if (was_locked and self.lock_acquired_wall is not None
                    and (now_wall - self.lock_acquired_wall) > self.stuck_lock_timeout_s):
                self.get_logger().warn(
                    f"stuck-lock timeout ({self.stuck_lock_timeout_s:.0f}s) -- "
                    f"abandoning current target and re-acquiring",
                    throttle_duration_sec=1.0)
                self.locked_target_pos = None
                self.locked_target_wall = None
                was_locked = False

            # Throttled diagnostic: distinguishes "nothing detected",
            # "detected but filtered out", and "filtered in but out of
            # reach" at a glance, without needing a one-off debug patch
            # for every future investigation.
            self.get_logger().info(
                f"fragments={len(msg.fragments)} eligible={len(eligible)} "
                f"reachable={len(eligible_world)} "
                f"locked={'yes' if self.locked_target_pos is not None else 'no'}",
                throttle_duration_sec=2.0)

            if not eligible_world:
                self.locked_target_pos = None
                return

            target, target_pos, self.locked_target_pos, self.locked_target_wall = (
                select_locked_target(eligible_world, self.locked_target_pos,
                                      self.locked_target_wall, self.belt_velocity,
                                      now_wall, self.gate_dist))

            if not was_locked:
                self.lock_acquired_wall = now_wall

            target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # geometry-only target orientation

            ee_pos, ee_quat = self.ee_pose
            lin, ang, err = pbvs_step(ee_pos, ee_quat, target_pos, target_quat,
                                       self.belt_velocity, self.gains)

            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z = lin.tolist()
            cmd.twist.angular.x, cmd.twist.angular.y, cmd.twist.angular.z = ang.tolist()
            self.cmd_pub.publish(cmd)

    return PBVSNode


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
