"""
T1 -- PyBullet Physics Workspace Setup.

NOTE ON THIS SANDBOX: PyBullet has no prebuilt wheel for this container's
Python/platform and compiling it from source exceeds this tool's execution
time limit, so this module was authored but NOT executed here. It targets
the standard `pybullet` >=3.2 API. On your machine:
    pip install pybullet
    python3 -m tyre_sorting.sim.env   # opens a GUI window and runs a smoke test

Provides:
  - a parametric conveyor (kinematic belt via constraint-driven velocity,
    since PyBullet has no native conveyor primitive)
  - a 6-DOF manipulator loaded from a URDF (defaults to PyBullet's bundled
    kuka_iiwa; swap for your own arm's URDF via `arm_urdf`)
  - an RGB-D virtual camera plugin (getCameraImage-based)
  - a spawner that drops fragment meshes from the T2 dataset onto the belt
"""
from __future__ import annotations

import os
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass

from tyre_sorting.sim.camera_geom import CAMERA_EYE, CAMERA_TARGET, CAMERA_UP, world_to_camera


# Collision filter groups. The arm and fragments are placed in mutually
# exclusive groups so PyBullet's broadphase never resolves contact between
# them (see p.setCollisionFilterGroupMask calls in __init__/spawn_fragment
# below). This is deliberate: there is no grasp/pick mechanism implemented
# yet (PBVS only drives the end-effector toward a converging position, it
# never attaches to a fragment), so with default collision the arm's
# 120N-driven motion was physically colliding with and scattering the
# fragment stream every time it reached for a target, jamming belt flow.
# Ground/belt/rails keep PyBullet's default group=1, mask=-1, so they're
# untouched and still collide with both the arm and fragments normally.
COLLISION_GROUP_ARM = 2
COLLISION_GROUP_FRAGMENT = 4
COLLISION_GROUP_BIN = 8   # Added alongside the bin-footprint widening
                          # (GOOD_BIN_INNER_HALF_X/Y): the arm was never
                          # excluded from bin-wall collision, because at
                          # the old, smaller wall size (0.09 square, 0.12
                          # tall) the carry trajectory had enough
                          # clearance to never touch a wall in practice.
                          # The wider, taller walls close that clearance
                          # margin -- confirmed live: the arm can now
                          # physically snag a wall mid-carry and stall
                          # there indefinitely, since neither the 'carry'
                          # nor 'return' phase had a timeout to recover
                          # from a jammed, non-progressing state. Bins
                          # get the same treatment fragments already have
                          # below: excluded from the arm's mask, and the
                          # arm excluded from theirs, so the walls remain
                          # solid to fragments (still contained properly)
                          # but the arm can pass through them freely --
                          # release always happens well above the wall
                          # top regardless (z=0.25 vs wall top 0.18), so
                          # this does not allow a fragment to visibly
                          # clip through a wall during normal operation,
                          # only removes an unintended obstacle for the
                          # arm's own transit path.

# ---------------------------------------------------------------------------
# WORKSPACE LAYOUT -- explicit zones (see README.md's SOP section)
#
# The belt runs along +x. The arm base sits beside the belt at (0.1, 0.45, 0).
#
# PICK POINT design (claw-machine style): the arm hovers at ONE fixed,
# deeply-verified (x, y) location near the belt's middle and only descends
# when a fragment is detected within a small radius underneath it -- no
# chasing a moving target across a wide zone. An earlier "Action Zone"
# design (a whole x-range, always targeting whichever fragment was furthest
# downstream, plus a belt-velocity lead prediction) kept pushing the actual
# intercept target past the edge of what was verified reachable -- confirmed
# live as repeated INTERCEPT_AND_PICK stalls even after the zone itself
# checked out fine. A single fixed point sidesteps that: find_action_zone.py
# showed x=-0.30 at the CENTRE of the comfortable range (joint_load=0.68,
# the lowest/safest value measured, with real margin on both sides -- even
# neighbours 0.10m away stayed under 0.76), not an edge case.
PICK_POINT_XY = (-0.30, 0.0)
PICK_DETECT_RADIUS = 0.15  # a fragment must be within this xy-radius of
                            # PICK_POINT_XY to be marked -- small enough that
                            # the correction INTERCEPT_AND_PICK makes never
                            # leaves the verified-comfortable neighbourhood

# Four recycling destinations, kept in a tight 2x2 cluster beside the arm.
# The coordinates deliberately stay close to the previously verified
# GOOD_BIN_DROP so the robot does not gain a long/strange IK transit path just
# because the tyre type changed.
ROUTE_DROP_POSITIONS = {
    # Single row along y at x=-0.35, verified against three constraints:
    #   (a) IK reachability -- the manipulator's target is always this
    #       fixed drop point (bx, by, 0.25), never the bin's physical
    #       walls, so reach depends only on these coordinates, not on
    #       bin size. Worst case (the y=1.03 bin) is 0.775m from the arm
    #       base (iiwa reach ~0.8m) -- re-check with find_four_bins.py
    #       if these coordinates ever move.
    #   (b) physical non-overlap in y -- bins are rectangular
    #       (GOOD_BIN_INNER_HALF_X=0.15, GOOD_BIN_INNER_HALF_Y=0.09,
    #       grown from an original 0.09 square once half the dataset's
    #       fragments turned out too large for the old footprint, see
    #       GOOD_BIN_INNER_HALF_X's own comment). Y is the axis with
    #       spacing between bins (0.21m centre-to-centre) so it stays
    #       capped near the original 0.09; X has no such constraint
    #       since only one bin occupies each row position, so it grew
    #       freely without affecting (a) or (b).
    #   (c) OUT OF THE INSPECTION CAMERA'S VIEW -- the upstream camera
    #       (see camera_geom.CAMERA_EYE) covers x in [-1.23, -0.47], so
    #       bins at x=-0.35 sit downstream of the frame entirely. The
    #       previous 2x2 grid had a column at x=-0.56 which fell inside
    #       that window and was segmented as a "fragment".
    # Nearest bin edge to the belt is ~0.24m (belt edge 0.2m) after the
    # X growth described above (was 0.298m at the old, smaller size).
    "devulcanization_cbr": (-0.35, 0.40, 0.25),        # OTR / mining
    "high_grade_crumb": (-0.35, 0.61, 0.25),           # truck / HGV
    "recompounding_raw_material": (-0.35, 0.82, 0.25), # passenger
    "speciality_crumb": (-0.35, 1.03, 0.25),           # motorcycle
}
DEFAULT_ROUTE = "recompounding_raw_material"
GOOD_BIN_XY = ROUTE_DROP_POSITIONS[DEFAULT_ROUTE][:2]
GOOD_BIN_INNER_HALF_Y = 0.09  # Unchanged: this is the axis along which the
                               # four bins are arranged in a single row
                               # (spacing 0.21m centre-to-centre), so it is
                               # capped by inter-bin non-overlap -- growing
                               # past ~0.093 here would make adjacent bins
                               # intersect.
GOOD_BIN_INNER_HALF_X = 0.15   # Grown from 0.09. Half of the dataset's
                               # fragments (160/320, see metadata.csv) have
                               # a longest footprint dimension exceeding the
                               # old 0.18m clear span (2x0.09) -- physically
                               # too large to fit inside the old bin
                               # regardless of how precisely they were
                               # released, a likely real contributor to the
                               # live delivery-rate shortfall. X has no
                               # inter-bin overlap constraint (only one bin
                               # per row position), and critically the
                               # manipulator's IK target is always the fixed
                               # drop point (bx, by, 0.25) above the bin,
                               # never the bin's physical walls -- verified
                               # numerically that reach distance to the drop
                               # point is unchanged by wall position, so
                               # growing X does not affect reachability at
                               # all. At 0.15 the clear X-span is 0.30m,
                               # comfortably fitting the largest fragment
                               # (0.26m) with margin, and the nearest bin
                               # edge remains ~0.24m clear of the belt
                               # (previously 0.298m at the old, smaller size).
GOOD_BIN_INNER_HALF = GOOD_BIN_INNER_HALF_Y  # kept for any external reference
GOOD_BIN_WALL_HEIGHT = 0.18   # Raised from 0.12. The release point (z=0.25,
                               # see ROUTE_DROP_POSITIONS) sits comfortably
                               # above the new wall top (0.25-0.18=0.07m
                               # clearance, down from 0.13m at the old
                               # height but still well clear), giving more
                               # wall to catch a fragment that drifts
                               # laterally on the way down rather than
                               # falling perfectly plumb.
GOOD_BIN_WALL_THICK = 0.012
GOOD_BIN_DROP = ROUTE_DROP_POSITIONS[DEFAULT_ROUTE]


def route_drop_position(route: str) -> tuple:
    """Return the simulator drop point for a recycling route.
    Unknown/reject routes intentionally return ``None`` so they remain in
    the belt exit/reject tray rather than being silently placed in a good bin.
    """
    pos = ROUTE_DROP_POSITIONS.get(route)
    return tuple(pos) if pos is not None else None

# HOME / hover pose = PICK_POINT_XY at hover height, satisfying the four
# stated criteria:
#  - Vertical wrist orientation: enforced separately via home_quat.
#  - Z clearance: 0.25m is well above the belt surface (0.07m) and the side
#    rails (~0.13m) -- "low to medium hover distance", non-uniform shards
#    pass underneath untouched until something is actually marked.
#  - Kinematic mid-range: PICK_POINT_XY was chosen as the *lowest* joint_load
#    point in the sweep (0.68), not just one that happened to pass.
#  - Centred near the belt's middle, per the request this responds to,
#    rather than upstream of a wide zone -- the arm only moves once a
#    fragment is actually detected nearby, so it doesn't need lead distance
#    to accelerate toward a target it hasn't confirmed yet.
HOME_POS = (PICK_POINT_XY[0], PICK_POINT_XY[1], 0.25)

# Clear of the belt rails (top ~0.13m) with real margin.
SAFE_TRANSIT_Z = 0.25

# Tyre type controls the recycling stream.  Surface class is retained as a
# per-piece attribute and eligibility/audit field; it does not invent a new
# recycling chemistry route.
TYRE_ROUTE = {
    "truck_hgv": "high_grade_crumb",
    "otr_mining": "devulcanization_cbr",
    "passenger": "recompounding_raw_material",
    "motorcycle": "speciality_crumb",
}


class WaypointNavigator:
    """Stateful lift-transit-descend navigator for carrying a held
    fragment (or approaching one) without clipping the belt's side rails.

    Replaces an earlier stateless version that recomputed lift-vs-transit
    from raw position every tick: when actual height converged to almost
    exactly the lift/transit threshold, sub-millimetre numerical noise
    flipped the decision every single tick, and the arm made no net
    progress for 100+ seconds (confirmed live via diagnose_carry.py --
    z sat at 0.16 while the printed waypoint alternated between two
    different targets each step). Phase here only ever moves forward
    (lift -> transit -> descend), so that class of oscillation can't
    happen: once a phase is left, nothing re-enters it for this episode.
    """
    def __init__(self, safe_z: float = SAFE_TRANSIT_Z, xy_tol: float = 0.08, z_tol: float = 0.05):
        self.safe_z = safe_z
        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.phase = "lift"

    def reset(self):
        self.phase = "lift"

    def step(self, ee_pos, final_target):
        """Returns (waypoint, arrived). `arrived` is only True once
        actually at final_target, never at an intermediate waypoint."""
        ee_pos = np.asarray(ee_pos, dtype=float)
        final_target = np.asarray(final_target, dtype=float)
        if self.phase == "lift" and ee_pos[2] >= self.safe_z - self.z_tol:
            self.phase = "transit"
        if self.phase == "transit":
            xy_dist = float(np.linalg.norm(ee_pos[:2] - final_target[:2]))
            if xy_dist < self.xy_tol:
                self.phase = "descend"
        if self.phase == "lift":
            return np.array([ee_pos[0], ee_pos[1], self.safe_z]), False
        if self.phase == "transit":
            return np.array([final_target[0], final_target[1], self.safe_z]), False
        arrived = float(np.linalg.norm(ee_pos - final_target)) < self.xy_tol / 2
        return final_target, arrived


import enum


class PickState(enum.Enum):
    HOME = 0                     # STATE 0: PRE_START / HOME
    DETECT_AND_MARK = 1          # STATE 1: SENSOR_DETECT_AND_MARK
    INTERCEPT_AND_PICK = 2       # STATE 2: INTERCEPT_AND_PICK
    SAFE_LIFT = 3                # STATE 3: SAFE_LIFT
    TRANSIT_TO_CONTAINER = 4     # STATE 4: TRANSIT_TO_CONTAINER
    DROP_AND_RELEASE = 5         # STATE 5: DROP_AND_RELEASE
    RESET_TO_BELT = 6            # STATE 6: RESET_TO_BELT


class PickPlaceController:
    """Explicit state machine for one full pick-and-place cycle.

    One instance per arm; call `.step(env)` once per simulation tick.
    Loops HOME -> ... -> RESET_TO_BELT -> HOME indefinitely, matching
    "loop continuously until all incoming stream fragments are
    processed."

    STATE 0 HOME: hover at the ready pose above the belt's pickup zone,
        waiting for a reachable fragment.
    STATE 1 DETECT_AND_MARK: mark the nearest tracked fragment as target.
        This standalone controller reads simulator ground-truth position
        directly rather than processing an actual depth image -- a
        deliberate, honest simplification for fast local iteration; the
        real ROS2 pipeline's perception/segmentation_node.py does actual
        depth-image-based detection from the same camera this env
        exposes via get_rgbd(). "Optimal routing criteria" here is
        nearest-by-distance, not e.g. furthest-along-belt: the arm's
        reach time can exceed a fragment's transit time across the belt,
        so targeting whichever fragment is closest to exiting (least
        time margin) is provably the worse choice -- confirmed both
        analytically and live.
    STATE 2 INTERCEPT_AND_PICK: move down onto the marked fragment
        (predicted interception point = its current position + a small
        vertical offset, re-evaluated every tick so it tracks the belt's
        motion automatically rather than needing an explicit velocity
        prediction term) and grasp it.
    STATE 3 SAFE_LIFT: rise straight up to clear the belt's side rails
        (top ~0.13m) before any lateral motion -- a direct low path from
        a belt-side grasp point to the container clips them, confirmed
        live ("smacking into the belt").
    STATE 4 TRANSIT_TO_CONTAINER: move laterally, still at clearance
        height, to directly above the container.
    STATE 5 DROP_AND_RELEASE: release well above the container's open
        top -- the fragment falls in under gravity, never touching the
        container's solid walls/floor.
    STATE 6 RESET_TO_BELT: return to HOME above the belt; loop.
    """
    def __init__(self, home_pos=HOME_POS, drop_pos=GOOD_BIN_DROP,
                 safe_z: float = SAFE_TRANSIT_Z, grasp_radius: float = 0.05,
                 xy_tol: float = 0.08, z_tol: float = 0.05,
                 approach_speed: float = 0.35,
                 pick_point_xy=PICK_POINT_XY, pick_detect_radius: float = PICK_DETECT_RADIUS,
                 state_timeout_s: float = 15.0):
        self.home_pos = np.asarray(home_pos, dtype=float)
        self.drop_pos = np.asarray(drop_pos, dtype=float)
        self.safe_z = safe_z
        self.grasp_radius = grasp_radius
        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.approach_speed = approach_speed
        self.pick_point_xy = np.asarray(pick_point_xy, dtype=float)
        self.pick_detect_radius = pick_detect_radius
        self.state_timeout_s = state_timeout_s
        self.state = PickState.HOME
        self.target_body_id: int | None = None
        self.target_route: str | None = None
        self.target_pick_pos: np.ndarray | None = None
        self.drop_pos = np.asarray(drop_pos, dtype=float)
        self.ticks_in_state = 0
        self.timeout_count = 0
        self.completed_drops = 0

    def _fragments_near_pick_point(self, env):
        """Return tracked, classified fragments passing under the fixed
        capture point.

        The old implementation selected a fragment here, then chased its
        continuously moving world position.  At 0.05 m/s that target could
        leave the verified IK neighbourhood while the arm was descending.
        Selection is now a short "capture window"; the arm descends to the
        fixed pick point instead of pursuing the moving body.
        """
        out = []
        for body_id in list(env._fragment_ids.keys()):
            try:
                pos, _ = p.getBasePositionAndOrientation(body_id)
            except p.error:
                continue  # removed between the dict read and this call
            pos = np.array(pos)
            if float(np.linalg.norm(pos[:2] - self.pick_point_xy)) <= self.pick_detect_radius:
                # PERCEPTION GATE: the depth camera must have actually
                # detected this fragment (see get_rgbd's perception gate)
                # before it can be picked. Previously any tracked body
                # inside the radius qualified, using physics ground-truth
                # position -- so the arm picked fragments before they had
                # ever appeared on camera, which demonstrated nothing about
                # the perception pipeline.
                if body_id not in env.camera_confirmed_ids:
                    continue
                out.append((body_id, pos))
        return out

    def _move_toward(self, env, ee_pos, target, speed_cap: float) -> float:
        """Drive the end-effector toward an absolute Cartesian waypoint."""
        target = np.asarray(target, dtype=float)
        dist = float(np.linalg.norm(target - ee_pos))
        if dist > 1e-4:
            env.command_cartesian_target(target, speed_cap)
        return dist

    def step(self, env) -> "PickState":
        """Advance one tick. Returns the current state (for logging)."""
        ee_pos, _ = env.get_ee_pose()
        entry_state = self.state

        # Watchdog: no state should take anywhere near this long. Every
        # stall tonight (chattering waypoints, unreachable drop target,
        # fragment snagging a container wall) manifested as one state
        # running forever with zero recovery -- burning entire 3-minute
        # runs. Aborting back to a safe state costs one wasted cycle
        # instead of the whole run, and the timeout_count is itself a
        # useful reliability metric.
        if self.state != PickState.HOME:
            self.ticks_in_state += 1
        else:
            self.ticks_in_state = 0
        if self.state != PickState.HOME and self.ticks_in_state > self.state_timeout_s * 240:
            self.timeout_count += 1
            print(f"    [watchdog] {self.state.name} exceeded "
                  f"{self.state_timeout_s}s -- aborting to "
                  f"{'RESET_TO_BELT' if env.held_body_id is not None else 'HOME'}")
            if env.held_body_id is not None:
                env.release_grasp()  # drop wherever we are; better than hanging
                self.state = PickState.RESET_TO_BELT
            else:
                self.state = PickState.HOME
            self.target_body_id = None
            self.target_route = None
            self.target_pick_pos = None
            self.drop_pos = np.asarray(env.route_drop_position(DEFAULT_ROUTE), dtype=float)
            self.ticks_in_state = 0
            return self.state

        if self.state == PickState.HOME:
            self._move_toward(env, ee_pos, self.home_pos, 0.30)
            if self._fragments_near_pick_point(env):
                self.state = PickState.DETECT_AND_MARK

        elif self.state == PickState.DETECT_AND_MARK:
            # Only fragments within PICK_DETECT_RADIUS of the fixed
            # PICK_POINT_XY are pick-eligible -- i.e. actually passing
            # underneath the hover point right now. Take whichever is
            # closest to the point itself (should already be very close,
            # by construction of the small detection radius).
            candidates = self._fragments_near_pick_point(env)
            if candidates:
                self.target_body_id = min(
                    candidates,
                    key=lambda item: float(np.linalg.norm(item[1][:2] - self.pick_point_xy)))[0]
                meta = getattr(env, "body_metadata", {}).get(self.target_body_id, {})
                self.target_route = TYRE_ROUTE.get(meta.get("tyre_type"), DEFAULT_ROUTE)
                target_drop = env.route_drop_position(self.target_route)
                if target_drop is None:
                    target_drop = env.route_drop_position(DEFAULT_ROUTE)
                    self.target_route = DEFAULT_ROUTE
                if env.freeze_fragment(self.target_body_id):
                    self.target_pick_pos = np.asarray(
                        p.getBasePositionAndOrientation(self.target_body_id)[0], dtype=float)
                    self.drop_pos = np.asarray(target_drop, dtype=float)
                    self.state = PickState.INTERCEPT_AND_PICK
                else:
                    self.target_body_id = None
                    self.target_route = None
                    self.target_pick_pos = None
                    self.state = PickState.HOME
            else:
                self.state = PickState.HOME  # nothing nearby yet

        elif self.state == PickState.INTERCEPT_AND_PICK:
            if self.target_body_id not in env._fragment_ids:
                self.target_body_id = None
                self.target_route = None
                self.target_pick_pos = None
                self.drop_pos = np.asarray(env.route_drop_position(DEFAULT_ROUTE), dtype=float)
                self.state = PickState.HOME
            else:
                # The selected piece is frozen at the capture window. Move the
                # tool to one fixed, verified pick location instead of chasing
                # its body position. This keeps the IK problem static and
                # makes the pick/drop cycle repeatable.
                if self.target_pick_pos is None:
                    self.state = PickState.HOME
                else:
                    intercept_point = np.asarray(self.target_pick_pos, dtype=float)
                    dist = self._move_toward(env, ee_pos, intercept_point, self.approach_speed)
                    if dist < self.grasp_radius:
                        bid = self.target_body_id
                        if bid in env._fragment_ids and env.grasp_fragment(bid):
                            env._fragment_ids.pop(bid, None)
                            env.unfreeze_fragment(bid)
                            self.state = PickState.SAFE_LIFT

        elif self.state == PickState.SAFE_LIFT:
            lift_target = np.array([ee_pos[0], ee_pos[1], self.safe_z])
            self._move_toward(env, ee_pos, lift_target, 0.30)
            if ee_pos[2] >= self.safe_z - self.z_tol:
                self.state = PickState.TRANSIT_TO_CONTAINER

        elif self.state == PickState.TRANSIT_TO_CONTAINER:
            transit_target = np.array([self.drop_pos[0], self.drop_pos[1], self.safe_z])
            self._move_toward(env, ee_pos, transit_target, 0.30)
            xy_dist = float(np.linalg.norm(ee_pos[:2] - self.drop_pos[:2]))
            if xy_dist < self.xy_tol:
                self.state = PickState.DROP_AND_RELEASE

        elif self.state == PickState.DROP_AND_RELEASE:
            dist = self._move_toward(env, ee_pos, self.drop_pos, 0.20)
            if dist < self.xy_tol / 2:
                env.release_grasp()
                self.completed_drops += 1
                self.target_body_id = None
                self.target_route = None
                self.target_pick_pos = None
                self.drop_pos = np.asarray(env.route_drop_position(DEFAULT_ROUTE), dtype=float)
                self.state = PickState.RESET_TO_BELT

        elif self.state == PickState.RESET_TO_BELT:
            dist = self._move_toward(env, ee_pos, self.home_pos, 0.30)
            if dist < self.xy_tol:
                self.state = PickState.HOME

        if self.state != entry_state:
            self.ticks_in_state = 0  # fresh timeout budget for the new state
        return self.state


@dataclass
class ConveyorConfig:
    length: float = 2.4          # doubled -- see SOP: Zone 2 (Identification)
                                  # needs real transit distance for the
                                  # perception pipeline to classify a
                                  # fragment before it's reachable, which
                                  # is a different bottleneck than the
                                  # arm's own cycle time (Zone 3).
    width: float = 0.4
    speed: float = 0.05          # m/s, along +x. ~48s full transit at the
                                  # new length -- real margin for both
                                  # identification (Zone 2) and pick/place
                                  # (Zone 3) before a fragment reaches
                                  # Zone 4 (reject).
    z_height: float = 0.05
    spawn_x: float = -1.1        # fragments spawn at the belt's -x end


class TyreSortingEnv:
    def __init__(self, gui: bool = True, arm_urdf: str | None = None,
                 conveyor_cfg: ConveyorConfig | None = None,
                 dataset_dir: str = "dataset"):
        self.cfg = conveyor_cfg or ConveyorConfig()
        self.dataset_dir = dataset_dir
        self.fragment_metadata = {}
        meta_path = os.path.join(self.dataset_dir, "metadata.csv")
        if os.path.exists(meta_path):
            import csv
            with open(meta_path, newline="") as f:
                self.fragment_metadata = {r["fragment_id"]: r for r in csv.DictReader(f)}
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        self.show_camera_overlay = bool(gui)
        self._camera_window_ready = False
        self._camera_overlay_error_logged = False
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        self.plane_id = p.loadURDF("plane.urdf")
        self._build_conveyor()
        self._build_bins()

        arm_urdf = arm_urdf or os.path.join(
            pybullet_data.getDataPath(), "kuka_iiwa", "model.urdf")
        # Positioned to the SIDE of the belt (clear of its y in [-0.2,0.2]
        # footprint), reaching in over the belt rather than sitting in the
        # material's path -- matches real conveyor-side pick-and-place
        # layouts, and removes the arm's own base as a possible physical
        # obstruction now that PBVS actively drives it into the fragment
        # stream. x=0.1 keeps it mid-belt-length; y=0.45 clears the belt
        # edge (0.2) by 0.25m, well within the iiwa's ~0.8m reach envelope
        # to the belt centerline (see reach check in commit notes).
        self.arm_id = p.loadURDF(arm_urdf, basePosition=[0.1, 0.45, 0.0],
                                  useFixedBase=True)
        self.n_joints = p.getNumJoints(self.arm_id)
        self.held_body_id: int | None = None
        self._held_constraint: int | None = None
        # See COLLISION_GROUP_* comment above: exclude fragments and bin
        # geometry from every arm link's collision mask, not just the base.
        arm_mask = -1 & ~COLLISION_GROUP_FRAGMENT & ~COLLISION_GROUP_BIN
        for link in range(-1, self.n_joints):
            p.setCollisionFilterGroupMask(self.arm_id, link, COLLISION_GROUP_ARM, arm_mask)

        self._fragment_ids: dict[int, float] = {}   # body_id -> spawn depth offset
        self.body_fragment_id: dict[int, str] = {}
        self.body_metadata: dict[int, dict] = {}
        # Bodies the depth camera has actually detected at least once --
        # populated in get_rgbd()'s perception gate, consulted by the pick
        # controller so nothing is picked before it has genuinely been seen.
        self.camera_confirmed_ids: set[int] = set()
        self._frozen_body_ids: set[int] = set()

        # Vertical wrist orientation (spec criterion 1): the tool frame
        # points straight down at the belt, perpendicular to the conveyor
        # plane, so it aligns cleanly with the camera/workspace frames.
        # The kuka_iiwa's zero configuration points the end-effector
        # straight UP (+z, measured: ee_pos z=1.261 at all-joints-zero),
        # so a pi rotation about x turns it to face down.
        self.home_quat = np.array(p.getQuaternionFromEuler([math.pi, 0.0, 0.0]))

        # Joint limits and null-space rest poses for singularity avoidance
        # (spec criterion 3). Queried from the actual loaded URDF at
        # runtime rather than hardcoded -- self_check caught a real bug
        # here: passing restPoses alone (without lowerLimits/upperLimits/
        # jointRanges together) does NOT make PyBullet enforce joint
        # limits at all, so IK was silently returning solutions with
        # joint 3 up to 82% PAST its physical limit -- resetJointState
        # can force a joint there directly (bypassing physics), which is
        # why the self-check still reported 0.0000m error: it was
        # confirming an unreachable, physically-impossible configuration.
        # Rest poses are the exact midpoint of each joint's real range
        # (not a hand-picked guess), so "mid-range" is guaranteed by
        # construction regardless of this URDF's specific numeric limits.
        self.joint_lower, self.joint_upper, self.joint_ranges, self.rest_poses = [], [], [], []
        for j in range(self.n_joints):
            info = p.getJointInfo(self.arm_id, j)
            lo, hi = info[8], info[9]
            self.joint_lower.append(lo)
            self.joint_upper.append(hi)
            self.joint_ranges.append(hi - lo)
            self.rest_poses.append(0.5 * (lo + hi))

        # Snap to the HOME pose. Uses resetJointState (instant teleport)
        # since this is one-time setup, not a driven motion.
        ready_joints = p.calculateInverseKinematics(
            self.arm_id, self.n_joints - 1, list(HOME_POS), self.home_quat.tolist(),
            lowerLimits=self.joint_lower, upperLimits=self.joint_upper,
            jointRanges=self.joint_ranges, restPoses=self.rest_poses,
            maxNumIterations=200, residualThreshold=1e-5)
        for j in range(min(self.n_joints, len(ready_joints))):
            p.resetJointState(self.arm_id, j, ready_joints[j])

    def route_drop_position(self, route: str):
        """Return a route drop point for the standalone simulator."""
        return route_drop_position(route)

    # ---------------- conveyor ----------------
    def _build_conveyor(self):
        c = self.cfg
        half_extents = [c.length / 2, c.width / 2, 0.02]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                   rgbaColor=[0.15, 0.15, 0.15, 1])
        self.belt_id = p.createMultiBody(
            baseMass=0,  # static base; motion is applied to fragments directly
            baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[0, 0, c.z_height])

        # Low side rails so a fragment bumped sideways during spawning can't
        # leave the belt and become permanently stuck next to it, outside
        # every "on-belt" check in step_belt(). Open at both ends so
        # fragments still exit normally off the far end.
        rail_half = [c.length / 2, 0.01, 0.03]
        rail_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=rail_half)
        rail_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=rail_half,
                                        rgbaColor=[0.1, 0.1, 0.1, 1])
        rail_z = c.z_height + 0.02 + rail_half[2]
        for side in (1, -1):
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rail_col,
                               baseVisualShapeIndex=rail_vis,
                               basePosition=[0, side * (c.width / 2 - 0.01), rail_z])

    def _build_bins(self):
        """Build four recycling bins plus the belt-exit reject tray.

        Each route gets a DISTINCT colour plus a floating text label, so a
        viewer can tell the bins apart and verify that fragments are being
        sorted correctly rather than dumped indiscriminately. Previously
        all four were the same green and were visually indistinguishable.
        """
        inner_x = GOOD_BIN_INNER_HALF_X
        inner_y = GOOD_BIN_INNER_HALF_Y
        wall_h = GOOD_BIN_WALL_HEIGHT
        wt = GOOD_BIN_WALL_THICK

        # Colour per route, matching the proposal's high-value streams.
        route_colours = {
            "devulcanization_cbr": ([0.85, 0.75, 0.15, 1], [0.70, 0.60, 0.10, 1]),   # yellow: OTR/mining
            "high_grade_crumb": ([0.20, 0.45, 0.90, 1], [0.15, 0.35, 0.75, 1]),      # blue: truck/HGV
            "recompounding_raw_material": ([0.20, 0.80, 0.35, 1], [0.15, 0.65, 0.30, 1]),  # green: passenger
            "speciality_crumb": ([0.75, 0.30, 0.85, 1], [0.60, 0.25, 0.70, 1]),      # purple: motorcycle
        }

        for route, (bx, by, _) in ROUTE_DROP_POSITIONS.items():
            floor_rgba, wall_rgba = route_colours.get(
                route, ([0.2, 0.8, 0.35, 1], [0.15, 0.65, 0.3, 1]))
            floor_half = [inner_x + wt, inner_y + wt, 0.01]
            fcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half)
            fvis = p.createVisualShape(p.GEOM_BOX, halfExtents=floor_half,
                                       rgbaColor=floor_rgba)
            floor_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=fcol,
                              baseVisualShapeIndex=fvis,
                              basePosition=[bx, by, -floor_half[2]])
            p.setCollisionFilterGroupMask(
                floor_body, -1, COLLISION_GROUP_BIN, -1 & ~COLLISION_GROUP_ARM)
            wall_specs = [
                ([wt, inner_y + wt, wall_h / 2], [bx + inner_x + wt, by, wall_h / 2]),
                ([wt, inner_y + wt, wall_h / 2], [bx - inner_x - wt, by, wall_h / 2]),
                ([inner_x + wt, wt, wall_h / 2], [bx, by + inner_y + wt, wall_h / 2]),
                ([inner_x + wt, wt, wall_h / 2], [bx, by - inner_y - wt, wall_h / 2]),
            ]
            for half, center in wall_specs:
                wcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
                wvis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                           rgbaColor=wall_rgba)
                wall_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wcol,
                                  baseVisualShapeIndex=wvis,
                                  basePosition=center)
                p.setCollisionFilterGroupMask(
                    wall_body, -1, COLLISION_GROUP_BIN, -1 & ~COLLISION_GROUP_ARM)
            # Floating label above each bin so the route is readable in the
            # GUI without cross-referencing colours against the source.
            try:
                p.addUserDebugText(route, [bx, by, wall_h + 0.06],
                                   textColorRGB=wall_rgba[:3], textSize=1.1)
            except p.error:
                pass  # DIRECT mode / no GUI -- labels are cosmetic only

        # Reject tray at the belt exit. Unknown / low-confidence material is
        # never actively carried to a good bin; it is allowed to exit here.
        c = self.cfg
        tray_x0 = c.length / 2
        floor_half = [0.35, c.width / 2 + 0.15, 0.02]
        floor_cx = tray_x0 + floor_half[0]
        fcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half)
        fvis = p.createVisualShape(p.GEOM_BOX, halfExtents=floor_half,
                                   rgbaColor=[0.8, 0.3, 0.2, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=fcol,
                          baseVisualShapeIndex=fvis,
                          basePosition=[floor_cx, 0.0, -floor_half[2]])
        wall_half = [0.02, floor_half[1], 0.06]
        wcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half)
        wvis = p.createVisualShape(p.GEOM_BOX, halfExtents=wall_half,
                                   rgbaColor=[0.65, 0.22, 0.15, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wcol,
                          baseVisualShapeIndex=wvis,
                          basePosition=[floor_cx + floor_half[0], 0.0, wall_half[2]])
        side_half = [floor_half[0], 0.02, 0.06]
        for side in (-1, 1):
            scol = p.createCollisionShape(p.GEOM_BOX, halfExtents=side_half)
            svis = p.createVisualShape(p.GEOM_BOX, halfExtents=side_half,
                                       rgbaColor=[0.65, 0.22, 0.15, 1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=scol,
                              baseVisualShapeIndex=svis,
                              basePosition=[floor_cx, side * floor_half[1], side_half[2]])
        self.reject_count = 0

    def freeze_fragment(self, body_id: int) -> bool:
        """Temporarily stop one target on the belt while the arm descends.

        This makes the standalone pick/drop demo deterministic: once a piece
        enters the capture window, it is held at that capture location instead
        of forcing the arm to chase a moving target.
        """
        if body_id not in self._fragment_ids:
            return False
        try:
            p.resetBaseVelocity(body_id, linearVelocity=[0.0, 0.0, 0.0],
                                angularVelocity=[0.0, 0.0, 0.0])
        except p.error:
            return False
        self._frozen_body_ids.add(body_id)
        return True

    def unfreeze_fragment(self, body_id: int) -> None:
        self._frozen_body_ids.discard(body_id)

    def grasp_fragment(self, body_id: int) -> bool:
        """Rigidly attach `body_id` to the end-effector via a fixed
        constraint -- a deliberately simple stand-in for a real gripper
        (no finger contacts / force closure modelled). Returns False if
        already holding something, or if `body_id` no longer exists (it
        can legitimately be removed -- exiting the belt -- in the window
        between being selected as a target and being reached; confirmed
        live via a pybullet.error crash before this check was added)."""
        if self.held_body_id is not None:
            return False
        try:
            frag_pos, frag_quat = p.getBasePositionAndOrientation(body_id)
        except p.error:
            return False  # target was removed (exited the belt) before we reached it
        ee_pos, ee_quat = self.get_ee_pose()
        # Constraint frame = the fragment's current offset from the
        # end-effector, so it doesn't visually snap/teleport on attach.
        inv_ee_pos, inv_ee_quat = p.invertTransform(ee_pos.tolist(), ee_quat.tolist())
        child_frame_pos, child_frame_quat = p.multiplyTransforms(
            inv_ee_pos, inv_ee_quat, frag_pos, frag_quat)
        self._held_constraint = p.createConstraint(
            parentBodyUniqueId=self.arm_id, parentLinkIndex=self.n_joints - 1,
            childBodyUniqueId=body_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=child_frame_pos, childFramePosition=[0, 0, 0],
            parentFrameOrientation=child_frame_quat, childFrameOrientation=[0, 0, 0, 1])
        self.held_body_id = body_id
        return True

    def release_grasp(self):
        """Detach whatever's currently held; it then falls freely."""
        if self._held_constraint is not None:
            p.removeConstraint(self._held_constraint)
        self._held_constraint = None
        self.held_body_id = None

    def step_belt(self):
        """Apply constant +x velocity to any fragment resting on the belt.
        PyBullet has no native conveyor, so we directly set the linear
        velocity's x-component on contact each step (kinematic approximation
        -- acceptable given DfR evaluation targets throughput, not exact
        friction dynamics)."""
        c = self.cfg
        for body_id in list(self._fragment_ids.keys()):
            pos, _ = p.getBasePositionAndOrientation(body_id)
            if body_id not in self._frozen_body_ids and abs(pos[2] - c.z_height) < 0.08:  # resting on belt surface
                lin_vel, ang_vel = p.getBaseVelocity(body_id)
                p.resetBaseVelocity(body_id, linearVelocity=[c.speed, lin_vel[1], lin_vel[2]],
                                     angularVelocity=ang_vel)
            off_far_end = pos[0] > c.length / 2
            off_side_or_fallen = abs(pos[1]) > c.width / 2 + 0.05 or pos[2] < -0.2
            if off_far_end:
                # Reached the belt's end without being picked in time --
                # stop tracking/pushing it (so it settles under gravity
                # into the reject tray, see _build_bins) but keep the
                # body alive: this is the "worse pieces" destination.
                del self._fragment_ids[body_id]
                self.body_fragment_id.pop(body_id, None)
                self.body_metadata.pop(body_id, None)
                self.camera_confirmed_ids.discard(body_id)
                self.reject_count += 1
            elif off_side_or_fallen:  # genuinely stuck/fell -- not a normal exit, recycle
                p.removeBody(body_id)
                del self._fragment_ids[body_id]
                self.body_fragment_id.pop(body_id, None)
                self.body_metadata.pop(body_id, None)
                self.camera_confirmed_ids.discard(body_id)

    # ---------------- fragment spawning ----------------
    def spawn_fragment(self, obj_path: str, mass: float = 0.05, fragment_id: str | None = None) -> int:
        c = self.cfg
        col = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_path,
                                   rgbaColor=[0.2, 0.2, 0.2, 1])
        y = np.random.uniform(-c.width / 3, c.width / 3)
        body_id = p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[c.spawn_x, y, c.z_height + 0.08])
        # See COLLISION_GROUP_* comment near the top of this file.
        p.setCollisionFilterGroupMask(body_id, -1, COLLISION_GROUP_FRAGMENT,
                                       -1 & ~COLLISION_GROUP_ARM)
        self._fragment_ids[body_id] = c.z_height
        if fragment_id is None:
            fragment_id = os.path.splitext(os.path.basename(obj_path))[0]
        self.body_fragment_id[body_id] = fragment_id
        self.body_metadata[body_id] = self.fragment_metadata.get(fragment_id, {})
        return body_id

    # ---------------- RGB-D camera plugin ----------------
    def get_rgbd(self, width: int = 320, height: int = 240,
                 eye=CAMERA_EYE, target=CAMERA_TARGET) -> tuple[np.ndarray, np.ndarray]:
        """Top-down camera over the belt. Returns (rgb[H,W,3], depth[H,W] in meters)."""
        view = p.computeViewMatrix(cameraEyePosition=list(eye),
                                    cameraTargetPosition=list(target),
                                    cameraUpVector=list(CAMERA_UP))
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height,
                                             nearVal=0.1, farVal=2.0)
        _, _, rgb_raw, depth_raw, _ = p.getCameraImage(
            width, height, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb = np.reshape(rgb_raw, (height, width, 4))[:, :, :3].copy()
        depth_buf = np.reshape(depth_raw, (height, width))
        near, far = 0.1, 2.0
        depth = far * near / (far - (far - near) * depth_buf)

        try:
            import cv2
            from tyre_sorting.perception.segmentation_node import segment_depth_frame
            # belt_plane_depth MUST match the true camera-to-belt-top
            # distance, computed from scene geometry rather than hardcoded:
            # a stale constant here silently breaks all segmentation.
            #
            # Was hardcoded 0.85 while the real value is
            # CAMERA_EYE.z - belt_top = 0.9 - 0.07 = 0.83. Since the
            # foreground test is (belt_plane_depth - depth) > depth_tol,
            # i.e. (0.85 - 0.83) = 0.02 > 0.01, THE EMPTY BELT ITSELF
            # registered as foreground -- segmentation returned one huge
            # blob spanning belt, arm and bins, so boxes were drawn around
            # the robot and the green bin while real fragments were never
            # isolated. That also starved the perception gate, so nothing
            # was ever pick-eligible (0 delivered, 0 watchdog timeouts:
            # the arm correctly sat at HOME because it was never offered
            # a confirmed target).
            belt_top_z = self.cfg.z_height + 0.02  # belt box half-thickness
            belt_plane_depth = float(CAMERA_EYE[2] - belt_top_z)
            _, detections = segment_depth_frame(
                depth.astype(np.float32), belt_plane_depth=belt_plane_depth,
                min_area_px=12)

            fx = fy = height / (2.0 * math.tan(math.radians(60.0) / 2.0))
            cx, cy = width / 2.0, height / 2.0

            def project(world_xyz):
                xc, yc, zc = world_to_camera(world_xyz)
                if zc <= 0.05:
                    return None
                return (fx * xc / zc + cx, fy * yc / zc + cy)

            # BELT REGION OF INTEREST: keep only detections whose image
            # column falls within the belt's own width.
            #
            # The camera's y-footprint is much wider than the belt, so it
            # also sees the arm base (y=0.45) and both bin rows (y=0.40,
            # 0.60). Those are genuinely raised above the belt plane, so
            # the depth threshold correctly flags them -- they were being
            # reported as "fragments" and drawn with tread/sidewall labels
            # (visible live as boxes around the robot arm and the green
            # bins). A real inspection cell defines an ROI over the belt
            # for exactly this reason; this is that ROI.
            #
            # world_to_camera maps world y to the image's x axis, so the
            # belt's usable width in world y becomes a column range.
            #
            # Uses the belt INTERIOR (inside the side rails), not the full
            # belt width: the rails sit at y=+/-(width/2 - 0.01) raised
            # 0.03m above the belt surface, so they clear the depth
            # threshold and segment as two long thin blobs running the
            # length of the image. A long thin blob has a low aspect ratio,
            # which the vision prior classifies as "sidewall" -- that was
            # the full-height sidewall box seen live, i.e. the camera
            # detecting the conveyor's own rails as a tyre fragment.
            rail_inset = 0.04  # keep detections clear of both rails
            belt_half_w = self.cfg.width / 2.0 - rail_inset
            # Project at the CAMERA'S OWN x, not x=0: the upstream camera
            # no longer sits above the belt's midpoint, and projecting a
            # point outside the frustum gives a meaningless column.
            roi_cols = []
            for sign in (-1.0, 1.0):
                q = project((CAMERA_EYE[0], sign * belt_half_w, belt_top_z))
                if q is not None:
                    roi_cols.append(q[0])
            if len(roi_cols) == 2:
                roi_u_min, roi_u_max = min(roi_cols), max(roi_cols)
            else:  # projection failed -- fall back to accepting everything
                roi_u_min, roi_u_max = 0.0, float(width)

            def on_belt(det):
                # Require the WHOLE bbox inside the ROI, not just its
                # midpoint: a rail blob spans nearly the full image height
                # and is wide enough that its midpoint can fall inside a
                # midpoint-only test while the blob itself is mostly rail.
                u0, v0, u1, v1 = det.bbox
                return roi_u_min <= u0 and u1 <= roi_u_max

            detections = [d for d in detections if on_belt(d)]

            # PERCEPTION GATE: record which fragment bodies the depth camera
            # has ACTUALLY detected this frame. The pick controller consults
            # this set (see PickPlaceController._fragments_near_pick_point)
            # so a fragment cannot be picked until the camera has genuinely
            # seen it.
            #
            # Without this, the controller read
            # p.getBasePositionAndOrientation for every tracked body --
            # physics-engine ground truth -- so it knew exactly where every
            # fragment was whether or not the camera had ever observed it,
            # and picked pieces before they entered camera view. Same class
            # of issue as the bounding-box overlay: real perception was being
            # computed and then bypassed.
            #
            # Association is nearest-neighbour in image space: project each
            # body's centre into the camera and pair it with the closest
            # detection bbox centre within a pixel tolerance. This stands in
            # for the ROS2 pipeline's tracker.py, which does the equivalent
            # job across real message streams.
            for det in detections:
                u0, v0, u1, v1 = det.bbox
                du, dv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
                best_id, best_d = None, 1e9
                for body_id in list(self._fragment_ids.keys()):
                    try:
                        bpos, _ = p.getBasePositionAndOrientation(body_id)
                    except p.error:
                        continue
                    q = project(bpos)
                    if q is None:
                        continue
                    d = math.hypot(q[0] - du, q[1] - dv)
                    if d < best_d:
                        best_d, best_id = d, body_id
                if best_id is not None and best_d < 30.0:  # px tolerance
                    # Require successful IDENTIFICATION, not merely detection.
                    # The gate previously confirmed any matched blob, so a
                    # fragment the classifier returned "unknown" for was
                    # still pick-eligible -- the arm picked pieces the
                    # camera had explicitly failed to identify. A real cell
                    # would route an unidentified piece to reject, not to a
                    # material-specific bin.
                    if det.vision_surface_class in ("tread", "sidewall"):
                        self.camera_confirmed_ids.add(best_id)

            # Draw boxes ONLY from what depth-based segmentation actually
            # detected in THIS frame (the `detections` computed above).
            #
            # This previously projected each body's physics AABB and labelled
            # it from spawn-time ground-truth metadata, for every tracked
            # fragment regardless of whether the camera could see it -- which
            # is why boxes appeared before pieces entered the camera view.
            # That demonstrated nothing about perception: the real
            # segmentation was already being computed here and then thrown
            # away. A fragment outside the frustum, or too small/occluded to
            # segment, now correctly gets NO box.
            for det in detections:
                u0, v0, u1, v1 = det.bbox
                x0 = max(0, int(u0)); y0 = max(0, int(v0))
                x1 = min(width - 1, int(u1)); y1 = min(height - 1, int(v1))
                if x1 <= x0 or y1 <= y0:
                    continue
                # Label is the VISION classifier's own output plus its
                # confidence -- including "unknown" when the geometry prior
                # can't decide, which is itself honest perception output.
                label = f"{det.vision_surface_class} {det.vision_surface_confidence:.2f}"
                cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(rgb, label, (x0, max(14, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)

            pu = project((PICK_POINT_XY[0], PICK_POINT_XY[1], self.cfg.z_height))
            if pu is not None:
                cv2.circle(rgb, (int(pu[0]), int(pu[1])),
                           max(5, int(fx * PICK_DETECT_RADIUS / 0.85)), (255, 0, 255), 2)
                cv2.putText(rgb, 'PICK WINDOW', (max(2, int(pu[0]) - 34), min(height - 4, int(pu[1]) + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 0, 255), 1, cv2.LINE_AA)

            if self.show_camera_overlay:
                if not self._camera_window_ready:
                    cv2.namedWindow('Synthetic Camera RGB - Bounding Boxes', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Synthetic Camera RGB - Bounding Boxes', width * 2, height * 2)
                    self._camera_window_ready = True
                cv2.imshow('Synthetic Camera RGB - Bounding Boxes',
                           cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
        except Exception as exc:
            if self.show_camera_overlay and not self._camera_overlay_error_logged:
                print(f'[camera overlay] disabled: {exc}')
                self._camera_overlay_error_logged = True
        return rgb, depth

    def get_ee_pose(self):
        """Return end-effector world pose using the KUKA final link."""
        link = self.n_joints - 1
        state = p.getLinkState(self.arm_id, link, computeForwardKinematics=True)
        return np.asarray(state[4], dtype=float), np.asarray(state[5], dtype=float)

    def command_cartesian_target(self, target: np.ndarray, speed_cap: float = 0.35):
        """Position-control the end-effector toward an absolute Cartesian target."""
        target = np.asarray(target, dtype=float)
        joints = p.calculateInverseKinematics(
            self.arm_id, self.n_joints - 1, target.tolist(), self.home_quat.tolist(),
            lowerLimits=self.joint_lower, upperLimits=self.joint_upper,
            jointRanges=self.joint_ranges, restPoses=self.rest_poses,
            maxNumIterations=100, residualThreshold=1e-5)
        max_vel = max(0.8, float(speed_cap) * 5.0)
        for j in range(min(self.n_joints, len(joints))):
            info = p.getJointInfo(self.arm_id, j)
            if info[2] != p.JOINT_FIXED:
                p.setJointMotorControl2(
                    self.arm_id, j, p.POSITION_CONTROL, targetPosition=float(joints[j]),
                    force=500, positionGain=0.75, velocityGain=0.6, maxVelocity=max_vel)

    def apply_cartesian_velocity(self, linear: np.ndarray, angular: np.ndarray):
        """Compatibility wrapper for ROS/PBVS commands.

        Look-ahead dt was hardcoded to 1/60 (the ROS timer's period), but
        since sim_bridge.py's tick() fix (4 control+physics iterations per
        60Hz tick, to match PyBullet's real 1/240s step) this is called
        4x per tick -- effectively at 240Hz. A stale 1/60 look-ahead
        placed the IK carrot 4x further ahead than intended on every one
        of those calls, compounding with the joint velocity cap into
        erratic, overshoot-prone motion. 1/240 matches the actual call
        rate and PyBullet's own timestep (p.setTimeStep in __init__).
        """
        pos, _ = self.get_ee_pose()
        target = pos + np.asarray(linear, dtype=float) * (1.0 / 240.0)
        self.command_cartesian_target(target, float(np.linalg.norm(linear)) + 0.25)

    def step(self):
        self.step_belt()
        p.stepSimulation()

    def close(self):
        if self.show_camera_overlay:
            try:
                import cv2
                cv2.destroyWindow('Synthetic Camera RGB - Bounding Boxes')
            except Exception:
                pass
        p.disconnect(self.client)


def self_check(env, seed: int = 42, tolerance: float = 0.02) -> bool:
    """Pre-flight IK/FK reachability check -- run this before the arm
    starts moving for real. For each named target, solves IK, applies
    the solution via resetJointState (instantaneous, not a driven
    motion), reads back the resulting end-effector position via forward
    kinematics, and compares it to the intended target. A large residual
    means that target is not actually reachable in this
    orientation/configuration -- catching that here costs milliseconds;
    discovering it live costs a stalled demo (confirmed: 175s with zero
    diagnostic signal beyond "state didn't change").

    Checks, per the request this responds to:
    - ZERO position: the URDF's default all-joints-zero configuration --
      a sanity check that the arm model and forward kinematics are
      behaving as expected at all (not an IK target -- there's nothing
      to solve for here, just report what it is).
    - RANDOM position: a reproducible (fixed-seed) sample from the
      arm's real operating envelope, catching IK problems that aren't
      specific to any one hand-picked point.
    - HOME and GOOD_BIN_DROP: the two coordinates the state machine
      actually depends on.

    Restores the arm's original joint configuration before returning --
    this check must never leave side effects on the pose the real run
    starts from.
    """
    original_joints = [p.getJointState(env.arm_id, j)[0] for j in range(env.n_joints)]
    rng = np.random.default_rng(seed)
    # Diagnostic random point is sampled only from the verified pick
    # neighbourhood.  The old broad sample could land in an IK-singular,
    # near-limit pose and made a useful sanity check fail even though the
    # actual sorting workspace target was valid.
    random_target = np.array([
        rng.uniform(PICK_POINT_XY[0] - 0.10, PICK_POINT_XY[0] + 0.10),
        rng.uniform(PICK_POINT_XY[1] - 0.10, PICK_POINT_XY[1] + 0.10),
        rng.uniform(0.10, 0.25)])

    print("=== Pre-flight IK/FK self-check ===")
    for j in range(env.n_joints):
        p.resetJointState(env.arm_id, j, 0.0)
    zero_ee_pos, _ = env.get_ee_pose()
    print(f"  ZERO position       : all joints=0 -> ee_pos={zero_ee_pos.round(3)} "
          f"(sanity check only, no target to compare against)")
    # Restore before the target loop below -- the bug this replaces left
    # the arm at the zero-pose's orientation (pointing straight up) for
    # every subsequent check, since nothing restored it here. That's why
    # HOME failed by 0.34m live: IK was being asked to reach a low
    # position while locked to a "pointing up" orientation, an
    # artificially impossible combination this check accidentally
    # created, not a real reachability problem.
    for j in range(env.n_joints):
        p.resetJointState(env.arm_id, j, original_joints[j])

    all_ok = True
    targets = {"HOME": np.asarray(HOME_POS), "RANDOM": random_target}
    targets.update({f"DROP_{route.upper()}": np.asarray(pos)
                    for route, pos in ROUTE_DROP_POSITIONS.items()})
    for name, target in targets.items():
        joints = p.calculateInverseKinematics(
            env.arm_id, env.n_joints - 1, target.tolist(), env.home_quat.tolist(),
            lowerLimits=env.joint_lower, upperLimits=env.joint_upper,
            jointRanges=env.joint_ranges, restPoses=env.rest_poses,
            maxNumIterations=200, residualThreshold=1e-5)
        for j in range(min(env.n_joints, len(joints))):
            p.resetJointState(env.arm_id, j, joints[j])
        achieved_pos, achieved_quat = env.get_ee_pose()
        error = float(np.linalg.norm(achieved_pos - target))
        status = "OK" if error < tolerance else "UNREACHABLE"
        if error >= tolerance:
            all_ok = False

        # Spec criterion 1 -- vertical wrist: the tool's local -z axis
        # should point down (world -z). Rotate the local z axis by the
        # achieved orientation and check its world-frame z component.
        rot = p.getMatrixFromQuaternion(achieved_quat.tolist())
        tool_z_world = np.array([rot[2], rot[5], rot[8]])
        downward = float(np.dot(tool_z_world, np.array([0.0, 0.0, -1.0])))

        # Spec criterion 3 -- kinematic mid-range: how close is each joint
        # to the middle of its own range? 0.0 = dead centre, 1.0 = at a
        # limit. A worst-case near 1.0 means a near-singular pose.
        worst_norm, worst_j = 0.0, -1
        for j in range(env.n_joints):
            info = p.getJointInfo(env.arm_id, j)
            lower, upper = info[8], info[9]
            if upper > lower:  # a real, limited joint
                mid = 0.5 * (lower + upper)
                half_range = 0.5 * (upper - lower)
                norm = abs(p.getJointState(env.arm_id, j)[0] - mid) / max(half_range, 1e-9)
                if norm > worst_norm:
                    worst_norm, worst_j = norm, j
        midrange_flag = "" if worst_norm < 0.85 else "  <-- NEAR JOINT LIMIT"

        print(f"  {name:20s}: target={target.round(3)} achieved={achieved_pos.round(3)} "
              f"error={error:.4f}m -> {status}")
        print(f"  {'':20s}  wrist_downward={downward:+.3f} (want ~+1.0)  "
              f"worst_joint_load={worst_norm:.2f} (0=centred, 1=at limit, "
              f"joint {worst_j}){midrange_flag}")

    for j in range(env.n_joints):  # restore -- this check must be side-effect-free
        p.resetJointState(env.arm_id, j, original_joints[j])
    print(f"=== self-check {'PASSED' if all_ok else 'FAILED'} ===\n")
    return all_ok


def smoke_test():
    """Spawns a few fragments from the T2 dataset and runs the belt for a
    few seconds -- run this first on your machine to confirm the workspace
    is wired up correctly before building on top of it."""
    import glob
    env = TyreSortingEnv(gui=True)
    meshes = sorted(glob.glob("dataset/meshes/*.obj"))[:6]
    for i, m in enumerate(meshes):
        env.spawn_fragment(m)
        for _ in range(60):          # let each fragment land and settle
            env.step()
            time.sleep(1.0 / 240.0)  # real-time pacing -- without this the
                                      # whole burst executes near-instantly
                                      # and meshes can spawn on top of each
                                      # other before the previous one settles

    for _ in range(1000):
        env.step()
        time.sleep(1.0 / 240.0)

    rgb, depth = env.get_rgbd()
    print("captured frame:", rgb.shape, depth.shape, "depth range:", depth.min(), depth.max())
    env.close()


def pick_and_sort_demo(speed: float = 0.05, spawn_interval: float = 5.0,
                        gui: bool = True, max_spawns: int = 12,
                        n_steps: int = 21600, verbose: bool = True):
    """Standalone demo of the 7-state pick-and-place cycle (see
    PickPlaceController), no ROS2 needed -- run this first, before the
    full ROS2 stack, to check the state machine and grasp mechanics in
    isolation. Much faster to iterate on than `ros2 launch`.

    Spawns fragments periodically; PickPlaceController drives the arm
    through HOME -> DETECT_AND_MARK -> INTERCEPT_AND_PICK -> SAFE_LIFT ->
    TRANSIT_TO_CONTAINER -> DROP_AND_RELEASE -> RESET_TO_BELT -> HOME,
    looping for as long as the demo runs. Fragments the arm doesn't
    manage to pick before they reach the belt's end settle into the
    reject tray instead (see step_belt/_build_bins) -- the belt's own
    natural exit is the "worse pieces" destination, no extra sort logic
    needed.

    speed, spawn_interval, gui, max_spawns and n_steps default to exactly
    what this function always used, so `python3 -m tyre_sorting.sim.env
    sort` is unchanged. They're parameters so run_sweep.py (O5) can call
    this directly for each (speed, spawn_interval) combination instead of
    a manual edit-rebuild-run cycle per data point.

    Returns a dict of the same metrics this function prints (see the end
    of the function for the exact keys), for the sweep script to collect.
    """
    import glob
    env = TyreSortingEnv(gui=gui, conveyor_cfg=ConveyorConfig(speed=speed))
    self_check_ok = self_check(env)
    if not self_check_ok and verbose:
        print("WARNING: self-check found an unreachable target above -- "
              "the run below may stall the same way. Fix the flagged "
              "coordinate before relying on this run's results.\n")
    meshes = sorted(glob.glob("dataset/meshes/*.obj"))
    controller = PickPlaceController()

    last_spawn = -spawn_interval
    delivered_count = 0
    total_spawned = 0
    last_state = controller.state
    last_cam = 0.0
    cycle_start_t = None       # sim-time (i/240) when the current pick attempt began
    cycle_times = []
    per_route_delivered: dict[str, int] = {}
    in_flight_route = None           # completed HOME->RESET_TO_BELT durations, seconds

    # Delivered and rejected fragments are never removed from the
    # simulation (only untracked) -- they settle and stay as live
    # physics bodies permanently. Confirmed live: by t=115s of an
    # uncapped run, 125 live bodies had accumulated and the physics step
    # cost had grown enough to stall the arm entirely, independent of any
    # control logic. Capping total spawns is what actually bounds this --
    # 40 keeps the live body count in a range that stays fast for the
    # whole run. The dataset itself has 320+ unique meshes (confirmed,
    # real) -- this cap is a per-demo-run choice for simulation
    # stability, not a claim about dataset size.
    MAX_SPAWNS = max_spawns  # was a hardcoded local; now a parameter so
                              # run_sweep.py can control it per-run

    for i in range(n_steps):
        sim_t = i / 240.0  # simulated time -- correct regardless of how
                            # fast this loop actually executes in real
                            # time. Previously this used time.time(), which
                            # was only valid because the loop's sleep()
                            # forced wall-clock time to track simulated
                            # time; without that sleep (headless runs)
                            # wall-clock barely advances across thousands
                            # of fast ticks and spawn/camera logic would
                            # almost never fire.
        if (sim_t - last_spawn > spawn_interval and meshes
                and total_spawned < MAX_SPAWNS):
            idx = int(sim_t / spawn_interval) % len(meshes)
            env.spawn_fragment(meshes[idx])
            last_spawn = sim_t
            total_spawned += 1

        state = controller.step(env)
        if state != last_state:
            if verbose:
                print(f"  [t={i/240:.1f}s] {last_state.name} -> {state.name}")
            if state == PickState.DETECT_AND_MARK and last_state == PickState.HOME:
                cycle_start_t = i / 240.0
            if state == PickState.INTERCEPT_AND_PICK and controller.target_body_id is not None:
                bid = controller.target_body_id
                meta = env.body_metadata.get(bid, {})
                in_flight_route = controller.target_route
                if verbose:
                    print(f"    target body={bid} tyre={meta.get('tyre_type','unknown')} "
                          f"surface={meta.get('surface_class','unknown')} route={controller.target_route} "
                          f"pick_pos={np.round(controller.target_pick_pos,3) if controller.target_pick_pos is not None else None}")
            if controller.completed_drops > delivered_count:
                delivered_count = controller.completed_drops
                if cycle_start_t is not None:
                    cycle_times.append(i / 240.0 - cycle_start_t)
                if in_flight_route:
                    per_route_delivered[in_flight_route] = \
                        per_route_delivered.get(in_flight_route, 0) + 1
                if verbose:
                    print(f"    delivered -> {in_flight_route} "
                          f"(total {delivered_count}, rejected/unpicked: {env.reject_count})")
            last_state = state

        env.step()
        if sim_t - last_cam > 0.1:  # ~10Hz. This is no longer just a GUI
                                    # refresh: get_rgbd() now drives the
                                    # perception gate that decides which
                                    # fragments are pick-eligible, so the
                                    # rate matters. At the old 3Hz a
                                    # fragment could cross the pick window
                                    # between frames and never be confirmed.
            env.get_rgbd()
            last_cam = sim_t
        if gui:
            time.sleep(1.0 / 240.0)  # paces the GUI to real time for a
                                       # human watching; pure dead time in
                                       # headless sweep runs, so skipped
                                       # there -- a 9-point sweep would
                                       # otherwise cost ~13 minutes of
                                       # sleeping with no one watching
        if i % 1200 == 0 and verbose:
            print(f"  ...t={i/240:.0f}s, state={controller.state.name}, "
                  f"tracked_fragments={len(env._fragment_ids)}, "
                  f"camera_confirmed={len(env.camera_confirmed_ids)}, "
                  f"delivered={delivered_count}, rejected={env.reject_count}")

    elapsed_min = n_steps / 240.0 / 60.0
    total_resolved = delivered_count + env.reject_count
    pick_success_rate = delivered_count / total_resolved if total_resolved > 0 else None
    mean_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else None
    throughput = delivered_count / elapsed_min if elapsed_min > 0 else 0.0

    if verbose:
        print(f"\nfinal: spawned={total_spawned}, delivered={delivered_count}, "
              f"rejected/unpicked={env.reject_count}")
        print("--- efficiency metrics (SOP evaluation methodology) ---")
        if pick_success_rate is not None:
            print(f"pick_success_rate: {pick_success_rate:.3f} "
                  f"(delivered / (delivered + rejected))")
        if mean_cycle_time is not None:
            print(f"mean_cycle_time: {mean_cycle_time:.1f}s "
                  f"over {len(cycle_times)} completed cycles")
        print(f"throughput: {throughput:.2f} deliveries/min "
              f"(belt_speed={env.cfg.speed}, belt_length={env.cfg.length}, "
              f"spawn_interval={spawn_interval}s)")
        print("deliveries per route (should match each fragment's classified tyre type):")
        for _r, _n in sorted(per_route_delivered.items()):
            print(f"    {_r}: {_n}")

    # PHYSICAL PLACEMENT AUDIT.
    # The per-route counts above come from the controller's own bookkeeping,
    # which only proves it *intended* the right bin. This instead reads each
    # surviving fragment's actual resting position out of the physics engine,
    # works out which bin it is physically sitting in, and compares that to
    # the bin its tyre_type should map to (TYRE_ROUTE). This is what actually
    # answers "did the truck tread end up in the truck bin, not the bike bin".
    if verbose:
        print("physical placement audit (where fragments ACTUALLY came to rest):")
    bin_half_x = GOOD_BIN_INNER_HALF_X + GOOD_BIN_WALL_THICK
    bin_half_y = GOOD_BIN_INNER_HALF_Y + GOOD_BIN_WALL_THICK
    correct = wrong = unplaced = 0
    for body_id, meta in list(env.body_metadata.items()):
        try:
            pos, _ = p.getBasePositionAndOrientation(body_id)
        except p.error:
            continue  # body already removed
        tyre = meta.get("tyre_type")
        expected_route = TYRE_ROUTE.get(tyre)
        if expected_route is None:
            continue
        landed_in = None
        for route, (bx, by, _bz) in ROUTE_DROP_POSITIONS.items():
            if abs(pos[0] - bx) <= bin_half_x and abs(pos[1] - by) <= bin_half_y:
                landed_in = route
                break
        if landed_in is None:
            unplaced += 1
        elif landed_in == expected_route:
            correct += 1
        else:
            wrong += 1
            if verbose:
                print(f"    MISPLACED: {tyre} fragment expected in "
                      f"{expected_route} but rests in {landed_in}")
    total_placed = correct + wrong
    sorting_accuracy = correct / total_placed if total_placed > 0 else None
    if verbose:
        if total_placed:
            print(f"    correctly placed: {correct}/{total_placed} "
                  f"({100.0 * correct / total_placed:.0f}% sorting accuracy)")
        else:
            print("    no fragments found resting inside any bin")
        if unplaced:
            print(f"    still on belt / in reject tray / elsewhere: {unplaced}")
        print(f"watchdog_timeouts: {controller.timeout_count} "
              f"(states aborted for exceeding {controller.state_timeout_s}s; "
              f"0 is the healthy value)")
    env.close()

    return {
        "speed": speed,
        "spawn_interval": spawn_interval,
        "spawned": total_spawned,
        "delivered": delivered_count,
        "rejected": env.reject_count,
        "pick_success_rate": pick_success_rate,
        "mean_cycle_time_s": mean_cycle_time,
        "throughput_per_min": throughput,
        "sorting_accuracy": sorting_accuracy,
        "watchdog_timeouts": controller.timeout_count,
        "self_check_passed": self_check_ok,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sort":
        pick_and_sort_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        _env = TyreSortingEnv(gui=True)
        self_check(_env)
        _env.close()
    else:
        smoke_test()
