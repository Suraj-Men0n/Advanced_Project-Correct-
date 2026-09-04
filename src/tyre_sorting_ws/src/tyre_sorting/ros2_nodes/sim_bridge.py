"""ROS 2 bridge for the PyBullet simulation.

Publishes RGB-D frames (with the simulator bounding-box overlay), synthetic
sensor signatures, and the arm end-effector pose. The simulator controller
performs the deterministic pick -> lift -> route -> drop -> repeat cycle while
ROS perception/classification nodes process the same streams.
"""
from __future__ import annotations
import os, glob, time
import numpy as np
import pybullet as p
from tyre_sorting.data.spectroscopy import TYRE_COMPOSITIONS, SpectroscopyConfig, synthesize_signature
from tyre_sorting.sim.env import (TyreSortingEnv, ConveyorConfig, GOOD_BIN_DROP,
                                   PickPlaceController, PickState, WaypointNavigator,
                                   HOME_POS, SAFE_TRANSIT_Z, DEFAULT_ROUTE,
                                   ROUTE_DROP_POSITIONS, GOOD_BIN_INNER_HALF_X,
                                   GOOD_BIN_INNER_HALF_Y, GOOD_BIN_WALL_THICK, TYRE_ROUTE)
from tyre_sorting.sim.camera_geom import world_to_camera
from tyre_sorting.decision.routing import route_fragment
from tyre_sorting.paths import resolve_data_path

def build_node():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from tyre_sorting_msgs.msg import SpectralArray, SpectralSample
    from tyre_sorting_msgs.msg import FragmentArray, Fragment

    class SimulationBridge(Node):
        def __init__(self):
            super().__init__('pybullet_sim_bridge')
            speed = float(self.declare_parameter('belt_speed', 0.05).value)
            gui = bool(self.declare_parameter('gui', True).value)
            self.dataset_dir = resolve_data_path(self.declare_parameter('dataset_dir', 'dataset').value)
            self.env = TyreSortingEnv(gui=gui, conveyor_cfg=ConveyorConfig(speed=speed), dataset_dir=self.dataset_dir)
            self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
            self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
            self.spec_pub = self.create_publisher(SpectralArray, '/sensors/spectroscopy', 10)
            self.ee_pub = self.create_publisher(PoseStamped, '/ee_pose', 10)
            self.cmd_sub = self.create_subscription(TwistStamped, '/arm/cartesian_velocity_cmd', self.on_cmd, 10)
            self.cmd = np.zeros(6)
            self.last_cmd_wall = 0.0   # when the last PBVS command arrived
            # 'state_machine' (default): the arm is driven by
            #   PickPlaceController using ground-truth fragment positions --
            #   the path proven by the standalone demo, and the one that
            #   performs full pick-and-place including grasp and drop.
            # 'pbvs': the arm is driven by the Cartesian velocity commands
            #   published by pbvs_controller_node, which derive from
            #   CAMERA-BASED fragment centroids -- this is the actual O4
            #   closed visual-servoing loop.
            #
            # This parameter exists because the loop was NOT closed before:
            # on_cmd stored incoming commands in self.cmd and nothing ever
            # read them, so pbvs_controller_node was computing and
            # publishing correct velocities into the void while the arm
            # moved entirely on ground truth. O4 could never have been
            # demonstrated in that configuration.
            self.control_mode = str(self.declare_parameter(
                'control_mode', 'state_machine').value)
            if self.control_mode not in ('state_machine', 'pbvs'):
                self.get_logger().warn(
                    f"unknown control_mode '{self.control_mode}', "
                    "falling back to 'state_machine'")
                self.control_mode = 'state_machine'
            self.get_logger().info(f"control_mode = {self.control_mode}")

            # PBVS grasp/carry sub-state. Previously pbvs mode only ever
            # called apply_cartesian_velocity() -- there was no grasp
            # trigger and the arm/fragment collision groups are mutually
            # exclusive (see COLLISION_GROUP_* in sim/env.py), so "picks
            # nothing up" was true by construction, independent of how
            # well the tracking converged. Once select_locked_target()
            # drives the end-effector within grasp range of a real
            # fragment, hand off from visual servoing to a fixed
            # lift/transit/drop carry (reusing WaypointNavigator, the same
            # component the standalone demo's carry phase is built on) --
            # the same deliberate simplification PickPlaceController
            # already uses for the post-grasp portion of its cycle, ground
            # truth from the environment for the drop mechanics, camera-
            # derived tracking for the actual reach-and-grasp.
            self.pbvs_phase = 'track'   # 'track' -> 'carry' -> 'return' -> 'track'
            self.pbvs_nav = WaypointNavigator(safe_z=SAFE_TRANSIT_Z)
            self.pbvs_drop_pos = None
            self.pbvs_grasp_radius = 0.06
            self.pbvs_home_pos = np.asarray(HOME_POS, dtype=float)
            self.pbvs_completed = 0
            self._last_pbvs_completed = 0

            # Carry/return stall guard: confirmed live that the arm can
            # physically snag on scene geometry mid-carry (see
            # COLLISION_GROUP_BIN's comment in sim/env.py for the bin-wall
            # case that motivated this) and simply never satisfy the
            # 'arrived'/home-proximity check again, freezing indefinitely
            # with nothing else in either phase to notice or recover.
            # Track when the current phase was entered; if 'carry' or
            # 'return' runs far longer than its normal ~2-3s duration
            # (observed live), force a recovery rather than hang forever.
            self.pbvs_phase_entered_wall = time.time()
            self.pbvs_phase_timeout_s = 12.0

            self.bridge = None
            try:
                from cv_bridge import CvBridge
                self.bridge = CvBridge()
            except Exception:
                self.get_logger().error('cv_bridge is required for the simulation bridge')
            self.spec_cfg = SpectroscopyConfig(seed=123)
            self.meta = self._load_meta()
            self.last_spawn = 0.0
            self.total_spawned = 0
            self.controller = PickPlaceController()
            self.last_state = self.controller.state
            self.timer = self.create_timer(1/60.0, self.tick)
            self._shutdown_logged = False
            self._last_progress_wall = time.time()

        def _load_meta(self):
            import csv
            p = os.path.join(self.dataset_dir, 'metadata.csv')
            out = {}
            if os.path.exists(p):
                with open(p, newline='') as f:
                    for row in csv.DictReader(f): out[row['fragment_id']] = row
            return out

        def on_cmd(self, msg):
            self.cmd[:] = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                           msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
            self.last_cmd_wall = time.time()

        def _spawn(self):
            # Delivered/rejected fragments are never removed from the
            # simulation, only untracked -- capping total spawns is what
            # actually bounds the live body count over a long-running
            # demo (see sim/env.py's MAX_SPAWNS comment for the live
            # confirmation: an uncapped run reached 125 live bodies and
            # stalled the arm entirely from physics-step cost alone).
            # 40, not 12: the dissertation's live integration table needs
            # enough fragments per run for the delivery-rate confidence
            # interval to mean something -- at 12/run, a pooled two-run
            # Wilson 95% CI came out 37 percentage points wide (roughly
            # "somewhere between a third and three-quarters delivered"),
            # too uninformative to support any real throughput claim. 40
            # is the same live-body count sim/env.py's own standalone
            # demo already runs by default and documents as staying fast
            # for a full run, so this is not a new risk, just reusing a
            # ceiling already confirmed safe (see env.py's MAX_SPAWNS
            # comment for the live confirmation that an uncapped run
            # reached 125 bodies and stalled -- 40 is comfortably clear
            # of that).
            if self.total_spawned >= 40:
                return
            self._last_progress_wall = time.time()
            meshes = sorted(glob.glob(os.path.join(self.dataset_dir, 'meshes', '*.obj')))
            if not meshes: return
            self.env.spawn_fragment(meshes[int(time.time()*10) % len(meshes)])
            self.total_spawned += 1

        def _publish_img(self, rgb, depth):
            if self.bridge is None: return
            stamp = self.get_clock().now().to_msg()
            a = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8'); a.header.stamp = stamp; a.header.frame_id='camera_link'
            d = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1'); d.header.stamp = stamp; d.header.frame_id='camera_link'
            self.rgb_pub.publish(a); self.depth_pub.publish(d)

        def _publish_pose(self):
            pos, quat = self.env.get_ee_pose()
            msg = PoseStamped(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id='world'
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float,pos)
            msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = map(float,quat)
            self.ee_pub.publish(msg)

        def _publish_spectra(self):
            from tyre_sorting.data.spectroscopy import synthesize_signature
            msg = SpectralArray(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id='camera_link'
            # Sensor simulation exposes a noisy signature for every currently tracked body.
            for body_id in list(self.env._fragment_ids.keys()):
                frag_id = self.env.body_fragment_id.get(body_id, f'body_{body_id}')
                row = self.meta.get(frag_id)
                if not row: continue
                comp = TYRE_COMPOSITIONS[row['tyre_type']]
                rng = np.random.default_rng(body_id + int(time.time()*10))
                sig = synthesize_signature(comp, self.spec_cfg, rng, surface_class=row['surface_class']).astype(np.float32).ravel()
                s = SpectralSample(); s.id = int(body_id)
                pos, _ = p.getBasePositionAndOrientation(body_id)
                # Ground-truth world position -> camera-optical frame, so this
                # matches the geometry stream's backprojected centroids (see
                # sim/camera_geom.py for the derivation).
                cx, cy, cz = world_to_camera(pos)
                s.centroid.x = float(cx); s.centroid.y = float(cy); s.centroid.z = float(cz)
                s.values = sig.tolist(); msg.samples.append(s)
            self.spec_pub.publish(msg)

        def tick(self):
            now = time.time()
            if now - self.last_spawn > 5.0:
                self._spawn()
                self.last_spawn = now

            # PyBullet's internal timestep is fixed at 1/240s
            # (p.setTimeStep in TyreSortingEnv.__init__), but this ROS2
            # timer only fires at 60Hz. The standalone demo (env.py)
            # calls controller.step() and env.step() together, once each,
            # at ~240Hz -- every gain, watchdog timeout, and self_check
            # threshold was calibrated against that 1:1 ratio. Simply
            # wrapping env.step() alone in a 4x loop would fix belt speed
            # but leave the ARM under-driven: apply_cartesian_velocity's
            # dt=1/240 assumption is baked into a single call inside
            # controller.step(), so calling that once per tick while
            # physics ran 4x underneath it would still move the arm at
            # only 1/4 its intended speed even with the belt now correct.
            # Running 4 full control+physics iterations per tick restores
            # the exact ratio the standalone demo uses: 60 * 4 * 1/240 =
            # 1.0 real-time ratio, with control and physics still 1:1.
            #
            # Confirmed live before this fix: an "incredibly slow" belt
            # and a 163-second wait before the first pick even started,
            # where the standalone demo took ~15-20s for the same thing.
            for _ in range(4):
                if self.control_mode == 'pbvs':
                    # O4 CLOSED LOOP: drive the arm from the Cartesian
                    # velocity commands published by pbvs_controller_node,
                    # computed from CAMERA-derived fragment centroids (via
                    # /perception/fused_fragments) rather than ground truth,
                    # up to the point of grasp. Grasp/carry/drop/return
                    # reuse the same ground-truth carry mechanics as the
                    # state machine (see the pbvs_phase comment in
                    # __init__) -- only the reach-and-track portion is
                    # genuinely camera-closed-loop, which is the actual O4
                    # claim.
                    ee_pos, _ = self.env.get_ee_pose()

                    # Carry/return stall guard -- see __init__ comment.
                    # Checked before the phase dispatch below so a forced
                    # recovery takes effect the same tick it fires.
                    if self.pbvs_phase in ('carry', 'return'):
                        stalled_for = time.time() - self.pbvs_phase_entered_wall
                        if stalled_for > self.pbvs_phase_timeout_s:
                            self.get_logger().warn(
                                f"PBVS phase '{self.pbvs_phase}' exceeded "
                                f"{self.pbvs_phase_timeout_s:.0f}s without "
                                f"completing -- forcing recovery (likely "
                                f"snagged on scene geometry)")
                            if self.pbvs_phase == 'carry' and self.env.held_body_id is not None:
                                try:
                                    p.resetBaseVelocity(
                                        self.env.held_body_id,
                                        linearVelocity=[0.0, 0.0, 0.0],
                                        angularVelocity=[0.0, 0.0, 0.0])
                                except p.error:
                                    pass
                                self.env.release_grasp()
                                self.get_logger().warn(
                                    "forced release (stall recovery) -- NOT "
                                    "counted as a completed delivery, since "
                                    "it was not confirmed to reach the "
                                    "correct bin")
                            self.pbvs_phase = 'return'
                            self.pbvs_phase_entered_wall = time.time()

                    if self.pbvs_phase == 'track':
                        # Staleness guard: if commands stop arriving
                        # (perception pipeline stalled, no fragment in
                        # view, node died), zero the velocity instead of
                        # coasting indefinitely on the last one. A stale
                        # velocity command is worse than no command -- it
                        # drives the arm somewhere nothing asked it to go.
                        if time.time() - self.last_cmd_wall > 0.5:
                            if np.any(self.cmd != 0.0):
                                self.get_logger().warn(
                                    'no PBVS command for >0.5s -- holding position')
                            self.cmd[:] = 0.0
                        self.env.apply_cartesian_velocity(self.cmd[:3], self.cmd[3:])

                        # Grasp check: has the gripper actually reached a
                        # real, live fragment? (Ground truth here, same as
                        # PickPlaceController's own grasp_fragment() call
                        # site -- the CAMERA signal is what steered the
                        # arm to this point, not what confirms contact.)
                        if self.env.held_body_id is None:
                            for body_id in list(self.env._fragment_ids.keys()):
                                try:
                                    frag_pos, _ = p.getBasePositionAndOrientation(body_id)
                                except p.error:
                                    continue
                                if np.linalg.norm(np.array(frag_pos) - ee_pos) < self.pbvs_grasp_radius:
                                    if self.env.grasp_fragment(body_id):
                                        self.env._fragment_ids.pop(body_id, None)
                                        meta = self.env.body_metadata.get(body_id, {})
                                        decision = route_fragment(
                                            meta.get('tyre_type', ''), 1.0,
                                            meta.get('surface_class'))
                                        drop = self.env.route_drop_position(decision.route)
                                        if drop is None:
                                            drop = self.env.route_drop_position(DEFAULT_ROUTE)
                                        self.pbvs_drop_pos = np.asarray(drop, dtype=float)
                                        self.pbvs_nav.reset()
                                        self.pbvs_phase = 'carry'
                                        self.pbvs_phase_entered_wall = time.time()
                                        self.get_logger().info(
                                            f"PBVS grasp id={body_id} "
                                            f"tyre={meta.get('tyre_type', 'unknown')} "
                                            f"route={decision.route}")
                                    break

                    elif self.pbvs_phase == 'carry':
                        # Speed raised from 0.30 to 0.55 m/s (carry) and
                        # return, based on live diagnostic evidence: across
                        # 138 on_fragments() samples in one run, reachable=0
                        # in 55% of them (nothing eligible was within the
                        # arm's reach window at all -- inherent to belt
                        # transit time vs. the reachable zone's width, not a
                        # detection failure: fragments=0 never occurred, and
                        # eligible=0 only 1% of the time). With the
                        # reachable window this narrow, every second spent
                        # in 'carry'/'return' is a second the arm cannot
                        # catch whatever briefly becomes reachable, so
                        # cutting that busy time is the most direct lever
                        # available without touching the reachability
                        # window itself (which was verified-safe at its
                        # current bounds and not re-verified at wider ones).
                        waypoint, arrived = self.pbvs_nav.step(ee_pos, self.pbvs_drop_pos)
                        self.env.command_cartesian_target(waypoint, 0.55)
                        if arrived:
                            # Confirmed live: fragments were launching out of the
                            # bin instead of dropping in. release_grasp() only
                            # removes the fixed constraint -- it does not touch
                            # velocity -- so a fragment released while still
                            # being carried at up to 0.30 m/s keeps that
                            # momentum once free and can fly clear of the
                            # bin's small footprint (inner half-width 0.09m).
                            # Zero the body's velocity in the same step, before
                            # the constraint is removed, so the only force
                            # acting on it once released is gravity -- a clean
                            # vertical drop, matching the state machine's own
                            # "release well above the open top, let it fall"
                            # design intent.
                            if self.env.held_body_id is not None:
                                try:
                                    p.resetBaseVelocity(
                                        self.env.held_body_id,
                                        linearVelocity=[0.0, 0.0, 0.0],
                                        angularVelocity=[0.0, 0.0, 0.0])
                                except p.error:
                                    pass
                            self.env.release_grasp()
                            self.pbvs_completed += 1
                            self.get_logger().info(
                                f"PBVS DROP COMPLETE total={self.pbvs_completed}")
                            self.pbvs_phase = 'return'
                            self.pbvs_phase_entered_wall = time.time()

                    elif self.pbvs_phase == 'return':
                        self.env.command_cartesian_target(self.pbvs_home_pos, 0.55)
                        if float(np.linalg.norm(ee_pos - self.pbvs_home_pos)) < 0.05:
                            self.pbvs_phase = 'track'
                            self.pbvs_phase_entered_wall = time.time()
                else:
                    state = self.controller.step(self.env)
                    if state != self.last_state:
                        self._last_progress_wall = time.time()
                        self.get_logger().info(f"state {self.last_state.name} -> {state.name}")
                        if state == PickState.INTERCEPT_AND_PICK and self.controller.target_body_id is not None:
                            bid = self.controller.target_body_id
                            meta = self.env.body_metadata.get(bid, {})
                            self.get_logger().info(
                                f"target id={bid} tyre={meta.get('tyre_type','unknown')} "
                                f"surface={meta.get('surface_class','unknown')} route={self.controller.target_route}")
                        if self.controller.completed_drops > 0 and state == PickState.RESET_TO_BELT:
                            self.get_logger().info(
                                f"DROP COMPLETE total={self.controller.completed_drops}")
                        self.last_state = state
                self.env.step()

            rgb, depth = self.env.get_rgbd()
            self._publish_img(rgb, depth)
            self._publish_pose()
            self._publish_spectra()

            # AUTO-SHUTDOWN: once every spawned fragment has been resolved
            # (delivered or rejected), the run is genuinely finished --
            # spin forever waiting for nothing was the exact complaint
            # ("doesn't run to completion, forces me to Ctrl+C"). Now
            # grasp/carry/drop exists in pbvs mode too (see pbvs_phase),
            # so it has the same completion signal as state_machine mode,
            # just counted through pbvs_completed instead of
            # controller.completed_drops.
            now_wall = time.time()
            delivered = (self.controller.completed_drops
                         if self.control_mode == 'state_machine' else self.pbvs_completed)
            resolved_naturally = (
                self.total_spawned >= 40
                and (delivered + self.env.reject_count) >= self.total_spawned)
            # Backstop: confirmed live that the accounting above can wait
            # forever if a single fragment gets stranded -- missed the
            # pick window (so it's never delivered) but not yet at the
            # belt's end either (so it's never counted rejected). 90s of
            # zero forward progress (no spawn, no state transition) means
            # nothing further is going to happen regardless of the count.
            # In pbvs mode "progress" is a completed drop, since there's
            # no state-machine transition to watch.
            if self.control_mode == 'pbvs' and delivered > self._last_pbvs_completed:
                self._last_progress_wall = now_wall
                self._last_pbvs_completed = delivered
            stalled = (self.total_spawned > 0
                       and (now_wall - self._last_progress_wall) > 90.0)
            if (resolved_naturally or stalled) and not self._shutdown_logged:
                self._shutdown_logged = True
                reason = ('all fragments resolved' if resolved_naturally
                          else 'no progress for 90s -- a fragment likely got stranded')

                # PHYSICAL PLACEMENT AUDIT -- ground truth for whether
                # "delivered" fragments (counted at the release decision
                # in the 'carry' phase above) actually came to rest in
                # the correct bin, or bounced/rolled out of it. Mirrors
                # sim/env.py's pick_and_sort_demo() audit, which only ran
                # in the standalone demo, never in this live pbvs path --
                # added specifically to test whether bin size/geometry is
                # a real contributor to the reported delivery rate, rather
                # than assuming either way. If audit_correct is close to
                # `delivered` below, bin geometry is not the issue; if
                # it's meaningfully lower, some delivered fragments are
                # genuinely leaving their bins after release.
                bin_half_x = GOOD_BIN_INNER_HALF_X + GOOD_BIN_WALL_THICK
                bin_half_y = GOOD_BIN_INNER_HALF_Y + GOOD_BIN_WALL_THICK
                audit_correct = audit_wrong = audit_unplaced = 0
                for body_id, meta in list(self.env.body_metadata.items()):
                    try:
                        pos, _ = p.getBasePositionAndOrientation(body_id)
                    except p.error:
                        continue
                    expected_route = TYRE_ROUTE.get(meta.get("tyre_type"))
                    if expected_route is None:
                        continue
                    landed_in = None
                    for route, (bx, by, _bz) in ROUTE_DROP_POSITIONS.items():
                        if abs(pos[0] - bx) <= bin_half_x and abs(pos[1] - by) <= bin_half_y:
                            landed_in = route
                            break
                    if landed_in is None:
                        audit_unplaced += 1
                    elif landed_in == expected_route:
                        audit_correct += 1
                    else:
                        audit_wrong += 1
                self.get_logger().info(
                    f"PHYSICAL PLACEMENT AUDIT: correct={audit_correct} "
                    f"wrong_bin={audit_wrong} unplaced_or_bounced={audit_unplaced} "
                    f"(ground-truth resting position at shutdown, independent "
                    f"of the release-based delivered count below)")

                self.get_logger().info(
                    f"RUN COMPLETE ({reason}): spawned={self.total_spawned} "
                    f"delivered={delivered} "
                    f"rejected={self.env.reject_count} -- shutting down "
                    f"(no Ctrl+C needed)")
                self.timer.cancel()
                rclpy.shutdown()

        def _maybe_grasp(self):
            return

    return SimulationBridge

def main():
    import rclpy
    rclpy.init(); node = build_node()(); rclpy.spin(node); node.destroy_node()
    # tick()'s completion check may have already called rclpy.shutdown()
    # itself to make spin() return -- calling it a second time here would
    # raise (context already invalid), so only call it if still needed.
    if rclpy.ok():
        rclpy.shutdown()
