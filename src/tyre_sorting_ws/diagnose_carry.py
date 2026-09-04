"""Isolated diagnostic for the carry-to-bin waypoint logic -- skips the
seek phase entirely (spawn + drive straight to one fragment + grasp) so we
can time JUST the lift-transit-descend-release sequence, without waiting
through another unpredictable-length seek phase first.

    PYTHONPATH=src python3 diagnose_carry.py
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
import pybullet as p
from tyre_sorting.sim.env import TyreSortingEnv, GOOD_BIN_DROP, WaypointNavigator

env = TyreSortingEnv(gui=True)

body_id = env.spawn_fragment("dataset/meshes/frag_0000.obj")
for _ in range(120):
    env.step()
    time.sleep(1.0 / 240.0)
pos, _ = p.getBasePositionAndOrientation(body_id)
print(f"fragment resting at {np.array(pos).round(3)}")

grasp_point = np.array(pos) + np.array([0.0, 0.0, 0.03])
print(f"driving directly to grasp point {grasp_point.round(3)} (skipping seek timing)...")
t0 = time.time()
i = 0
reached = False
for i in range(3600):  # up to 15s
    ee_pos, _ = env.get_ee_pose()
    dist = float(np.linalg.norm(grasp_point - ee_pos))
    if dist < 0.05:
        reached = True
        break
    direction = (grasp_point - ee_pos) / dist
    env.apply_cartesian_velocity(direction * min(0.35, dist * 3.0), np.zeros(3))
    env.step()
    time.sleep(1.0 / 240.0)
if reached:
    print(f"reached grasp point after {time.time() - t0:.1f}s ({i} steps)")
else:
    print(f"FAILED to reach grasp point within 15s ({i} steps, final dist={dist:.3f}m) -- aborting")
    env.close()
    sys.exit(1)

ok = env.grasp_fragment(body_id)
print(f"grasp_fragment succeeded: {ok}")

carry_target = np.array(GOOD_BIN_DROP)
print(f"\ncarrying to {carry_target}...")
t0 = time.time()
delivered = False
nav = WaypointNavigator()
for i in range(3600):  # up to 15s
    ee_pos, _ = env.get_ee_pose()
    waypoint, arrived = nav.step(ee_pos, carry_target)
    if arrived:
        env.release_grasp()
        print(f"\nDELIVERED after {time.time() - t0:.1f}s ({i} steps)")
        delivered = True
        break
    direction = waypoint - ee_pos
    dist = float(np.linalg.norm(direction))
    if dist > 1e-4:
        speed = min(0.30, dist * 2.0)
        env.apply_cartesian_velocity(direction / dist * speed, np.zeros(3))
    env.step()
    time.sleep(1.0 / 240.0)
    if i % 120 == 0:
        print(f"  t={i/240:.1f}s ee_pos={ee_pos.round(3)} waypoint={waypoint.round(3)} dist_to_wp={dist:.3f}")

if not delivered:
    print("\nTIMED OUT after 15s without delivering")

env.close()
