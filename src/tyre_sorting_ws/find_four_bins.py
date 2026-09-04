"""Find four reachable drop positions for the four routing bins, biased
toward the START of the belt (more negative x) so the green containers sit
clear of the robot base rather than clipping into it.

    PYTHONPATH=src python3 find_four_bins.py

Headless, ~20s. Prints the best (x, y) pair grid found, ready to paste into
ROUTE_DROP_POSITIONS in src/tyre_sorting/sim/env.py.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pybullet as p
from tyre_sorting.sim.env import TyreSortingEnv

env = TyreSortingEnv(gui=False)
DROP_Z = 0.25
# Bins are physical boxes -- candidate grids must be spaced at least one
# full bin width apart or the containers intersect each other. The earlier
# version of this script only checked IK reachability and happily proposed
# an overlapping grid.
from tyre_sorting.sim.env import GOOD_BIN_INNER_HALF, GOOD_BIN_WALL_THICK
BIN_W = 2.0 * (GOOD_BIN_INNER_HALF + GOOD_BIN_WALL_THICK)
BELT_HALF_WIDTH = 0.2
LOAD_LIMIT = 0.85
ERR_LIMIT = 0.02


def evaluate(x, y, z=DROP_Z):
    target = np.array([x, y, z])
    joints = p.calculateInverseKinematics(
        env.arm_id, env.n_joints - 1, target.tolist(), env.home_quat.tolist(),
        lowerLimits=env.joint_lower, upperLimits=env.joint_upper,
        jointRanges=env.joint_ranges, restPoses=env.rest_poses,
        maxNumIterations=200, residualThreshold=1e-5)
    for j in range(min(env.n_joints, len(joints))):
        p.resetJointState(env.arm_id, j, joints[j])
    achieved, _ = env.get_ee_pose()
    error = float(np.linalg.norm(achieved - target))
    worst = 0.0
    for j in range(env.n_joints):
        lo, hi = env.joint_lower[j], env.joint_upper[j]
        if hi > lo:
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            worst = max(worst, abs(p.getJointState(env.arm_id, j)[0] - mid) / max(half, 1e-9))
    return error, worst


# Search a 2x2 grid: two x columns (both toward the belt start) x two y rows.
# Bias x strongly negative so containers sit well clear of the robot base.
best = None
for x_near in np.arange(-0.62, -0.24, 0.02):
    for x_far in np.arange(x_near - 0.42, x_near - BIN_W + 0.001, 0.02):
        for y_near in np.arange(0.36, 0.61, 0.02):
            if y_near - BIN_W / 2.0 <= BELT_HALF_WIDTH:
                continue  # bin would overlap the belt
            for y_far in np.arange(y_near + BIN_W, y_near + BIN_W + 0.30, 0.02):
                corners = [(x_near, y_near), (x_near, y_far),
                           (x_far, y_near), (x_far, y_far)]
                results = [evaluate(cx, cy) for cx, cy in corners]
                if any(e >= ERR_LIMIT or w >= LOAD_LIMIT for e, w in results):
                    continue
                worst_load = max(w for _, w in results)
                if best is None or worst_load < best[0]:
                    best = (worst_load, corners, results)

if best is None:
    print("No 2x2 grid found where all four corners are comfortable.")
    print("Widen the search ranges or lower LOAD_LIMIT expectations.")
else:
    worst_load, corners, results = best
    print(f"Best 2x2 bin grid -- worst joint_load across all four = {worst_load:.2f}\n")
    names = ["devulcanization_cbr", "high_grade_crumb",
             "recompounding_raw_material", "speciality_crumb"]
    print("ROUTE_DROP_POSITIONS = {")
    for name, (cx, cy), (err, load) in zip(names, corners, results):
        print(f'    "{name}": ({cx:.2f}, {cy:.2f}, {DROP_Z}),'
              f'  # err={err:.4f}m load={load:.2f}')
    print("}")

env.close()
