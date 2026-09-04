"""Reachability sweep for the container position -- finds a coordinate to
the LEFT of the arm that the arm can actually reach with real margin
(low position error AND joint load comfortably under 1.0), using the
exact same honest, joint-limit-aware check that just caught the bad one.

    PYTHONPATH=src python3 find_container_position.py

Takes about 10-15 seconds. Prints a ranked list; the top one is your
new GOOD_BIN_XY.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pybullet as p
from tyre_sorting.sim.env import TyreSortingEnv, GOOD_BIN_WALL_HEIGHT

env = TyreSortingEnv(gui=False)  # no GUI needed, just kinematics -- fast

drop_z = GOOD_BIN_WALL_HEIGHT + 0.15
results = []
for x in np.arange(-0.45, 0.06, 0.05):
    for y in np.arange(0.40, 0.71, 0.05):
        target = np.array([x, y, drop_z])
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
        results.append((error, worst, x, y))

results.sort(key=lambda r: (r[0] > 0.02, r[1], r[0]))  # reachable first, then lowest joint load
print(f"{'x':>7} {'y':>7} {'error(m)':>10} {'joint_load':>11}  status")
for error, worst, x, y in results[:8]:
    status = "OK" if error < 0.02 and worst < 0.85 else ("TIGHT" if error < 0.02 else "UNREACHABLE")
    print(f"{x:7.2f} {y:7.2f} {error:10.4f} {worst:11.2f}  {status}")

best = results[0]
print(f"\nRECOMMENDED: GOOD_BIN_XY = ({best[2]:.2f}, {best[3]:.2f})  "
      f"error={best[0]:.4f}m  joint_load={best[1]:.2f}")
env.close()
