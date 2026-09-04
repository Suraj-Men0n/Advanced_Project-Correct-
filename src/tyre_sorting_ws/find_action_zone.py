"""Sweep along the belt centerline at pick height to find the x-range
that's ACTUALLY comfortably reachable -- the Action Zone was picked
without this check, and INTERCEPT_AND_PICK's 'always chase the furthest
downstream fragment' rule specifically biases toward the zone's far edge,
which self_check's RANDOM point (x=0.219, inside the current zone) already
showed is at/past a joint limit.

    PYTHONPATH=src python3 find_action_zone.py

Takes about 10-15 seconds, headless.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pybullet as p
from tyre_sorting.sim.env import TyreSortingEnv

env = TyreSortingEnv(gui=False)

pick_z = 0.10  # matches INTERCEPT_AND_PICK's grasp height (fragment top + ~0.03)
results = []
for x in np.arange(-0.55, 0.61, 0.05):
    target = np.array([x, 0.0, pick_z])
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
    results.append((x, error, worst))

print(f"{'x':>7} {'error(m)':>10} {'joint_load':>11}  status")
comfortable_xs = []
for x, error, worst in results:
    ok = error < 0.02 and worst < 0.85
    if ok:
        comfortable_xs.append(x)
    print(f"{x:7.2f} {error:10.4f} {worst:11.2f}  {'OK' if ok else 'AVOID'}")

if comfortable_xs:
    print(f"\nRECOMMENDED Action Zone: x in [{min(comfortable_xs):.2f}, {max(comfortable_xs):.2f}]")
else:
    print("\nNo comfortable x found on the centerline at this height -- widen the sweep or lower pick_z.")
env.close()
