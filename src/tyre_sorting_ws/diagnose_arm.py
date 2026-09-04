"""Isolated diagnostic for TyreSortingEnv.apply_cartesian_velocity --
nothing else running (no belt, no spawning, no seek logic) to get the
cleanest possible signal on why the arm didn't move in the pick-and-sort
demo. Run from the tyre_sorting_ws directory:

    PYTHONPATH=src python3 diagnose_arm.py

Prints joint structure (to check for a fixed-joint indexing mismatch in
apply_cartesian_velocity), then commands a steady 0.1 m/s sideways
velocity for 3 seconds and prints end-effector position every 0.25s.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
import pybullet as p
from tyre_sorting.sim.env import TyreSortingEnv

env = TyreSortingEnv(gui=True)

print("\n=== joint structure ===")
n = p.getNumJoints(env.arm_id)
print(f"n_joints = {n}")
fixed_count = 0
for j in range(n):
    info = p.getJointInfo(env.arm_id, j)
    jtype = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED", 2: "SPHERICAL", 3: "PLANAR"}.get(info[2], str(info[2]))
    if info[2] == p.JOINT_FIXED:
        fixed_count += 1
    print(f"  joint {j}: name={info[1].decode()}, type={jtype}")
print(f"fixed joints: {fixed_count} / {n}")

print("\n=== IK solution length check ===")
pos, quat = env.get_ee_pose()
target = pos + np.array([0.05, 0.0, 0.0])
joints = p.calculateInverseKinematics(env.arm_id, env.n_joints - 1, target.tolist(), quat.tolist(),
                                       maxNumIterations=50, residualThreshold=1e-4)
print(f"len(IK solution) = {len(joints)}  vs  n_joints = {n}")
print(f"IK solution: {[round(j_, 3) for j_ in joints]}")
print(f"current joint positions: {[round(p.getJointState(env.arm_id, j)[0], 3) for j in range(n)]}")

print("\n=== commanding steady velocity, watching ee_pos ===")
start_pos, _ = env.get_ee_pose()
print(f"t=0.00s  ee_pos={start_pos.round(4).tolist()}")
for i in range(720):  # 3s at 240Hz
    env.apply_cartesian_velocity(np.array([0.1, 0.0, 0.0]), np.zeros(3))
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
    if (i + 1) % 60 == 0:  # every 0.25s
        ee_pos, _ = env.get_ee_pose()
        moved = np.linalg.norm(ee_pos - start_pos)
        print(f"t={(i+1)/240:.2f}s  ee_pos={ee_pos.round(4).tolist()}  moved_from_start={moved:.4f}m")

final_pos, _ = env.get_ee_pose()
print(f"\nTotal displacement after 3s @ 0.1m/s commanded (expected ~0.3m if tracking well): "
      f"{np.linalg.norm(final_pos - start_pos):.4f}m")
env.close()
