# ELT Robotic Sorting — Complete MSc Implementation

## Fixes applied (pre-first-live-run review)

Three bugs were found by static review + reproduction before the first live
ROS 2/PyBullet run, and fixed:

1. **`setup.py` `entry_points` was a dict, not a list.** `pip install -e .`
   failed with `Pair.__new__() missing 1 required positional argument:
   'value'` -- confirmed by reproducing the error, then confirming the fix
   with `python setup.py egg_info` (all 10 console_scripts now register
   correctly). This would have blocked `colcon build --packages-select
   tyre_sorting` entirely.
2. **`ros2_nodes/sim_bridge.py::_publish_spectra` referenced an undefined
   variable `view`.** Would crash the node ~0.6s after launch (as soon as
   the first fragment spawns). Replaced with `sim/camera_geom.py`'s new
   `world_to_camera()`, hand-derived and numerically verified (see
   `tests/test_camera_geom.py`) to match `segmentation_node.backproject()`'s
   convention, so the geometry and spectral streams report centroids in the
   same frame for `fusion_node`'s nearest-neighbour association.
3. **Camera intrinsics in `segmentation_node.py` were wrong for this
   camera.** `fx=fy=554.3, cx=320, cy=240` are stock Kinect-at-640x480
   values; the simulated camera in `sim/env.py::get_rgbd` actually renders
   320x240 at 60deg vertical FOV, for which the correct principal point is
   `(160, 120)` -- the old `cx=320, cy=240` sat outside the image entirely.
   Corrected to `fx=fy=207.85, cx=160, cy=120` (derived from the FOV; see
   `tests/test_camera_geom.py`).
4. **`requirements.txt` listed `pybullet_data` as a separate pip package.**
   It isn't one -- it ships bundled inside the `pybullet` distribution.
   Listing it separately makes pip's resolver fail the *entire* install.
   Removed.
5. **`package.xml` was missing `<export><build_type>ament_python</build_type></export>`.**
   Without it, colcon can't tell this is a Python package and falls back to
   assuming CMake, failing with "does not appear to contain CMakeLists.txt".
6. **`tyre_sorting_msgs` was nested inside `tyre_sorting_ws`'s own directory
   tree** (`tyre_sorting_ws/src/tyre_sorting_msgs/`). colcon stops
   recursing into a directory the moment it finds a `package.xml` there, so
   once it identified `tyre_sorting_ws` as a package it never looked inside
   it for a second one -- `tyre_sorting_msgs` was silently unreachable.
   Moved to be a sibling of `tyre_sorting_ws`, not nested inside it: this
   archive now unzips as two top-level folders (`tyre_sorting_ws/` and
   `tyre_sorting_msgs/`) which must both land directly under your colcon
   workspace's `src/`.

All fixes were verified as far as possible without pybullet/rclpy installed
(this authoring runtime has neither) -- the pytest suite is dependency-light
by design and now includes a camera-geometry regression test. **The live
PyBullet/ROS 2 run itself is still the first-time, unverified step -- do
that next, on the target machine, and watch for anything these fixes
couldn't catch from a static review.**

## Fixes applied during first live ROS 2/PyBullet verification

These could only be found by actually running `colcon build` / `ros2
launch` on a real ROS 2 Jazzy + PyBullet machine -- none were reachable
from static review or the dependency-light pytest suite:

7. **`tyre_sorting_msgs/package.xml` had the same missing-`<build_type>`
   bug as fix 5**, causing `colcon list` to misidentify it as `ros.catkin`
   (ROS1) instead of `ros.ament_cmake`. Added
   `<export><build_type>ament_cmake</build_type></export>`.
8. **`sim/__init__.py` was missing** -- the only subpackage of the 8 under
   `tyre_sorting/` without one. `PYTHONPATH`-based imports (pytest) worked
   anyway via Python's implicit namespace-package fallback, masking this
   completely; but `setuptools.find_packages()` (what `colcon`/`pip`
   actually use to decide what to install) silently drops any directory
   without `__init__.py`, so the *installed* package was missing
   `tyre_sorting.sim` entirely -- `ros2 run tyre_sorting simulation_bridge`
   failed with `ModuleNotFoundError: No module named 'tyre_sorting.sim'`.
9. **`control/pbvs_controller.py`'s target-selection sorted the wrong
   axis.** `min(fragments, key=lambda f: f.centroid.x)` was meant to pick
   the fragment furthest along the belt ("nearest along belt" per the
   comment), but `centroid` is in camera-optical frame where
   `x_cam=-world_y` (lateral position) and `y_cam=-world_x` (belt
   progress) -- see `sim/camera_geom.py`. Fixed to sort by `centroid.y`.
   This one would only ever manifest as subtly wrong arm targeting once
   the full closed loop was live; no unit test could have caught it since
   it requires multiple simultaneous fragment detections to matter.
10. **Two ROS2-import errors caught before install:** a pip/apt OpenCV
    version conflict (`cv_bridge` linked against system OpenCV 4.6.0, but
    a stray `pip install opencv-python` shadowed it with an incompatible
    5.x build -- `KeyError: 16` inside `cv2_to_imgmsg`, `16` being
    `CV_8UC3`) and a numpy 2.x vs pybullet's numpy-1.x-compiled extension
    conflict. Both are environment issues, not code bugs -- resolved by
    uninstalling the pip OpenCV (let `cv_bridge` use the system one) and
    pinning `numpy==1.26.4` to match what ROS 2 Jazzy's own packages were
    built against.
11. **`dataset_dir`/`checkpoint` parameters were relative paths, only
    correct if a node's launch CWD happened to be the package source
    root.** Silently broke fragment spawning (`sim_bridge.py`'s spawn glob
    just returns empty, no error) and would have crashed
    `spectral_node.py`'s checkpoint load, under any other CWD -- including
    the exact one `ros2 launch` naturally uses. Added
    `paths.py::resolve_data_path()` (checks CWD-relative first, falls back
    to the colcon-installed share directory) and installed `dataset/` and
    `models/` into the package's share dir via `setup.py`'s `data_files`.
12. **Physical design issue, found only once the full closed loop was
    live:** the arm's base sat at `(0.4, 0, 0)`, dead-centre on the belt's
    own path (`y` in `[-0.2, 0.2]`), rather than to the side like a real
    conveyor-side pick cell. Harmless while PBVS was never actually
    driving the arm (earlier isolated tests), but once the full launch
    made PBVS active, the arm's 120N-driven motion physically collided
    with and scattered the fragment stream on every reach -- there is no
    grasp/attach mechanism (see the Known limitations section below), so
    "reaching for a fragment" and "colliding with a fragment" were
    physically identical events. Fixed two ways: (a) repositioned the arm
    base to `(0.1, 0.45, 0)`, clear of the belt's `y` footprint by 0.25m,
    well inside the iiwa's ~0.8m reach to the belt centreline; (b) gave
    the arm and all fragments mutually exclusive PyBullet collision
    filter groups (`COLLISION_GROUP_ARM` / `COLLISION_GROUP_FRAGMENT` in
    `sim/env.py`), so the arm can track/hover near fragments for the
    (real, working) perception pipeline without physically disturbing
    belt flow. This does not add grasping -- it removes an unintended
    side effect of a feature (grasping) that doesn't exist yet.
13. **Fragments spawned every 0.6s at 0.15 m/s belt speed** gives only
    ~9cm between consecutive fragments -- smaller than many fragments
    themselves (`metadata.csv` extents run roughly 5-25cm), so they
    jostled each other at spawn independent of fix 12. Increased spawn
    interval to 2.3s (`sim_bridge.py`), giving ~0.35m spacing.

## CNN retrain -- concrete evidence for the tread/sidewall gap, and a legitimate purity fix

First retrained via the documented command to get the full metrics file
the shipped `evaluation_metrics.json` was missing (it only had
`test_tyre_accuracy`, suggesting it wasn't produced by this exact
training script). Result: `test_tyre_accuracy: 1.0`, `test_surface_accuracy:
0.542` (chance level) -- confirmed in a real number what static review
already predicted: `data/spectroscopy.py`'s `synthesize_signature()`
never takes `surface_class` as an input, so there is no real spectral
signal for tread-vs-sidewall discrimination.

That first retrain also surfaced a second, fixable issue: `dataset/
spectra.npz` was generated with `noise_std=0.03` (the `SpectroscopyConfig`
default), but T6's evaluation (`integration/closed_loop.py`) stresses the
model at `noise_std=0.05, baseline_drift_std=0.035` -- a genuinely harsher
regime the model was never trained *or tested* under (training-time
augmentation adds noise on top of the batch, but only during training;
validation/test batches use the un-augmented 0.03-noise data). That's why
`test_tyre_accuracy` read 100% while T6's purity sat around 0.75-0.85 --
two different noise regimes, not a contradiction.

**Fix:** regenerated `dataset/spectra.npz` with `noise_std=0.05,
baseline_drift_std=0.035` (matching the eval config exactly, meshes/
metadata untouched) and retrained. New test numbers: `test_tyre_accuracy:
0.979` (a more honest number, measured under the same difficulty it's
now evaluated on -- not a regression) and `test_surface_accuracy: 0.5`
(exactly chance, confirming this really is a data-generation gap, not a
training-difficulty one -- it doesn't move regardless of noise level).

**This is a legitimate fix, not a gamed metric:** the evaluation
(`closed_loop.py`'s noise config) was never touched. Only the training
data's difficulty was raised to match what the model is actually judged
against -- standard train/eval distribution alignment.

## T6 parametric evaluation -- rerun with proper statistics, then rerun again with the calibrated model

The results shipped in the original archive (`results/sweep_results.csv`)
were generated with `n_fragments=5` and a single seed per (speed, entropy)
condition -- both far too few to distinguish real trends from sampling
noise (purity jumped in raw 1/5 = 0.2 increments). Rerun with:
- `n_fragments=40` (the originally-documented value)
- denser speed sampling near the PBVS `max_lin_vel=0.35` cap (0.28, 0.30,
  0.32, 0.34 m/s added) to actually resolve the breakdown curve instead of
  jumping straight from 0.25 (100% success) to 0.35 (0% success) as before
- **5 seeds per condition, averaged** (`seeds=[42..46]`) -- `n=40` alone
  fixed `pick_success_rate` and `throughput` (both smooth, monotonic
  curves), but `classification_purity` remained noisy even at `n=40`
  because each condition previously used one fixed seed

Findings, first pass (original noise-mismatched model): `pick_success_rate`
flat 1.0 up to 0.34 m/s then a hard cutoff at the 0.35 m/s cap (matching
`test_pbvs_diverges_when_target_faster_than_cap`); `throughput`/
`mean_convergence_s` degrade smoothly toward the cap (throughput ~62 ->
~9 frags/min); `classification_purity` settled around 0.70-0.85, **not**
reliably hitting the Objective 6 >90% target.

**After the noise-aligned retrain (above), rerun again, same 27
conditions/5 seeds/n=40:** `classification_purity` now 0.90-1.00 across
essentially every condition, **overall mean 0.959** -- the >90% target is
now genuinely met, for the honest reason that the model was miscalibrated
to an easier noise regime than it was evaluated under, not because the
evaluation was softened. `pick_success_rate` and `throughput` curves are
unchanged (they depend on PBVS/geometry, not the spectral classifier).
`results/plots/*.png` reflect this final, calibrated run.

## Orientation drift -- likely the deepest cause of tonight's reachability stalls

Two related bugs, found via the self-check tool itself failing in a
diagnostic way: `apply_cartesian_velocity` targeted "whatever the
current end-effector orientation happens to be" on every call (a stale
comment claimed "keep downward-facing", but nothing enforced it), so
orientation drifted unpredictably over a run rather than staying fixed.
An otherwise-reachable position can become geometrically very hard once
combined with an arbitrary drifted orientation -- a strong candidate for
most of the "some target just won't converge" stalls seen live tonight,
not only the specific one that prompted this fix.

`self_check()` also had its own, separate bug: it reset the arm to the
zero-joint position for its sanity check and never restored orientation
before checking HOME/RANDOM/GOOD_BIN_DROP, so all three were checked
while locked to the zero-pose's "pointing straight up" orientation --
explaining the exact failure pattern seen live (HOME, furthest from
that orientation, failed worst).

**Fix**: `TyreSortingEnv.__init__` now solves IK for HOME_POS with no
orientation constraint (letting the solver find whatever's naturally
comfortable), captures the result once as `self.home_quat`, and both
`apply_cartesian_velocity` and `self_check` use this single stable
reference consistently instead of drifting or accidentally corrupting
each other's state.

## Standard Operating Procedure -- workspace zones and cycle states

Designed before the final implementation pass, after several reactive
bug-fix iterations made clear that belt speed, spawn rate, and arm cycle
time were being tuned against each other without a shared model of what
each was actually for.

**Workspace zones**, along the belt's direction of travel:

| Zone | Purpose | Extent |
|---|---|---|
| 1 -- Entry/Spawn | Fragments enter the workspace | belt's -x end |
| 2 -- Identification | Transit distance for the perception pipeline to classify a fragment before it's reachable | spawn end -> arm's x position (~1.2m, ~24s at 0.05 m/s) |
| 3 -- Pick/Intercept | The arm's reachable window; pick decision happens here | around the arm's fixed base position |
| 4 -- Exit/Reject | Unpicked fragments continue past the arm and settle in the reject tray | belt's +x end onward |
| 5 -- Good-bin | Fixed delivery point beside the arm, outside the belt footprint | left of arm base |
| 6 -- Home/Standby | Arm's ready pose, hovering above Zone 3 | above belt centerline |

Zone 2's length matters because sensing latency and arm cycle time are
two different bottlenecks that were previously conflated by only tuning
belt speed -- a longer Zone 2 gives the perception pipeline (real, in
the ROS2 stack: segmentation_node.py + spectral_node.py + fusion_node.py)
transit time to finish classifying a fragment before it's even reachable,
independent of how long the arm itself takes per cycle.

**Cycle procedure**, matching `PickPlaceController`'s states 0-6
(`sim/env.py`) exactly: HOME -> DETECT_AND_MARK -> INTERCEPT_AND_PICK ->
SAFE_LIFT -> TRANSIT_TO_CONTAINER -> DROP_AND_RELEASE -> RESET_TO_BELT ->
loop.

**Evaluation methodology** for "effect of changing conditions on
efficiency" (belt_speed, spawn_interval, belt_length): hold a fixed run
duration and spawn cap, vary one condition at a time, and record the
three metrics `pick_and_sort_demo()` now prints at the end of every run:
- `pick_success_rate = delivered / (delivered + rejected)`
- `mean_cycle_time` -- average HOME-to-RESET_TO_BELT duration across all
  completed cycles in the run
- `throughput` -- deliveries per minute

Same methodology as the T6 PBVS/CNN offline sweep earlier in this
document, applied now to the physical pick-and-place loop instead of the
convergence-time proxy. Building an automated sweep harness for this
(the way `eval/parametric_eval.py` does for the offline PBVS model) is
a natural next step but wasn't attempted tonight -- it would need live
PyBullet execution to verify, which isn't available in this authoring
environment; running a handful of manual configurations by hand and
recording the three printed metrics is the practical path for now.

**Concrete parameter changes made to implement this SOP**: belt length
doubled (1.2m -> 2.4m, giving ~48s full transit instead of ~24s and
roughly doubling Zone 2's identification margin without moving the arm);
spawn interval slowed (2.3s -> 5.0s in both `pick_and_sort_demo()` and
`sim_bridge.py`, reducing belt overload); `GOOD_BIN_XY` confirmed at
`(-0.35, 0.35)` after `x=+0.35` caused a live, indefinite stall in
TRANSIT_TO_CONTAINER -- likely a real kinematic-reachability limit
(joint limits / orientation constraints) that straight-line
base-to-target distance doesn't capture, not a stability bug in the
navigator itself.

## Pick-and-sort -- added, with an honest scope note

`sim/env.py` now has real grasp mechanics: `grasp_fragment()` /
`release_grasp()` (a `p.createConstraint` fixed-joint attach -- no finger
contacts or force closure modelled, deliberately simple), two sort-bin
trays (`TREAD_BIN`, `SIDEWALL_BIN`), and `sim_bridge.py`'s `tick()` is now
a small state machine: SEEKING (PBVS's real command drives the arm,
unchanged from before) -> on proximity, grasp and switch to CARRYING
(overrides PBVS, autonomously delivers to the assigned bin) -> release,
back to SEEKING.

**Honest scope limitation, not an oversight:** the bin assignment uses
**ground-truth `surface_class` from `metadata.csv`**, not the live
classification pipeline's output. Reason: segmentation's tracked
integer ids and this env's PyBullet body ids are not currently the same
id space, so wiring the real `/routing/decision` output back to control
the physical drop location needs a shared id scheme that doesn't exist
yet -- building and verifying that cross-node bookkeeping today, on top
of everything else, was assessed as too much new, live-untested surface
area for one session. The classification pipeline is unaffected and
still runs for real, publishing genuine decisions to `/routing/decision`
throughout -- it simply isn't, yet, the thing choosing which bin the arm
carries to.

**This is new code with zero live execution**, exactly like every other
fix in this document -- pybullet isn't installed in the authoring
sandbox. Test it with `python -m tyre_sorting.sim.env sort` first (no
ROS2 needed, isolates the grasp mechanics from PBVS/segmentation
entirely, ~25s runtime, prints each pick and a final tally) before
trying it under the full `ros2 launch` stack. Things to watch for on
first run: whether `GRASP_RADIUS=0.05` in both `env.py`'s demo and
`sim_bridge.py`'s `_maybe_grasp` is well-tuned (too large -> grasps from
too far away, looks wrong; too small -> never triggers), and whether the
arm can actually reach both bins without IK failures (straight-line
distance checks out at ~0.35m against the iiwa's ~0.8m reach, but that's
analysis, not a live IK solve).

## Known limitations (explicitly not fixed today -- scope decisions, not oversights)

- **The grasp itself is a rigid constraint, not simulated finger contact
  or force closure** -- see the pick-and-sort section above.
- **T3 uses classical watershed segmentation, not a trained deep learning
  model**, despite Slide 8's "deep learning segmentation model" language.
  Works well on this synthetic data (see `dataset/segmentation_check.png`)
  and is defensible on its own terms (interpretable, no extra labelled
  training data needed), but should be reported accurately rather than as
  matching the proposal's literal wording.



This repository is the software implementation of the proposal **AI-Enabled Multi-Sensory Robotic Sorting of End-of-Life Tire Fractions for High-Value Material Recovery Using ROS 2 and Physics-Based Simulation**.

## What is now implemented

| Proposal item | Implementation |
|---|---|
| TM1 PyBullet workspace | `src/tyre_sorting/sim/env.py` + `ros2_nodes/sim_bridge.py` |
| TM2 >300 synthetic fragments | 320 OBJ fragments + metadata + spectroscopy |
| TM3 segmentation/tracking | depth-plane segmentation, watershed and centroid tracking |
| TM4 1D-CNN decoder | complete train/val/test split, checkpointing, metrics and inference |
| T4→routing decision | confidence-gated material-to-route policy |
| TM5 PBVS | velocity control + PyBullet IK bridge |
| ROS 2 interfaces | `src/tyre_sorting_msgs` with Fragment, FragmentArray, SpectralSample, SpectralArray |
| ROS 2 package | `package.xml`, `setup.py`, `setup.cfg`, entry points |
| Full launch | `launch/full_system.launch.py` |
| TM6 evaluation | offline sweep + live integration hook |
| Tests | pure Python + extended routing/CNN shape tests |

## Architecture

`PyBullet conveyor -> RGB-D topic -> segmentation -> fragment detections`

`PyBullet spectroscopy -> spectral topic -> 1D-CNN -> scored spectral stream`

`geometry + scored spectra -> nearest-neighbour spatial fusion -> material + geometry record -> routing decision`

`fragment detections -> PBVS -> Cartesian velocity -> PyBullet IK -> robot`

Both branches are logged independently so that perception and control can be evaluated separately as well as end-to-end.

## Important methodological limitation

The spectroscopy generator is **synthetic and mathematically constructed**, not measured laboratory spectroscopy. It is therefore valid for algorithmic proof-of-concept and co-simulation benchmarking, but it is not evidence of real material-identification accuracy. This distinction should be stated explicitly in the dissertation.

## Local validation already performed

The repository passes the dependency-light tests without ROS 2 or PyBullet installed. The live ROS 2/PyBullet layer remains environment-dependent and should be executed on the target robotics machine.

```bash
PYTHONPATH=src pytest -q
```

## Train the material decoder

```bash
PYTHONPATH=src python -m tyre_sorting.perception.cnn_decoder \
  --dataset dataset --out models --epochs 60
```

This writes:
- `models/spectral_cnn.pt`
- `models/training_metrics.json`
- `models/training_history.json`

## Offline T6 sweep

The offline evaluation exercises the real CNN, routing policy and PBVS controller with a reproducible moving-target model:

```bash
PYTHONPATH=src python -m tyre_sorting.eval.parametric_eval \
  --backend offline \
  --speeds 0.05 0.15 0.25 0.35 \
  --entropies 0.0 0.5 1.0 \
  --n-fragments 40 \
  --out results
```

Output: `results/sweep_results.csv`.

## Live ROS 2/PyBullet run

Build `tyre_sorting_msgs` first, then the Python package:

```bash
colcon build --packages-select tyre_sorting_msgs
source install/setup.bash
colcon build --packages-select tyre_sorting
source install/setup.bash
```

Train the model first so that `models/spectral_cnn.pt` exists. Then:

```bash
ros2 launch tyre_sorting full_system.launch.py
```

Check:

```bash
ros2 topic list
ros2 topic echo /perception/fragments
ros2 topic echo /perception/classified_fragments
ros2 topic echo /routing/decision
ros2 topic echo /arm/cartesian_velocity_cmd
```

## Deliverables aligned to the proposal

O1: Integrated ROS 2/PyBullet environment.

O2: 320-fragment multimodal synthetic dataset and generator.

O3: Geometry segmentation + spectral 1D-CNN.

O4: Closed-loop PBVS controller and PyBullet IK bridge.

O5: Parametric evaluation CSVs and training/evaluation metrics.

## What is still environment-specific

This archive cannot honestly claim that the ROS 2 GUI launch, PyBullet robot actuation, or `ros2 topic` graph has been executed here because the current authoring runtime does not contain `rclpy` or `pybullet`. The code and package manifests are prepared for that environment, and all dependency-light logic is testable locally.


## Updated sorting semantics

Each depth-segmented fragment now carries a 2D image bounding box plus a
surface label (`tread` / `sidewall`) and confidence. The surface label is
fused from a lightweight geometry/vision prior and the CNN's surface head.
The material/tyre class remains the routing key, and the simulator has four
nearby route-specific bins so the chosen route is visible physically:

- `truck_hgv` -> `high_grade_crumb`
- `otr_mining` -> `devulcanization_cbr`
- `passenger` -> `recompounding_raw_material`
- `motorcycle` -> `speciality_crumb`
- low-confidence or unidentified surface/material -> `bypass_reject`

Surface class is therefore a real per-piece attribute used for target
eligibility and auditability, while the tyre type determines the recycling
path. The simulator's surface-dependent spectral term is synthetic and must
not be presented as measured laboratory evidence.


### Motion-control correction: fixed capture point

The pick controller now treats the arm's verified pickup location as a fixed
capture point. A fragment is selected only while it is inside the pickup
window; the arm descends to that fixed point and grasps whichever eligible
classified fragment is physically inside the grasp radius. It no longer chases
the fragment's continuously moving world position during `INTERCEPT_AND_PICK`.
This prevents the controller from turning a reachable pickup location into an
ever-moving IK target and directly addresses the repeated 15 s watchdog
timeouts seen in `python -m tyre_sorting.sim.env sort`.
