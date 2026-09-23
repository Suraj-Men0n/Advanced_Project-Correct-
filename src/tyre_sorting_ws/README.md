# ELT Robotic Sorting

This repository contains the software implementation of the MSc project:

**AI-Enabled Multi-Sensory Robotic Sorting of End-of-Life Tire Fractions for High-Value Material Recovery Using ROS 2 and Physics-Based Simulation**

The system combines a PyBullet conveyor simulation, RGB-D perception, synthetic spectroscopy, a 1D CNN classifier, ROS 2 nodes, and a PBVS controller for the robot arm.

## Current implementation

The main parts of the proposed system are implemented as follows.

| Proposal item | Implementation |
|---|---|
| TM1: PyBullet workspace | `src/tyre_sorting/sim/env.py` and `ros2_nodes/sim_bridge.py` |
| TM2: Synthetic dataset | 320 OBJ fragments, metadata, and generated spectra |
| TM3: Segmentation and tracking | Depth-plane segmentation, watershed segmentation, and centroid tracking |
| TM4: Material classification | 1D CNN with train/validation/test splits, checkpointing, metrics, and inference |
| T4: Routing decision | Confidence-gated material-to-route policy |
| TM5: PBVS | Cartesian velocity control with PyBullet IK |
| ROS 2 interfaces | `src/tyre_sorting_msgs` with `Fragment`, `FragmentArray`, `SpectralSample`, and `SpectralArray` |
| ROS 2 package | `package.xml`, `setup.py`, `setup.cfg`, and console entry points |
| System launch | `launch/full_system.launch.py` |
| TM6: Evaluation | Offline parameter sweep and live integration hook |
| Tests | Dependency-light unit tests and routing/CNN shape tests |

## System architecture

The main data flow is:

```text
PyBullet conveyor
    -> RGB-D
    -> segmentation and tracking
    -> fragment detections

PyBullet spectroscopy
    -> spectral samples
    -> 1D CNN
    -> classified spectral stream

Fragment geometry + spectral result
    -> nearest-neighbour fusion
    -> material / surface / geometry record
    -> routing decision

Fragment detections
    -> PBVS
    -> Cartesian velocity
    -> PyBullet IK
    -> robot motion
```

The perception and control paths are logged separately so they can also be evaluated independently.

---

# Changes and fixes

The implementation went through several rounds of checking and testing. The main issues found are listed below.

## Package and build fixes

### 1. Fixed `setup.py` entry points

The `entry_points` definition in `setup.py` was initially a dictionary instead of a list. This caused editable installation to fail with:

```text
Pair.__new__() missing 1 required positional argument: 'value'
```

The entry-point definition was changed to the standard list format. `python setup.py egg_info` was then used to confirm that all ten console scripts were registered.

### 2. Added the missing Python package marker

`sim/__init__.py` was missing.

Python could still import the directory during local testing because namespace-package behaviour masked the problem, but `setuptools.find_packages()` does not include such directories. As a result, the installed package did not contain `tyre_sorting.sim`.

Adding `sim/__init__.py` fixed the installation and allowed commands such as:

```bash
ros2 run tyre_sorting simulation_bridge
```

to find the simulation package.

### 3. Corrected ROS 2 package build types

The Python package was missing:

```xml
<export>
  <build_type>ament_python</build_type>
</export>
```

and the message package was missing its corresponding:

```xml
<export>
  <build_type>ament_cmake</build_type>
</export>
```

These were added so that `colcon` identifies the packages correctly.

### 4. Moved `tyre_sorting_msgs` out of the Python package tree

The message package was originally placed inside the `tyre_sorting_ws` package directory:

```text
tyre_sorting_ws/src/tyre_sorting_msgs/
```

Because `tyre_sorting_ws` is itself a ROS package, recursive package discovery stopped at its `package.xml`. The message package was therefore not discovered.

The final layout has both packages as siblings:

```text
src/
├── tyre_sorting_ws/
└── tyre_sorting_msgs/
```

### 5. Removed `pybullet_data` from `requirements.txt`

`pybullet_data` is included with the `pybullet` distribution and is not a separate package that needs to be installed from pip. Keeping it in `requirements.txt` caused dependency resolution to fail.

It was removed.

### 6. Fixed ROS/OpenCV and NumPy environment conflicts

A separate `pip` installation of OpenCV had replaced the system OpenCV version used by `cv_bridge`. This produced errors inside `cv2_to_imgmsg`.

The extra pip OpenCV installation was removed so that `cv_bridge` could use the system version.

A second issue was caused by NumPy 2.x being incompatible with the NumPy version used when PyBullet's extension had been built. The environment was pinned to:

```text
numpy==1.26.4
```

---

## Camera and geometry fixes

### 7. Fixed the broken spectrum projection

`ros2_nodes/sim_bridge.py::_publish_spectra` used an undefined `view` variable. The function was changed to use the shared `world_to_camera()` transformation in:

```text
sim/camera_geom.py
```

The transformation was checked against the convention used by `segmentation_node.backproject()` so that spectral and segmentation centroids are expressed in the same camera frame.

### 8. Corrected the camera intrinsics

The segmentation node was using:

```text
fx = fy = 554.3
cx = 320
cy = 240
```

Those values correspond to a different 640x480 camera model.

The simulated camera is 320x240 with a 60 degree vertical field of view, so the values were changed to:

```text
fx = fy = 207.85
cx = 160
cy = 120
```

A camera-geometry regression test was also added under:

```text
tests/test_camera_geom.py
```

### 9. Corrected PBVS target selection

The target-selection code originally sorted fragments by:

```python
centroid.x
```

The centroid is expressed in the camera optical frame, where the relevant belt direction is represented by the `y` coordinate. The selection was therefore changed to use `centroid.y`.

---

# Simulation changes

## 10. Made dataset and model paths independent of the launch directory

The dataset and model paths were originally relative paths. That worked only when the node happened to be started from the package source directory.

This was especially problematic for:

- fragment spawning in `sim_bridge.py`
- model loading in `spectral_node.py`
- normal `ros2 launch` execution

A new helper was added:

```text
paths.py::resolve_data_path()
```

It checks the current working directory first and then falls back to the installed package share directory.

The dataset and model directories are also installed as package data.

## 11. Moved the robot away from the conveyor

The robot base originally sat at:

```text
(0.4, 0, 0)
```

which put it directly on the conveyor's footprint.

When PBVS control was enabled, the arm could collide with fragments and disturb their motion.

The base was moved to:

```text
(0.1, 0.45, 0)
```

This keeps the robot clear of the belt while remaining within the arm's working range.

The arm and fragments were also put into separate PyBullet collision groups so that the arm can track fragments without pushing them around.

## 12. Reduced the fragment spawn rate

At:

```text
0.6 s spawn interval
0.15 m/s belt speed
```

successive fragments were only about 9 cm apart. Since some fragments are larger than this, they could interfere with each other.

The initial spawn interval was increased to 2.3 s and later to 5.0 s as the full pick-and-sort cycle was tuned.

---

# CNN training and evaluation

## Dataset calibration

The first retraining run produced:

```text
test_tyre_accuracy    = 1.0
test_surface_accuracy = 0.542
```

The surface result was expected because the synthetic spectroscopy generator did not use `surface_class` when generating the spectrum. There was therefore no meaningful spectral information for distinguishing tread from sidewall.

A separate issue was that the training data used:

```text
noise_std = 0.03
```

while the T6 evaluation used:

```text
noise_std = 0.05
baseline_drift_std = 0.035
```

The training and evaluation conditions were therefore different.

The dataset was regenerated with the same noise and drift values used by the evaluation, and the CNN was retrained.

The resulting test values were:

```text
test_tyre_accuracy    = 0.979
test_surface_accuracy = 0.50
```

The unchanged surface result is consistent with the fact that the data generator still does not encode a useful surface-specific spectral signal.

The evaluation settings were not changed.

## T6 parameter sweep

The original sweep used only five fragments and one random seed per condition. That was too small for reliable comparisons.

The sweep was rerun with:

- `n_fragments = 40`
- five seeds per condition: `42` through `46`
- additional belt speeds around the PBVS velocity limit:
  - 0.28 m/s
  - 0.30 m/s
  - 0.32 m/s
  - 0.34 m/s

The final run covered 27 conditions in total.

### Results before retraining

With the earlier model:

- `pick_success_rate` stayed at 1.0 below the PBVS velocity limit and then dropped at the 0.35 m/s limit.
- Throughput decreased as belt speed approached that limit.
- Mean convergence time increased near the limit.
- Classification purity was generally around 0.70-0.85.

### Results after retraining

With the noise-aligned training set:

- `classification_purity` was between 0.90 and 1.00 for almost all conditions.
- The overall mean purity was 0.959.
- The pick-success and throughput curves remained effectively unchanged because they are governed mainly by the PBVS and geometry rather than by the classifier.

The final plots are stored under:

```text
results/plots/
```

---

# Orientation and reachability

The arm controller originally reused the current end-effector orientation on every Cartesian velocity update. This meant that orientation could drift during a run.

That made some otherwise reachable positions difficult to reach.

The self-check routine had a related problem. It reset the robot to the zero-joint position and then checked the test poses without first restoring the intended working orientation.

A stable reference orientation is now calculated from the home pose and reused by both the controller and the self-check.

The reference is stored as:

```python
self.home_quat
```

---

# Pick-and-sort operation

## Workspace zones

The working area is divided into the following zones:

| Zone | Purpose |
|---|---|
| 1 - Entry / Spawn | Fragments enter the conveyor |
| 2 - Identification | Perception and classification take place before the arm reaches the fragment |
| 3 - Pick / Intercept | The arm operates in this area |
| 4 - Exit / Reject | Unpicked fragments continue to the reject area |
| 5 - Good Bin | Sorted material is dropped here |
| 6 - Home / Standby | Robot waits above the pick area |

The belt was later lengthened from 1.2 m to 2.4 m so that the identification section has more time for perception and classification before the fragment reaches the arm.

## Pick-and-sort state machine

The physical loop follows these states:

```text
HOME
  -> DETECT_AND_MARK
  -> INTERCEPT_AND_PICK
  -> SAFE_LIFT
  -> TRANSIT_TO_CONTAINER
  -> DROP_AND_RELEASE
  -> RESET_TO_BELT
  -> HOME
```

The controller was also changed so that the final pickup is made at a fixed capture point.

A fragment is considered for pickup only while it is inside the pickup window. The arm moves to the verified pickup location rather than continuously chasing the fragment's changing world position during the final approach.

This avoids turning a reachable pickup point into a moving target for the IK solver.

## Grasping

`sim/env.py` now includes:

```python
grasp_fragment()
release_grasp()
```

The current grasp is implemented using a PyBullet fixed constraint rather than a detailed finger/contact model.

The simulation includes separate destination trays for:

- tread
- sidewall

The main sorting loop in `sim_bridge.py` works as a simple state machine:

```text
SEEKING
  -> CARRYING
  -> release
  -> SEEKING
```

PBVS remains responsible for the approach to the pickup point. Once a fragment is grasped, the controller takes over for the carry and drop motion.

### Current routing limitation

The live ROS 2 classifier produces surface/material decisions, but the current pick-and-sort demo does not yet use those decisions to select the physical drop bin.

The reason is that the fragment IDs used by segmentation and the PyBullet body IDs are currently different. There is no shared ID scheme connecting a classifier result directly to a specific PyBullet body.

For the current demo, the bin assignment still uses the ground-truth `surface_class` from `metadata.csv`.

This means the perception pipeline and routing decision node are running, but the physical drop location is not yet driven by the live classifier output.

---

# Updated sorting semantics

Each segmented fragment now contains:

- a 2D image bounding box
- a `surface` label
- a confidence value

The supported surface labels are:

```text
tread
sidewall
```

The routing policy currently maps tyre type to recycling route:

| Tyre type | Route |
|---|---|
| `truck_hgv` | `high_grade_crumb` |
| `otr_mining` | `devulcanization_cbr` |
| `passenger` | `recompounding_raw_material` |
| `motorcycle` | `speciality_crumb` |
| low confidence / unidentified | `bypass_reject` |

The surface class is retained as a per-fragment attribute for eligibility and logging. The tyre class remains the main routing key.

The surface-dependent spectroscopy is synthetic and should not be treated as measured laboratory data.

---

# Evaluation methodology

For the physical pick-and-place tests, the main metrics are:

```text
pick_success_rate = delivered / (delivered + rejected)

mean_cycle_time = average HOME-to-RESET_TO_BELT duration

throughput = deliveries per minute
```

The intended evaluation approach is to keep the run duration and fragment limit fixed and change one parameter at a time, such as:

- belt speed
- spawn interval
- belt length

The existing offline sweep in `eval/parametric_eval.py` uses a similar approach for the PBVS model.

A fully automated sweep of the live PyBullet loop would require running the system on a machine with ROS 2 and PyBullet. That was not available in the authoring environment.

---

# Known limitations

The following limitations are still present.

## Synthetic spectroscopy

The spectral signals are generated mathematically. They are not measurements from laboratory equipment.

The spectroscopy and CNN results therefore demonstrate the behaviour of the software pipeline, not real-world material-identification accuracy.

## Segmentation method

The current segmentation stage uses classical watershed segmentation rather than a trained deep-learning segmentation network.

This works on the synthetic dataset, but it does not exactly match the original proposal's wording about a deep-learning segmentation model.

## Simplified grasp model

The grasp is represented by a fixed PyBullet constraint. There is no detailed finger contact model or force-closure simulation.

## Physical routing is not yet connected to the live classifier

The classification node produces live routing decisions, but the current physical demo still uses `surface_class` from the dataset metadata to choose the destination bin because the fragment-ID mapping between ROS 2 perception and PyBullet objects is not yet shared.

---

# Validation status

The dependency-light Python tests can be run without ROS 2 or PyBullet:

```bash
PYTHONPATH=src pytest -q
```

The ROS 2 and PyBullet parts are environment-dependent.

The current archive should therefore not be treated as evidence that the complete ROS 2 GUI launch, robot actuation, and topic graph have been executed in the authoring environment itself.

---

# Running the project

## Train the material classifier

```bash
PYTHONPATH=src python -m tyre_sorting.perception.cnn_decoder \
  --dataset dataset \
  --out models \
  --epochs 60
```

This produces:

```text
models/spectral_cnn.pt
models/training_metrics.json
models/training_history.json
```

## Run the offline T6 sweep

```bash
PYTHONPATH=src python -m tyre_sorting.eval.parametric_eval \
  --backend offline \
  --speeds 0.05 0.15 0.25 0.35 \
  --entropies 0.0 0.5 1.0 \
  --n-fragments 40 \
  --out results
```

The main output is:

```text
results/sweep_results.csv
```

## Build the ROS 2 workspace

Build the message package first:

```bash
colcon build --packages-select tyre_sorting_msgs
source install/setup.bash
```

Then build the main package:

```bash
colcon build --packages-select tyre_sorting
source install/setup.bash
```

Make sure the CNN checkpoint exists:

```text
models/spectral_cnn.pt
```

Then launch the full system:

```bash
ros2 launch tyre_sorting full_system.launch.py
```

Useful topics to inspect:

```bash
ros2 topic list

ros2 topic echo /perception/fragments
ros2 topic echo /perception/classified_fragments
ros2 topic echo /routing/decision
ros2 topic echo /arm/cartesian_velocity_cmd
```

---

# Project deliverables

The implementation currently covers the following proposal outputs:

**O1** - Integrated ROS 2 / PyBullet environment

**O2** - 320-fragment synthetic multimodal dataset and generator

**O3** - Geometry-based segmentation and spectral 1D CNN

**O4** - PBVS controller and PyBullet IK bridge

**O5** - Parameter-sweep results, training metrics, and evaluation outputs
