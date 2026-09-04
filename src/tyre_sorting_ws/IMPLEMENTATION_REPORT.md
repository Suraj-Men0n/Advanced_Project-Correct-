# Tyre Sorting ROS 2 — bounding-box / surface / routing / motion correction

## Verified conclusion

The requested sorting specification is better with one important separation:

- **Bounding box:** every detected fragment gets a pixel bounding box `(xmin, ymin, xmax, ymax)`.
- **Surface class:** `tread` or `sidewall` is a per-fragment attribute from perception.
- **Tyre/material class:** `passenger`, `truck_hgv`, `motorcycle`, `otr_mining`.
- **Final recycling route:** controlled by tyre/material class.
- **Reject:** low-confidence material, unidentified surface, or out-of-spec fragments go to `bypass_reject`.

Surface class is not used to invent a different recycling chemistry route. If tread and sidewall need physically different processing, that needs an explicit route/bin policy; otherwise it should remain a classification/selection/audit attribute.

## What was actually wrong

1. `python -m tyre_sorting.sim.env sort` is a standalone PyBullet controller. It did **not** run the ROS 2 perception/routing chain.
2. `PickPlaceController` selected a fragment near the pickup point, then **chased the fragment's current moving world position**. Because the belt moves +x, that target could leave the verified IK workspace while the arm was descending. This is consistent with the repeated `INTERCEPT_AND_PICK` watchdog timeouts.
3. The `RANDOM` self-check sampled an overly broad workspace. It could produce a near-limit/IK-inconvenient pose even though the real fixed pickup/drop points were valid.
4. Segmentation already computed a bbox internally, but the ROS `Fragment` message did not carry it, so bbox information was lost before fusion/routing.
5. The CNN had a surface classification head, but the synthetic spectroscopy generator ignored `surface_class`. That made the surface head effectively chance-level.
6. `routing_node.py` did not pass `surface_class` into `route_fragment()`.
7. The simulator actuator previously sent all picked fragments to one `GOOD_BIN_DROP`.

## Implemented corrections

### Perception / bounding box
- `Fragment.msg` now carries `bbox_xmin/ymin/xmax/ymax`.
- Segmentation publishes each bbox.
- Segmentation also publishes a lightweight vision/geometry surface prior (`tread` / `sidewall` / `unknown`) with confidence.
- Fusion preserves bbox and combines the vision surface estimate with the spectral surface estimate.

### Spectroscopy / CNN
- `synthesize_signature()` now accepts `surface_class` and adds a documented **synthetic** surface-dependent signal.
- Dataset generation passes `surface_class` into spectroscopy.
- `SpectralInference.predict()` now returns:
  `(tyre_label, tyre_confidence, surface_label, surface_confidence)`.
- The ROS spectral node publishes both tyre and surface predictions.

This remains a co-simulation proxy. It is not measured laboratory spectroscopy and must not be presented as real material-ID evidence.

### Routing
The routing map remains:

- `otr_mining` -> `devulcanization_cbr`
- `truck_hgv` -> `high_grade_crumb`
- `passenger` -> `recompounding_raw_material`
- `motorcycle` -> `speciality_crumb`
- low-confidence / unknown -> `bypass_reject`

A known surface class is now required for a fragment to be considered routable by the policy, and the routing log includes bbox + surface class.

### Robot movement
The key motion correction is:

`HOME -> wait for a classified fragment inside pickup window -> mark target -> descend to the FIXED pickup point -> grasp only when the piece is physically inside the grasp radius -> lift vertically -> transit at safe Z -> descend -> release -> return HOME`

The arm no longer chases a moving fragment's continuously changing x-coordinate during the pick step.

### Simulation bins
The PyBullet environment now has four nearby route-specific recycling bins corresponding to the four tyre routes, plus the belt-exit reject tray.

For the **simulator actuator layer**, the physical drop route is currently obtained from the fragment's simulator metadata. The live ROS routing topic still publishes the real classification/routing decision, but stable cross-node track-ID wiring from `/routing/decision` back into the PyBullet body's route has not been fully connected.

## Validation performed here

- Python compilation: passed.
- Dependency-light test suite: **14 passed**.
- Retrained model using 320 metadata-labelled fragments.
- Held-out test metrics:
  - tyre-type accuracy: **97.9%**
  - surface accuracy: **100%** on this synthetic dataset

PyBullet is not installed in this execution environment, so the actual GUI/physics run was not executed here.

## Recommended validation on the Lenovo machine

From the package source directory:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 -m tyre_sorting.sim.env check
PYTHONPATH=src python3 -m tyre_sorting.sim.env sort
```

For ROS 2:

```bash
colcon build --packages-select tyre_sorting_msgs
source install/setup.bash
colcon build --packages-select tyre_sorting
source install/setup.bash
ros2 launch tyre_sorting full_system.launch.py
```

Then inspect:

```bash
ros2 topic echo /perception/fragments
ros2 topic echo /perception/fused_fragments
ros2 topic echo /routing/decision
ros2 topic echo /arm/cartesian_velocity_cmd
```

## Latest standalone-demo fixes

- The standalone `sort` path now treats every spawned fragment as pickable; classification does not block the mechanical smoke test.
- Once a fragment enters the fixed pickup window it is frozen on the conveyor, so the arm descends to one verified static pick point instead of chasing a moving target.
- HOME no longer trips the 15 s watchdog while simply waiting for the next piece.
- The active green collection bin opening was increased from 0.18 m to 0.44 m (inner half-width 0.22 m) and wall height increased to 0.12 m.
- The standalone RGB camera frame now overlays segmentation bounding boxes and a surface label. The simulated conveyor plane depth is 0.85 m for the 0.9 m top-down camera and 0.05 m belt height; segmentation/launch defaults were aligned to 0.85 m.
- The four route coordinates remain available for route-aware ROS2 operation; the standalone `sort` demo defaults to the common passenger/recompounding drop point so the mechanical cycle is easy to validate first.
