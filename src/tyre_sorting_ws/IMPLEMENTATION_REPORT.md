# Tyre Sorting (ROS 2): Bounding Boxes, Surface Class, Routing and Motion Fixes

## Summary

The sorting spec mostly holds up, but it works better if a few things are kept separate.

Each detected fragment gets a pixel bounding box `(xmin, ymin, xmax, ymax)` and a surface class (`tread` or `sidewall`) from perception. It also gets a tyre/material class: `passenger`, `truck_hgv`, `motorcycle` or `otr_mining`. The recycling route is decided by the tyre/material class only. Anything with low-confidence material, an unidentified surface, or out-of-spec geometry goes to `bypass_reject`.

Surface class doesn't create its own recycling route. If tread and sidewall ever need different physical processing, that should be added as an explicit route/bin policy. Until then, surface class is used for classification, pick selection and the audit log.

## Problems found

The `python -m tyre_sorting.sim.env sort` command runs a standalone PyBullet controller. It never touched the ROS 2 perception or routing nodes, so it wasn't testing the full chain.

`PickPlaceController` picked a fragment near the pickup point and then kept following that fragment's live position. With the belt moving in +x, the target could drift out of the verified IK workspace while the arm was still descending, which explains the repeated `INTERCEPT_AND_PICK` watchdog timeouts.

The `RANDOM` self-check sampled a workspace that was too broad. It sometimes produced poses near the joint limits, even though the actual fixed pickup and drop points were fine.

Segmentation was already computing a bounding box, but the ROS `Fragment` message had no fields for it, so it was dropped before fusion and routing.

The CNN has a surface classification head, but the synthetic spectroscopy generator ignored `surface_class`. The surface head was effectively guessing.

`routing_node.py` wasn't passing `surface_class` into `route_fragment()`.

In the simulator, every picked fragment was dropped into the same `GOOD_BIN_DROP`.

## Changes

### Perception and bounding boxes

`Fragment.msg` now has `bbox_xmin`, `bbox_ymin`, `bbox_xmax` and `bbox_ymax`, and segmentation fills them in. Segmentation also publishes a simple vision/geometry surface estimate (`tread`, `sidewall` or `unknown`) with a confidence value. The fusion node keeps the bounding box and combines the vision surface estimate with the spectral one.

### Spectroscopy and CNN

`synthesize_signature()` now takes `surface_class` and adds a synthetic surface-dependent component to the signal, and dataset generation passes the surface class through. `SpectralInference.predict()` returns `(tyre_label, tyre_confidence, surface_label, surface_confidence)`, and the ROS spectral node publishes both predictions.

Note that this is still a simulation proxy, not measured lab spectroscopy. It shouldn't be presented as evidence that real material identification works.

### Routing

The route map hasn't changed:

| Tyre class    | Route                        |
|---------------|------------------------------|
| `otr_mining`  | `devulcanization_cbr`        |
| `truck_hgv`   | `high_grade_crumb`           |
| `passenger`   | `recompounding_raw_material` |
| `motorcycle`  | `speciality_crumb`           |
| low confidence / unknown | `bypass_reject`   |

A fragment now needs a known surface class to be routable, and the routing log records the bounding box and surface class.

### Arm motion

The pick sequence is now:

1. Start at HOME and wait for a classified fragment to enter the pickup window.
2. Mark it as the target and descend to the fixed pickup point.
3. Grasp only once the fragment is physically within the grasp radius.
4. Lift straight up, move across at a safe Z height, descend over the bin and release.
5. Return to HOME.

The arm no longer follows the fragment's changing x-position during the pick.

### Simulator bins

The PyBullet scene now has four recycling bins next to the belt, one per route, plus the reject tray at the belt exit.

At the moment the simulator picks the drop bin from the fragment's own simulator metadata. The ROS routing topic still publishes the real classification and routing decision, but the link from `/routing/decision` back to the matching PyBullet body (via a stable track ID across nodes) isn't finished yet.

## Testing so far

Python compilation passes and the dependency-light test suite passes (14 tests). The model was retrained on 320 metadata-labelled fragments. On the held-out set it scored 97.9% on tyre type and 100% on surface class, though the surface figure is on synthetic data and shouldn't be read as real-world performance.

PyBullet wasn't installed in the environment these changes were tested in, so the GUI/physics run still needs to be done.

## Checks to run on the Lenovo machine

From the package source directory:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 -m tyre_sorting.sim.env check
PYTHONPATH=src python3 -m tyre_sorting.sim.env sort
```

For the full ROS 2 system:

```bash
colcon build --packages-select tyre_sorting_msgs
source install/setup.bash
colcon build --packages-select tyre_sorting
source install/setup.bash
ros2 launch tyre_sorting full_system.launch.py
```

Then check the topics:

```bash
ros2 topic echo /perception/fragments
ros2 topic echo /perception/fused_fragments
ros2 topic echo /routing/decision
ros2 topic echo /arm/cartesian_velocity_cmd
```

## Latest fixes to the standalone demo

In `sort` mode every spawned fragment is treated as pickable, so classification doesn't get in the way of testing the mechanics.

When a fragment enters the pickup window it's frozen on the belt, so the arm always goes to one verified, stationary pick point.

The arm waiting at HOME for the next piece no longer triggers the 15 s watchdog.

The green collection bin opening is now 0.44 m wide (was 0.18 m, so 0.22 m inner half-width) and the walls are 0.12 m high.

The standalone RGB camera view now draws the segmentation bounding boxes and a surface label on each fragment. The camera sits 0.9 m above the scene and the belt surface is at 0.05 m, so the belt plane depth is 0.85 m. The segmentation and launch defaults have been updated to match.

The four route bin coordinates are still there for running with ROS routing, but the standalone `sort` demo drops everything at the passenger/recompounding bin by default. That keeps the first test focused on getting the pick-and-place cycle working.
