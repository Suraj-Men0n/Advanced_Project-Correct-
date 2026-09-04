# Custom ROS 2 messages referenced by the nodes

`tyre_sorting_msgs` is a sibling `ament_cmake` package (lives at
`../../tyre_sorting_msgs` relative to this file, i.e. a sibling of
`tyre_sorting_ws` under your colcon workspace's `src/`) -- it must NOT be
nested inside `tyre_sorting_ws`'s own tree, or colcon will never discover it
(see README.md's changelog, bug 6, for what that failure looks like).

Build it first, before `tyre_sorting`, since the perception/control nodes
import from it:
```
colcon build --packages-select tyre_sorting_msgs
source install/setup.bash
colcon build --packages-select tyre_sorting
```

## Current message definitions

**msg/Fragment.msg** -- one geometry-stream detection, after fusion with
its matched spectral prediction
```
int32 id
geometry_msgs/Point centroid       # camera-optical frame, see sim/camera_geom.py
int32 pixel_area
string material_class              # e.g. "otr_mining", "passenger" -- filled in by fusion_node
string surface_class                # "tread" / "sidewall" -- geometry-derived, NOT from spectroscopy
                                     # (see README.md's tread/sidewall note: the spectral CNN has
                                     # no real signal for this axis, test_surface_accuracy ~0.54)
float32 classification_confidence
float32 surface_classification_confidence  # fusion_node's combined tread/sidewall
                                     # confidence -- was missing from this message
                                     # entirely until now, which crashed fusion_node
                                     # on every launch (AttributeError, confirmed live)
float32 track_vx, track_vy, track_vz  # tracker.py's estimated velocity, camera-frame units/s
```

**msg/FragmentArray.msg**
```
std_msgs/Header header
Fragment[] fragments
```

**msg/SpectralSample.msg / SpectralArray.msg** -- raw spectral stream from
`sim_bridge.py`, before classification
```
int32 id
geometry_msgs/Point centroid
float32[] values          # (4, 128) flattened multi-channel spectrum
```

**msg/ScoredSpectralSample.msg / ScoredSpectralArray.msg** -- after
`spectral_node.py`'s CNN inference
```
int32 id
geometry_msgs/Point centroid
string material_class
float32 classification_confidence
```

## Topic wiring (who publishes/subscribes what)

```
segmentation_node --/perception/fragments(FragmentArray)-->        fusion_node, pbvs_controller
sim_bridge        --/sensors/spectroscopy(SpectralArray)-->        spectral_node
spectral_node     --/sensors/scored_spectroscopy(ScoredSpectralArray)--> fusion_node
fusion_node       --/perception/fused_fragments(FragmentArray)-->  routing_node
routing_node      --/routing/decision(std_msgs/String, JSON)-->    (terminal output)
pbvs_controller   --/arm/cartesian_velocity_cmd(TwistStamped)-->   sim_bridge
```
