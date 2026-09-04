from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

try:
    from tyre_sorting.sim.camera_geom import CAMERA_EYE
    _CAMERA_Z = CAMERA_EYE[2]
except Exception:
    # Launch files run under the ros2 launch machinery, whose sys.path
    # doesn't always include the package's python modules the way a node
    # process does. Fall back to the current literal rather than failing
    # the whole launch on an ImportError -- but keep the derivation above
    # as the source of truth when it is importable.
    _CAMERA_Z = 1.05

# Derived, not hardcoded. The camera-to-belt-top distance must match the
# actual camera pose or depth segmentation breaks completely: if the value
# is LOWER than the true distance, the empty belt surface itself passes the
# foreground test and segmentation returns one blob spanning the whole
# scene (confirmed live -- boxes were drawn around the robot arm and bins,
# labelled as tyre fragments).
#
# This file previously hardcoded 0.85, which was correct only for an
# earlier camera height. Since a launch parameter OVERRIDES the node's own
# default, that stale value would have silently defeated the fix applied
# in segmentation_node.py.
BELT_TOP_Z = 0.05 + 0.02          # ConveyorConfig.z_height + belt box half-thickness
BELT_PLANE_DEPTH = _CAMERA_Z - BELT_TOP_Z
BELT_SPEED = 0.05                 # shared by sim_bridge and pbvs_controller --
                                   # these two MUST agree or the PBVS
                                   # feed-forward term fights the real belt


def generate_launch_description():
    sim_bridge_node = Node(package='tyre_sorting', executable='simulation_bridge', name='pybullet_sim_bridge', output='screen', parameters=[{'gui': True, 'belt_speed': BELT_SPEED, 'dataset_dir': 'dataset', 'control_mode': LaunchConfiguration('control_mode')}])
    return LaunchDescription([
        DeclareLaunchArgument(
            'control_mode', default_value='state_machine',
            description="'state_machine' drives the arm from ground-truth "
                        "fragment positions (proven full pick-and-place); "
                        "'pbvs' drives it from camera-derived velocity "
                        "commands published by pbvs_controller_node -- the "
                        "O4 closed visual-servoing loop."),
        sim_bridge_node,
        Node(package='tyre_sorting', executable='geometry_segmentation', name='geometry_segmentation_node', output='screen', parameters=[{'belt_plane_depth': BELT_PLANE_DEPTH}]),
        Node(package='tyre_sorting', executable='spectral_classifier', name='spectral_classifier_node', output='screen', parameters=[{'checkpoint': 'models/spectral_cnn.pt'}]),
        Node(package='tyre_sorting', executable='pbvs_controller', name='pbvs_controller_node', output='screen', parameters=[{'belt_speed': BELT_SPEED}]),
        Node(package='tyre_sorting', executable='multimodal_fusion', name='multimodal_fusion_node', output='screen'),
        Node(package='tyre_sorting', executable='routing_node', name='routing_decision_node', output='screen'),
        # AUTO-SHUTDOWN: sim_bridge now exits cleanly on its own once the
        # run is complete (see sim_bridge.py's completion check), but by
        # default that only stops sim_bridge itself -- the other 5 nodes
        # would keep spinning with nothing to do, still requiring Ctrl+C.
        # This handler shuts down the whole launch when sim_bridge exits.
        RegisterEventHandler(OnProcessExit(
            target_action=sim_bridge_node,
            on_exit=[EmitEvent(event=Shutdown(
                reason='simulation_bridge finished its run'))],
        )),
    ])
