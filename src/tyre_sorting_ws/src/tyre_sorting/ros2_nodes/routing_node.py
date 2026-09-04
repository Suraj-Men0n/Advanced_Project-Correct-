"""ROS 2 route-decision node with confidence gating and audit logging."""
from __future__ import annotations

def build_node():
    import rclpy
    from rclpy.node import Node
    from tyre_sorting_msgs.msg import FragmentArray
    from std_msgs.msg import String
    from tyre_sorting.decision.routing import route_fragment
    import json

    class RoutingNode(Node):
        def __init__(self):
            super().__init__('routing_decision_node')
            self.sub = self.create_subscription(FragmentArray, '/perception/fused_fragments', self.on_msg, 10)
            self.pub = self.create_publisher(String, '/routing/decision', 10)

        def on_msg(self, msg):
            for f in msg.fragments:
                d=route_fragment(f.material_class, float(f.classification_confidence), getattr(f, 'surface_class', 'unknown'))
                out=String(); out.data=json.dumps({'id':f.id,'route':d.route,'confidence':d.confidence,'surface_class':getattr(f,'surface_class','unknown'),'bbox':[f.bbox_xmin,f.bbox_ymin,f.bbox_xmax,f.bbox_ymax],'reason':d.reason})
                self.pub.publish(out); self.get_logger().info(out.data)
    return RoutingNode

def main():
    import rclpy
    rclpy.init(); node=build_node()(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
