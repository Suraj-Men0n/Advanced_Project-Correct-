"""ROS 2 spectral inference node."""
from __future__ import annotations
import numpy as np

def build_node():
    import rclpy
    from rclpy.node import Node
    from tyre_sorting_msgs.msg import SpectralArray, ScoredSpectralArray, ScoredSpectralSample
    from tyre_sorting.perception.cnn_decoder import SpectralInference
    from tyre_sorting.paths import resolve_data_path
    class SpectralClassifierNode(Node):
        def __init__(self):
            super().__init__('spectral_classifier_node')
            ckpt=resolve_data_path(self.declare_parameter('checkpoint','models/spectral_cnn.pt').value)
            self.infer=SpectralInference(ckpt)
            self.pub=self.create_publisher(ScoredSpectralArray,'/sensors/scored_spectroscopy',10)
            self.sub=self.create_subscription(SpectralArray,'/sensors/spectroscopy',self.on_spec,10)
        def on_spec(self,msg):
            out=ScoredSpectralArray(); out.header=msg.header
            for s in msg.samples:
                try:
                    label, conf, surface, surface_conf = self.infer.predict(
                        np.asarray(s.values,dtype=np.float32).reshape(4,-1))
                except Exception as exc:
                    self.get_logger().warn(f'Inference failed for {s.id}: {exc}'); continue
                x=ScoredSpectralSample()
                x.id=s.id; x.centroid=s.centroid
                x.material_class=label; x.classification_confidence=float(conf)
                x.surface_class=surface; x.surface_classification_confidence=float(surface_conf)
                out.samples.append(x)
            self.pub.publish(out)
    return SpectralClassifierNode

def main():
    import rclpy
    rclpy.init(); node=build_node()(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
