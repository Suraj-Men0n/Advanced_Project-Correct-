"""Associates geometric detections with spectral predictions in camera frame."""
from __future__ import annotations
import numpy as np

def build_node():
    import rclpy
    from rclpy.node import Node
    from tyre_sorting_msgs.msg import ScoredSpectralArray, FragmentArray
    from tyre_sorting_msgs.msg import Fragment

    class MultimodalFusionNode(Node):
        def __init__(self):
            super().__init__('multimodal_fusion_node')
            self.max_distance=float(self.declare_parameter('max_association_distance',0.15).value)
            self.geometry=None; self.spectra=None
            self.pub=self.create_publisher(FragmentArray,'/perception/fused_fragments',10)
            self.create_subscription(FragmentArray,'/perception/fragments',self.on_geometry,10)
            self.create_subscription(ScoredSpectralArray,'/sensors/scored_spectroscopy',self.on_spectra,10)
        def on_geometry(self,msg): self.geometry=msg; self._try_publish()
        def on_spectra(self,msg): self.spectra=msg; self._try_publish()
        def _try_publish(self):
            if self.geometry is None or self.spectra is None: return
            out=FragmentArray(); out.header=self.geometry.header
            remaining=list(self.spectra.samples)
            for g in self.geometry.fragments:
                if not remaining: break
                gp=np.array([g.centroid.x,g.centroid.y,g.centroid.z])
                j=min(range(len(remaining)),key=lambda k: np.linalg.norm(gp-np.array([remaining[k].centroid.x,remaining[k].centroid.y,remaining[k].centroid.z])))
                s=remaining[j]
                dist=np.linalg.norm(gp-np.array([s.centroid.x,s.centroid.y,s.centroid.z]))
                if dist>self.max_distance: continue
                f=Fragment()
                f.id=g.id; f.centroid=g.centroid; f.pixel_area=g.pixel_area
                f.bbox_xmin=g.bbox_xmin; f.bbox_ymin=g.bbox_ymin
                f.bbox_xmax=g.bbox_xmax; f.bbox_ymax=g.bbox_ymax
                f.vision_surface_class=g.vision_surface_class
                f.vision_surface_confidence=float(g.vision_surface_confidence)
                f.material_class=getattr(s,'material_class','')
                f.classification_confidence=getattr(s,'classification_confidence',0.0)

                spectral_surface=getattr(s,'surface_class','')
                spectral_conf=float(getattr(s,'surface_classification_confidence',0.0))
                vision_surface=getattr(g,'vision_surface_class','')
                vision_conf=float(getattr(g,'vision_surface_confidence',0.0))
                # Prefer agreement; otherwise use the higher-confidence modality.
                # Unknown remains explicit rather than silently inventing a class.
                if spectral_surface and spectral_surface == vision_surface and min(spectral_conf, vision_conf) >= 0.55:
                    f.surface_class=spectral_surface
                    f.surface_classification_confidence=float((spectral_conf + vision_conf) / 2.0)
                elif spectral_conf >= vision_conf and spectral_conf >= 0.60:
                    f.surface_class=spectral_surface
                    f.surface_classification_confidence=spectral_conf
                elif vision_conf >= 0.60:
                    f.surface_class=vision_surface
                    f.surface_classification_confidence=vision_conf
                else:
                    f.surface_class='unknown'
                    f.surface_classification_confidence=0.0
                out.fragments.append(f); remaining.pop(j)
            self.pub.publish(out)
            # Without this, an update to EITHER stream re-triggers a publish
            # immediately using whatever the OTHER stream's last value was --
            # the None-guard above only protects the very first pairing.
            # Confirmed real: geometry and spectra arrive on separate
            # subscriptions with no guaranteed ordering/timing.
            self.geometry = None
            self.spectra = None
    return MultimodalFusionNode

def main():
    import rclpy
    rclpy.init(); node=build_node()(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
