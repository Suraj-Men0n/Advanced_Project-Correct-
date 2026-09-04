import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from tyre_sorting.decision.routing import route_fragment
from tyre_sorting.perception.cnn_decoder import SpectralCNN

def test_routing_confidence_gate():
    assert route_fragment('otr_mining', 0.50).route == 'bypass_reject'
    assert route_fragment('otr_mining', 0.90).route == 'devulcanization_cbr'

def test_cnn_forward_shapes():
    import torch
    model=SpectralCNN(); a,b=model(torch.randn(3,4,128)); assert a.shape==(3,4); assert b.shape==(3,2)

def test_routing_surface_gate_and_reason():
    assert route_fragment('otr_mining', 0.90, 'tread').route == 'devulcanization_cbr'
    assert route_fragment('otr_mining', 0.90, 'unknown').route == 'bypass_reject'
    assert 'sidewall' in route_fragment('truck_hgv', 0.90, 'sidewall').reason
