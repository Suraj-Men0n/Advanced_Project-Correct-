"""Material-informed routing decision logic (T4 -> T5 -> actuator)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    confidence: float


def route_fragment(tyre_type: str, confidence: float, surface_class: str | None = None,
                   min_confidence: float = 0.65) -> RouteDecision:
    """Deterministic routing policy matching the proposal's high-value streams.

    Low-confidence classifications are rejected/bypassed instead of being
    silently sent to a premium material stream.
    """
    if confidence < min_confidence:
        return RouteDecision("bypass_reject", "classification_below_threshold", confidence)
    if surface_class not in {None, "tread", "sidewall"}:
        return RouteDecision("bypass_reject", "surface_unidentified", confidence)
    mapping = {
        "otr_mining": "devulcanization_cbr",
        "truck_hgv": "high_grade_crumb",
        "motorcycle": "speciality_crumb",
        "passenger": "recompounding_raw_material",
    }
    route = mapping.get(tyre_type, "bypass_reject")
    reason = f"{tyre_type}:{surface_class or 'unknown_surface'}"
    return RouteDecision(route, reason, confidence)
