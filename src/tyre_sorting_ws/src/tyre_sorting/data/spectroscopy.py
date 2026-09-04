"""
Synthetic multi-channel Mid-IR / Terahertz spectroscopy signature generator.

Models the absorption spectrum of a tyre rubber fragment as a sum of
Lorentzian/Gaussian absorption peaks whose position, width and amplitude are
driven by the fragment's underlying material composition (NR / SBR / carbon
black ratio), plus a carbon-black broadband continuum absorber and additive
sensor noise. This gives every fragment a physically-motivated, reproducible
"chemical fingerprint" that a 1D-CNN can learn to decode, without requiring
real TGA/pyrolysis-GC/MS lab data (Gap 4 in the proposal).

Composition ratios per tyre type are taken directly from Slide 3's table.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# --- Composition table (from proposal Slide 3), fractions of rubber mass ---
TYRE_COMPOSITIONS = {
    # tyre_type: (NR, SBR, carbon_black, steel_fibre)
    "passenger": dict(nr=0.14, sbr=0.27, cb=0.28, sf=0.16),
    "truck_hgv": dict(nr=0.27, sbr=0.14, cb=0.28, sf=0.145),
    "motorcycle": dict(nr=0.40, sbr=0.15, cb=0.25, sf=0.10),
    "otr_mining": dict(nr=0.70, sbr=0.08, cb=0.18, sf=0.15),
}

# Characteristic absorption bands (center in cm^-1-like normalized units
# 0-1 mapped across the simulated Mid-IR/THz sweep window). These are
# illustrative "material fingerprints", not measured lab spectra -- label
# them as such in your dissertation methodology section.
NR_PEAKS = [(0.18, 0.02, 1.0), (0.46, 0.015, 0.6), (0.72, 0.025, 0.4)]
SBR_PEAKS = [(0.24, 0.018, 0.9), (0.51, 0.02, 0.7), (0.80, 0.02, 0.5)]
CB_CONTINUUM_STRENGTH = 2.2  # broadband absorber, scales with cb fraction


@dataclass
class SpectroscopyConfig:
    n_channels: int = 128          # spectral resolution (samples across sweep)
    n_bands: int = 4               # multi-channel = e.g. 4 discrete THz/MIR sub-bands
    noise_std: float = 0.03        # sensor + electronic noise
    baseline_drift_std: float = 0.02
    seed: int | None = None


def _lorentzian(x: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * (width ** 2) / ((x - center) ** 2 + width ** 2)


def synthesize_signature(composition: dict, cfg: SpectroscopyConfig,
                          rng: np.random.Generator,
                          surface_class: str | None = None) -> np.ndarray:
    """
    Returns an array of shape (n_bands, n_channels): a multi-channel
    spectroscopy signature for one fragment.

    ``surface_class`` is an explicit synthetic factor.  The original dataset
    labelled tread/sidewall but did not include that label in the generated
    spectrum, so the CNN surface head was at chance.  The surface term below
    is a deliberately synthetic proxy for texture/exposure differences; it
    is useful for co-simulation, not a substitute for measured spectroscopy.
    """
    x = np.linspace(0.0, 1.0, cfg.n_channels)
    spectrum = np.zeros((cfg.n_bands, cfg.n_channels), dtype=np.float32)

    for b in range(cfg.n_bands):
        s = np.zeros(cfg.n_channels, dtype=np.float32)

        # NR and SBR contribute sharp absorption peaks, weighted by fraction
        for (c, w, a) in NR_PEAKS:
            s += _lorentzian(x, c + 0.01 * b, w, a * composition["nr"])
        for (c, w, a) in SBR_PEAKS:
            s += _lorentzian(x, c - 0.01 * b, w, a * composition["sbr"])

        # Carbon black: broadband continuum absorber, suppresses contrast
        # more strongly in lower bands (mimics NIR-blocking behaviour from
        # Slide 4 -- this is why multi-band SWIR+thermal fusion is modelled
        # rather than a single NIR channel).
        cb_suppression = CB_CONTINUUM_STRENGTH * composition["cb"] * (1.0 - 0.15 * b)
        s = s * np.exp(-cb_suppression * 0.3) + cb_suppression * 0.05

        # Synthetic surface-dependent micro-texture term.  Tread is modelled
        # with stronger fine-scale modulation than sidewall rubber.  This gives
        # the existing dual-head CNN a real training signal for surface_class.
        if surface_class == "tread":
            surface_amp = 0.060
            surface_phase = 0.4 + 0.15 * b
        elif surface_class == "sidewall":
            surface_amp = 0.008
            surface_phase = 1.2 + 0.08 * b
        else:
            surface_amp = 0.0
            surface_phase = 0.0
        s += surface_amp * np.sin(2 * np.pi * (5.0 + 0.4 * b) * x + surface_phase)

        # Sensor noise + slow baseline drift
        s += rng.normal(0, cfg.noise_std, size=cfg.n_channels)
        drift = rng.normal(0, cfg.baseline_drift_std) * np.sin(2 * np.pi * x + b)
        s += drift

        spectrum[b] = s

    return spectrum
