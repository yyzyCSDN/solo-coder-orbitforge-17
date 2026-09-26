"""Synthetic star image rendering.

Given the catalog, the TRUE attitude and the camera, project every visible
star onto the focal plane and expose the detector: Gaussian PSF spots,
Poisson photon noise, Gaussian read noise, flat background and full-well
clipping. Detector defects can be injected:

* ``n_spurious``   - unresolved bright blobs with no catalog counterpart
                     (cosmic rays / stray light); survive as detections.
* ``n_hot_pixels`` - single saturated pixels; the centroider rejects them
                     by minimum blob area.

This module is the ONLY place the true attitude enters the simulation.
Its :class:`RenderInfo` is sensor-side bookkeeping used for reporting and
post-hoc validation; the attitude estimator never receives it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from .camera import CameraModel
from .rotations import quat_to_dcm, dcm_apply


@dataclass
class StarRenderRecord:
    catalog_index: int
    mag: float
    u: float            # true projected center (may be off-sensor)
    v: float
    inside_sensor: bool
    edge_clipped: bool  # PSF intersects the border band
    rendered: bool
    reason: str         # ok | edge_clipped | magnitude | out_of_field
    truth_body: tuple


@dataclass
class RenderInfo:
    records: list = field(default_factory=list)
    spurious_uv: list = field(default_factory=list)
    hot_pixel_uv: list = field(default_factory=list)
    n_rendered: int = 0
    n_edge_clipped: int = 0
    n_skipped_magnitude: int = 0
    n_out_of_field: int = 0

    def visible_records(self):
        return [r for r in self.records if r.rendered]


def _poisson_draw(lam: float, rng: random.Random) -> float:
    """Poisson draw; exact Knuth loop for small lambda, normal approx above 30."""
    if lam >= 30.0:
        return max(0.0, rng.gauss(lam, math.sqrt(lam)))
    L, k, p = math.exp(-lam), 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return float(k - 1)


def _add_spot(expected, n, u0, v0, sigma, flux):
    """Add one Gaussian PSF, clipped at the image boundary (edge bias!)."""
    half = int(math.ceil(3.5 * sigma))
    ui0, vj0 = int(round(u0)), int(round(v0))
    norm = flux / (2.0 * math.pi * sigma * sigma)
    two_s2 = 2.0 * sigma * sigma
    for j in range(max(0, vj0 - half), min(n, vj0 + half + 1)):
        dy = j - v0
        base = j * n
        for i in range(max(0, ui0 - half), min(n, ui0 + half + 1)):
            dx = i - u0
            expected[base + i] += norm * math.exp(-(dx * dx + dy * dy) / two_s2)


def render_image(catalog, q_true, camera: CameraModel,
                 n_spurious=0, n_hot_pixels=0, rng=None):
    """Expose one frame. Returns (flat float image of length n*n, RenderInfo)."""
    if rng is None:
        rng = random.Random()
    n = camera.n_pixels
    A = quat_to_dcm(q_true)
    info = RenderInfo()

    expected = [camera.background_e_px] * (n * n)
    m = camera.edge_margin_px

    for star in catalog.stars:
        body = dcm_apply(A, star.unit)
        proj = camera.project(body)
        rec = StarRenderRecord(
            star.index, star.mag,
            proj[0] if proj else math.nan,
            proj[1] if proj else math.nan,
            False, False, False, "", body,
        )
        if proj is None:
            rec.reason = "out_of_field"
            info.n_out_of_field += 1
            info.records.append(rec)
            continue
        u, v = proj
        rec.inside_sensor = camera.on_sensor(u, v)
        # PSF can bleed in from centers up to half a patch outside the array.
        bleed = camera.psf_half_px
        in_bleed = (-bleed <= u < n + bleed) and (-bleed <= v < n + bleed)
        rec.edge_clipped = (rec.inside_sensor and camera.psf_touches_edge(u, v)) \
            or (in_bleed and not rec.inside_sensor)
        if not in_bleed:
            rec.reason = "out_of_field"
            info.n_out_of_field += 1
        elif star.mag > camera.limiting_magnitude:
            rec.reason = "magnitude"
            info.n_skipped_magnitude += 1
        else:
            rec.rendered = True
            rec.reason = "edge_clipped" if rec.edge_clipped else "ok"
            _add_spot(expected, n, u, v, camera.psf_sigma_px,
                      camera.star_electrons(star.mag))
            info.n_rendered += 1
            if rec.edge_clipped:
                info.n_edge_clipped += 1
        info.records.append(rec)

    # Non-catalog bright blobs (cosmic rays / stray-light glints). Kept clear
    # of the edge band so they exercise identification, not edge handling.
    pad = m + 6
    for _ in range(n_spurious):
        u = rng.uniform(pad, n - 1 - pad)
        v = rng.uniform(pad, n - 1 - pad)
        mag = rng.uniform(2.5, 5.0)
        info.spurious_uv.append((u, v))
        _add_spot(expected, n, u, v, camera.psf_sigma_px, camera.star_electrons(mag))

    # Single saturated hot pixels - detectable pixels, not valid blobs.
    for _ in range(n_hot_pixels):
        i, j = rng.randrange(n), rng.randrange(n)
        info.hot_pixel_uv.append((i, j))
        expected[j * n + i] = camera.full_well_e

    # Photon shot noise + read noise, then full well / zero clipping.
    img = [0.0] * (n * n)
    rn = camera.read_noise_e
    fw = camera.full_well_e
    for k, lam in enumerate(expected):
        e = _poisson_draw(lam, rng) + rng.gauss(0.0, rn)
        img[k] = 0.0 if e < 0.0 else fw if e > fw else e
    return img, info
