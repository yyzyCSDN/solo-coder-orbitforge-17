"""Pinhole star camera model.

Boiler sight is the body +z axis. Focal length is expressed in pixels,
derived from the (square) field of view, so the model is fully defined by
``n_pixels`` and ``fov_deg``.

    u = cx + f * bx / bz,  v = cy + f * by / bz
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CameraModel:
    n_pixels: int = 256
    fov_deg: float = 12.0
    psf_sigma_px: float = 1.3
    # Detection: only stars at/above this magnitude are rendered.
    limiting_magnitude: float = 6.0
    # Zero-magnitude star gives this many e-/s at the detector (throughput in).
    zero_mag_flux_e_s: float = 4.0e5
    exposure_s: float = 0.2
    background_e_px: float = 15.0
    read_noise_e: float = 5.0
    full_well_e: float = 9.0e4
    # PSF support; blobs touching this border band are flagged edge-clipped.
    edge_margin_px: int = 4
    detection_sigma: float = 5.0
    min_blob_pixels: int = 3

    @property
    def focal_px(self) -> float:
        return (self.n_pixels / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    @property
    def center_px(self) -> float:
        return (self.n_pixels - 1) / 2.0

    @property
    def arcsec_per_pixel(self) -> float:
        return math.degrees(math.atan(1.0 / self.focal_px)) * 3600.0

    def project(self, body_unit):
        """Pixel coordinates (u, v), or None when the star is behind the sensor."""
        bx, by, bz = body_unit
        if bz <= 1e-9:
            return None
        c = self.center_px
        f = self.focal_px
        return c + f * bx / bz, c + f * by / bz

    def unproject(self, u, v):
        """Pixel coordinates -> unit vector in the tracker body frame."""
        c = self.center_px
        f = self.focal_px
        xn, yn = (u - c) / f, (v - c) / f
        n = math.sqrt(1.0 + xn * xn + yn * yn)
        return xn / n, yn / n, 1.0 / n

    def on_sensor(self, u, v) -> bool:
        return 0.0 <= u < self.n_pixels and 0.0 <= v < self.n_pixels

    def in_edge_band(self, u, v) -> bool:
        m = self.edge_margin_px
        return u < m or v < m or u >= self.n_pixels - m or v >= self.n_pixels - m

    @property
    def psf_half_px(self) -> int:
        import math as _math
        return int(_math.ceil(3.5 * self.psf_sigma_px))

    def psf_touches_edge(self, u, v) -> bool:
        """True when the PSF support reaches into the border band (or off-array)."""
        m = self.edge_margin_px + self.psf_half_px
        return u < m or v < m or u >= self.n_pixels - m or v >= self.n_pixels - m

    def star_electrons(self, mag) -> float:
        return self.zero_mag_flux_e_s * self.exposure_s * 10.0 ** (-0.4 * mag)
