"""Image-only spot detection and sub-pixel centroiding.

Works purely on the exposed frame: robust background (median) and noise
(1.4826 * MAD) estimates, a detection threshold, 8-connected component
labeling, and intensity-weighted centroids.

Blobs touching the camera's edge band are flagged ``edge_clipped`` - their
PSF is truncated by the sensor boundary, so their centroid is biased and
the attitude estimator must not trust them at full weight.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .camera import CameraModel


@dataclass
class Detection:
    u: float
    v: float
    flux_e: float            # measured electrons above background (truncated)
    flux_corrected_e: float  # flux after threshold-truncation correction
    mag_est: float           # estimated magnitude from corrected flux
    sigma_px: float
    n_pixels: int
    edge_clipped: bool
    body_unit: tuple  # unprojected line of sight in the body frame


def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _label_blobs(mask, n):
    """8-connected components; returns list of pixel-index lists."""
    visited = bytearray(n * n)
    blobs = []
    for seed in range(n * n):
        if not mask[seed] or visited[seed]:
            continue
        stack = [seed]
        visited[seed] = 1
        blob = []
        while stack:
            p = stack.pop()
            blob.append(p)
            pj, pi = divmod(p, n)
            for dj in (-1, 0, 1):
                j = pj + dj
                if j < 0 or j >= n:
                    continue
                row = j * n
                for di in (-1, 0, 1):
                    i = pi + di
                    if i < 0 or i >= n or (dj == 0 and di == 0):
                        continue
                    q = row + i
                    if mask[q] and not visited[q]:
                        visited[q] = 1
                        stack.append(q)
        blobs.append(blob)
    return blobs


def _despike(image, n, bg, noise):
    """Hot-pixel correction: replace isolated saturated pixels by background.

    A real PSF peak is always supported by bright neighbours; a cosmic-ray /
    hot pixel stands alone. Left in, it would merge into a nearby blob and
    drag its centroid by ~a pixel.
    """
    img = list(image)
    limit = bg + 20.0 * noise
    for j in range(1, n - 1):
        row = j * n
        for i in range(1, n - 1):
            v = img[row + i]
            if v < limit:
                continue
            nb = max(
                img[row + i - 1], img[row + i + 1],
                img[row - n + i], img[row + n + i],
                img[row - n + i - 1], img[row - n + i + 1],
                img[row + n + i - 1], img[row + n + i + 1],
            )
            if v > 4.0 * nb + 5.0 * noise:
                img[row + i] = bg
    return img


def detect_spots(image, camera: CameraModel):
    """Threshold, label and centroid one frame. Returns list[Detection]."""
    n = camera.n_pixels
    bg = _median(image)
    mad = _median([abs(x - bg) for x in image])
    noise = max(1.4826 * mad, camera.read_noise_e, 1.0)
    thresh = bg + camera.detection_sigma * noise
    image = _despike(image, n, bg, noise)

    mask = bytearray(n * n)
    for k, x in enumerate(image):
        if x > thresh:
            mask[k] = 1

    detections = []
    for blob in _label_blobs(mask, n):
        if len(blob) < camera.min_blob_pixels:
            continue  # hot pixel / noise spike
        wsum = usum = vsum = 0.0
        edge = False
        for p in blob:
            j, i = divmod(p, n)
            w = image[p] - bg
            if w <= 0.0:
                continue
            wsum += w
            usum += w * i
            vsum += w * j
            if (i < camera.edge_margin_px or j < camera.edge_margin_px
                    or i >= n - camera.edge_margin_px
                    or j >= n - camera.edge_margin_px):
                edge = True
        if wsum <= 0.0:
            continue
        u, v = usum / wsum, vsum / wsum
        # Per-axis centroid uncertainty: photon term sigma_PSF^2/N plus
        # background shot/read noise over the blob. The ideal Cramer-Rao bound
        # is then inflated by an empirical ~1.6 (Monte-Carlo calibrated): the
        # centroid is formed only from pixels that crossed the detection
        # threshold, and thresholded weighted centroiding is that much worse
        # than the full-PSF bound for near-limit stars.
        bg_var = len(blob) * (bg + camera.read_noise_e ** 2)
        sigma_px = camera.psf_sigma_px * math.sqrt(1.0 / wsum + bg_var / wsum ** 2)
        sigma_px *= 1.6
        # Photometry: the blob only collects the PSF fraction above the
        # threshold. For a Gaussian of peak P, the captured fraction is
        # 1 - t/P (level-set of a 2-D Gaussian), with P recovered from the
        # measured sum. Correcting removes a ~0.4 mag systematic bias.
        t_rel = thresh - bg
        peak = wsum / (2.0 * math.pi * camera.psf_sigma_px ** 2) + t_rel
        captured = max(1.0 - t_rel / peak, 0.05)
        flux_corr = wsum / captured
        mag_est = -2.5 * math.log10(
            flux_corr / (camera.zero_mag_flux_e_s * camera.exposure_s))
        detections.append(Detection(
            u=u, v=v, flux_e=wsum, flux_corrected_e=flux_corr, mag_est=mag_est,
            sigma_px=sigma_px, n_pixels=len(blob),
            edge_clipped=edge, body_unit=camera.unproject(u, v),
        ))
    detections.sort(key=lambda d: -d.flux_corrected_e)
    return detections, {"background_e": bg, "noise_e": noise, "threshold_e": thresh}
