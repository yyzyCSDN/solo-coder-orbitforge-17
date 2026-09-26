from __future__ import annotations
from dataclasses import dataclass
import math
from orbitforge.core.vector import Vec3
from orbitforge.attitude.quaternion import Quaternion

@dataclass(frozen=True)
class CameraParams:
    n_pixels: int = 512
    fov_deg: float = 12.0
    psf_sigma_px: float = 0.8
    background_e: float = 60.0       # 背景电子数/像元
    read_noise_e: float = 8.0        # 读出噪声/像元
    flux_mag0_e: float = 2.5e5       # 0 等星总流量（电子数）
    edge_margin_px: int = 6          # 质心在阵列外该距离内的星仍可部分入窗

    @property
    def focal_px(self):
        return 0.5 * self.n_pixels / math.tan(0.5 * self.fov_rad)

    @property
    def fov_rad(self):
        return math.radians(self.fov_deg)

    @property
    def pixel_scale_rad(self):
        return math.atan(1.0 / self.focal_px)

    @property
    def psf_radius_px(self):
        return int(math.ceil(4.0 * self.psf_sigma_px))

@dataclass
class ProjectedStar:
    star_id: int          # -1 表示注入的假星（非星表源）
    px: float
    py: float
    mag: float
    flux_e: float
    body: Vec3 | None     # 本体系单位矢量（假星为 None）
    truncated: bool       # PSF 是否被视场边缘截断

def flux_from_mag(mag, params: CameraParams):
    return params.flux_mag0_e * 10.0 ** (-0.4 * mag)

def project_stars(catalog, attitude: Quaternion, params: CameraParams, mag_render_limit=7.5):
    """把星表星投影到焦平面。attitude 满足 v_body = attitude.rotate(v_inertial)。"""
    f = params.focal_px
    c = 0.5 * (params.n_pixels - 1)
    half = 0.5 * params.n_pixels + params.edge_margin_px
    r = params.psf_radius_px
    out = []
    for s in catalog.stars:
        if s.mag > mag_render_limit:
            continue
        b = attitude.rotate(s.unit)
        if b.z <= 0.0:
            continue
        px = f * b.x / b.z + c
        py = f * b.y / b.z + c
        if abs(px - c) > half or abs(py - c) > half:
            continue
        hi = params.n_pixels - 1 - r
        truncated = not (r <= px <= hi and r <= py <= hi)
        out.append(ProjectedStar(s.star_id, px, py, s.mag, flux_from_mag(s.mag, params), b, truncated))
    return out

def render_image(projected, params: CameraParams, rng, n_fake_stars=0, fake_mag_range=(3.0, 5.0)):
    """渲染星点图像：高斯 PSF + 背景 + 散粒噪声(高斯近似) + 读出噪声。

    返回 (image, truth)。truth 含全部真实星点（含假星，star_id=-1），
    仅用于评估，不参与后续识别与定姿。
    """
    n = params.n_pixels
    img = [[params.background_e] * n for _ in range(n)]
    sig2 = 2.0 * params.psf_sigma_px ** 2
    norm = 1.0 / (math.pi * sig2)
    r = params.psf_radius_px
    spots = [(p.px, p.py, p.flux_e) for p in projected]
    truth = list(projected)
    for _ in range(n_fake_stars):
        fx = rng.uniform(8.0, n - 9.0)
        fy = rng.uniform(8.0, n - 9.0)
        fmag = rng.uniform(*fake_mag_range)
        spots.append((fx, fy, flux_from_mag(fmag, params)))
        truth.append(ProjectedStar(-1, fx, fy, fmag, flux_from_mag(fmag, params), None, False))
    for px, py, flux in spots:
        x0 = int(px)
        y0 = int(py)
        for dy in range(-r, r + 1):
            yy = y0 + dy
            if yy < 0 or yy >= n:
                continue
            row = img[yy]
            for dx in range(-r, r + 1):
                xx = x0 + dx
                if 0 <= xx < n:
                    ddx = xx - px
                    ddy = yy - py
                    row[xx] += flux * norm * math.exp(-(ddx * ddx + ddy * ddy) / sig2)
    read2 = params.read_noise_e ** 2
    gauss = rng.gauss
    sqrt = math.sqrt
    for yy in range(n):
        row = img[yy]
        for xx in range(n):
            lam = row[xx]
            row[xx] = lam + gauss(0.0, sqrt(lam + read2))
    return img, truth
