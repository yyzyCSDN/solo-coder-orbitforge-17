from __future__ import annotations
from dataclasses import dataclass
import math
import statistics
from orbitforge.core.vector import Vec3
from .camera import CameraParams

@dataclass
class Centroid:
    px: float
    py: float
    flux_e: float
    peak_e: float
    n_pixels: int
    touches_edge: bool

def estimate_background(image, sample_step=5):
    n = len(image)
    samples = [image[y][x] for y in range(0, n, sample_step) for x in range(0, n, sample_step)]
    bg = statistics.median(samples)
    mad = statistics.median([abs(v - bg) for v in samples])
    return bg, max(1.4826 * mad, 1e-6)

def extract_centroids(image, threshold_sigma=5.0, min_pixels=3, max_pixels=400):
    """阈值分割 + 8 连通域标记 + 一阶矩质心。返回 (centroids, background, sigma)。"""
    n = len(image)
    bg, sigma = estimate_background(image)
    thr = bg + threshold_sigma * sigma
    visited = bytearray(n * n)
    cents = []
    for y0 in range(n):
        row0 = image[y0]
        base0 = y0 * n
        for x0 in range(n):
            if visited[base0 + x0] or row0[x0] <= thr:
                continue
            stack = [(x0, y0)]
            visited[base0 + x0] = 1
            blob = []
            while stack:
                x, y = stack.pop()
                blob.append((x, y))
                for dy in (-1, 0, 1):
                    yy = y + dy
                    if yy < 0 or yy >= n:
                        continue
                    base = yy * n
                    row = image[yy]
                    for dx in (-1, 0, 1):
                        xx = x + dx
                        if 0 <= xx < n and not visited[base + xx] and row[xx] > thr:
                            visited[base + xx] = 1
                            stack.append((xx, yy))
            if not (min_pixels <= len(blob) <= max_pixels):
                continue
            wsum = sx = sy = 0.0
            peak = -1e30
            for x, y in blob:
                w = image[y][x] - bg
                if w > 0.0:
                    wsum += w
                    sx += w * x
                    sy += w * y
                if image[y][x] > peak:
                    peak = image[y][x]
            if wsum <= 0.0:
                continue
            edge = any(x == 0 or y == 0 or x == n - 1 or y == n - 1 for x, y in blob)
            cents.append(Centroid(sx / wsum, sy / wsum, wsum, peak - bg, len(blob), edge))
    cents.sort(key=lambda c: -c.flux_e)
    return cents, bg, sigma

def centroid_to_body(cent: Centroid, params: CameraParams) -> Vec3:
    """针孔模型反投影：像元坐标 -> 本体系单位向量（+Z 为光轴）。"""
    c = 0.5 * (params.n_pixels - 1)
    return Vec3((cent.px - c) / params.focal_px, (cent.py - c) / params.focal_px, 1.0).unit()
