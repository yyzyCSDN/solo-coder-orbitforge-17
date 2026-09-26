from __future__ import annotations
from dataclasses import dataclass
import bisect
import math
import random
from orbitforge.core.vector import Vec3

@dataclass(frozen=True)
class CatalogStar:
    star_id: int
    unit: Vec3      # 惯性系单位矢量
    mag: float      # 视星等

class StarCatalog:
    """全天星表。星等分布按密度 ∝ 10^(0.4*mag) 采样（亮星少、暗星多）。"""

    def __init__(self, stars):
        self.stars = list(stars)
        self.by_id = {s.star_id: s for s in self.stars}

    def __len__(self):
        return len(self.stars)

    @classmethod
    def random(cls, n_stars=6000, mag_bright=1.5, mag_faint=6.8, seed=42):
        rng = random.Random(seed)
        lo = 10.0 ** (0.4 * mag_bright)
        hi = 10.0 ** (0.4 * mag_faint)
        stars = []
        for i in range(n_stars):
            z = rng.uniform(-1.0, 1.0)
            phi = rng.uniform(0.0, 2.0 * math.pi)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            unit = Vec3(r * math.cos(phi), r * math.sin(phi), z)
            mag = 2.5 * math.log10(lo + rng.random() * (hi - lo))
            stars.append(CatalogStar(i, unit, mag))
        return cls(stars)

    def brighter_than(self, mag_limit):
        return StarCatalog([s for s in self.stars if s.mag <= mag_limit])

    def cone(self, pointing: Vec3, radius_rad):
        c = math.cos(radius_rad)
        return [s for s in self.stars if s.unit.dot(pointing) >= c]

class PairDatabase:
    """星对角距数据库：按角距排序存储 (sep, id_a, id_b)，支持区间查询。"""

    def __init__(self, seps, ids_a, ids_b):
        self.seps = seps
        self.ids_a = ids_a
        self.ids_b = ids_b

    def __len__(self):
        return len(self.seps)

    @classmethod
    def build(cls, catalog: StarCatalog, min_sep_rad, max_sep_rad):
        stars = catalog.stars
        n = len(stars)
        xs = [s.unit.x for s in stars]
        ys = [s.unit.y for s in stars]
        zs = [s.unit.z for s in stars]
        ids = [s.star_id for s in stars]
        cos_lo = math.cos(max_sep_rad)
        cos_hi = math.cos(min_sep_rad)
        pairs = []
        for i in range(n):
            xi, yi, zi, idi = xs[i], ys[i], zs[i], ids[i]
            for j in range(i + 1, n):
                d = xi * xs[j] + yi * ys[j] + zi * zs[j]
                if cos_lo <= d <= cos_hi:
                    sep = math.acos(max(-1.0, min(1.0, d)))
                    pairs.append((sep, idi, ids[j]))
        pairs.sort(key=lambda p: p[0])
        return cls([p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs])

    def query(self, angle_rad, tol_rad):
        lo = bisect.bisect_left(self.seps, angle_rad - tol_rad)
        hi = bisect.bisect_right(self.seps, angle_rad + tol_rad)
        return [(self.ids_a[k], self.ids_b[k]) for k in range(lo, hi)]
