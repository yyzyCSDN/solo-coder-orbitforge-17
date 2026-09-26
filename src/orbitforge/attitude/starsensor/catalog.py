"""Synthetic star catalog and inter-star-angle pair database.

The catalog is a deterministic stand-in for Hipparcos-class catalogs:
isotropic unit vectors with a dN/dm proportional to 10**(0.32 m) magnitude
distribution (star counts roughly triple per magnitude).

The pair database holds every catalog pair closer than ``max_sep_rad``
together with its angular separation, sorted for binary-search lookup.
It is the geometric index used by the lost-in-space star identification
(star-pattern matching on angular distances).
"""
from __future__ import annotations

from dataclasses import dataclass
import bisect
import math

from .rotations import vdot


@dataclass(frozen=True)
class CatalogStar:
    index: int
    mag: float
    unit: tuple  # inertial unit direction (x, y, z)


class StarCatalog:
    def __init__(self, stars):
        self.stars = list(stars)

    @classmethod
    def synthetic(cls, n_stars=3000, mag_min=-1.5, mag_max=6.5, seed=20260926):
        """Generate a repeatable all-sky catalog."""
        import random
        rng = random.Random(seed)
        # Cumulative distribution over magnitude: P(<m) ~ 10**(0.32 m).
        lo, hi = 10.0 ** (0.32 * mag_min), 10.0 ** (0.32 * mag_max)
        stars = []
        for i in range(n_stars):
            u = rng.random() * (hi - lo) + lo
            mag = math.log10(u) / 0.32
            # Uniform on the unit sphere.
            z = rng.uniform(-1.0, 1.0)
            lam = rng.uniform(0.0, 2.0 * math.pi)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            stars.append(CatalogStar(i, mag, (r * math.cos(lam), r * math.sin(lam), z)))
        return cls(stars)

    def units(self):
        return [s.unit for s in self.stars]

    def build_pair_database(self, max_sep_rad, mag_limit=None):
        return PairDatabase.build(self.stars, max_sep_rad, mag_limit)


@dataclass
class PairDatabase:
    angles: list          # sorted angular separations (rad)
    pairs: list           # aligned (i, j) catalog index tuples
    mag_limit: float
    max_sep_rad: float

    @staticmethod
    def build(stars, max_sep_rad, mag_limit=None):
        kept = [s for s in stars if mag_limit is None or s.mag <= mag_limit]
        cos_min = math.cos(max_sep_rad)
        out = []
        n = len(kept)
        for a in range(n):
            u = kept[a].unit
            for b in range(a + 1, n):
                v = kept[b].unit
                d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
                if d >= cos_min:
                    d = min(1.0, d)
                    out.append((math.acos(d), kept[a].index, kept[b].index))
        out.sort(key=lambda t: t[0])
        return PairDatabase(
            [t[0] for t in out], [(t[1], t[2]) for t in out],
            mag_limit if mag_limit is not None else math.inf, max_sep_rad,
        )

    def query(self, angle, tol):
        """All catalog pairs whose separation is within [angle-tol, angle+tol]."""
        lo = bisect.bisect_left(self.angles, angle - tol)
        hi = bisect.bisect_right(self.angles, angle + tol)
        return self.pairs[lo:hi]

    def __len__(self):
        return len(self.angles)


def angle_between(a, b) -> float:
    return math.acos(max(-1.0, min(1.0, vdot(a, b))))
