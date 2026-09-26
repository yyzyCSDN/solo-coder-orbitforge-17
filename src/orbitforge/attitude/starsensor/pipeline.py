"""End-to-end star sensor simulation pipeline.

Strict separation of concerns:

* :meth:`StarSensorSim.render`    - sensor model; the ONLY consumer of the
                                    true attitude. Produces an image plus
                                    sensor-side bookkeeping (RenderInfo).
* :meth:`StarSensorSim.estimate`  - the estimator. Input is the image
                                    (and the onboard catalog). It has no
                                    access to the true attitude by
                                    construction, and returns an explicit
                                    failure status instead of any attitude
                                    when the scene is not observable.
* :func:`validate`                - post-hoc comparison of an estimate
                                    against the truth, for simulation
                                    verification only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from orbitforge.attitude.quaternion import Quaternion
from .camera import CameraModel
from .catalog import StarCatalog
from .centroid import detect_spots
from .estimation import estimate_with_rejection
from .identification import identify_stars
from .imaging import render_image
from .rotations import attitude_error_rad

ARCSEC = 206264.80624709636

STATUS_OK = "OK"
STATUS_DEGRADED = "DEGRADED"
STATUS_INSUFFICIENT_STARS = "INSUFFICIENT_STARS"
STATUS_LOST_IN_SPACE = "LOST_IN_SPACE"


@dataclass
class MatchReport:
    detection_index: int
    catalog_index: int
    votes: int
    residual_arcsec: float


@dataclass
class RejectedMatch:
    detection_index: int
    catalog_index: int
    votes: int
    residual_arcsec: float
    reason: str


@dataclass
class AttitudeSolution:
    status: str
    quaternion: Quaternion = None          # None unless an attitude was solved
    n_detected: int = 0
    n_edge_excluded: int = 0               # edge-clipped spots, not used
    n_unmatched: int = 0                  # detections with no catalog identity
    matches: list = field(default_factory=list)
    rejected_matches: list = field(default_factory=list)
    rms_residual_arcsec: float = math.nan
    sigma_body_arcsec: tuple = (math.nan, math.nan, math.nan)  # formal 1-sigma
    covariance_singular: bool = False
    detail: str = ""


class StarSensorSim:
    def __init__(self, catalog: StarCatalog, camera: CameraModel, seed=1,
                 pair_db=None):
        import random
        self.catalog = catalog
        self.camera = camera
        self.rng = random.Random(seed)
        self._pair_db = pair_db

    # ------------------------------------------------------------ sensor side

    def render(self, q_true, n_spurious=0, n_hot_pixels=0):
        """Expose one frame at the true attitude. Returns (image, RenderInfo)."""
        return render_image(self.catalog, q_true, self.camera,
                            n_spurious=n_spurious, n_hot_pixels=n_hot_pixels,
                            rng=self.rng)

    # ---------------------------------------------------------- estimator side

    def _pair_database(self):
        if self._pair_db is None:
            # Max separation of two stars in the square FOV is the full
            # diagonal (corner to corner), not the half-diagonal.
            diag = math.radians(self.camera.fov_deg) * math.sqrt(2.0)
            self._pair_db = self.catalog.build_pair_database(
                max_sep_rad=diag * 1.05,
                mag_limit=self.camera.limiting_magnitude,
            )
        return self._pair_db

    def estimate(self, image, k_sigma=3.5):
        """Identify stars and determine attitude from the image alone."""
        cam = self.camera
        detections, _det_stats = detect_spots(image, cam)
        n_edge = sum(1 for d in detections if d.edge_clipped)
        usable = [d for d in detections if not d.edge_clipped]

        if len(usable) < 3:
            return AttitudeSolution(
                status=STATUS_INSUFFICIENT_STARS,
                n_detected=len(detections), n_edge_excluded=n_edge,
                detail=(f"only {len(usable)} usable (non-edge) spots; "
                        f"need >= 3 for geometric identification"),
            )

        catalog_units = self.catalog.units()
        sigmas_all = [d.sigma_px / cam.focal_px for d in detections]
        matches, unmatched, seed_ok = identify_stars(
            detections, self._pair_database(), sigmas_all,
            self.catalog, cam, k_sigma=k_sigma)

        if len(matches) < 3:
            return AttitudeSolution(
                status=STATUS_LOST_IN_SPACE,
                n_detected=len(detections), n_edge_excluded=n_edge,
                n_unmatched=len(unmatched),
                detail=(f"no verified star pattern (seed_ok={seed_ok}); "
                        f"{len(matches)} supported matches, need >= 3"),
            )

        body = [detections[k].body_unit for k, _, _ in matches]
        inertial = [catalog_units[c] for _, c, _ in matches]
        sigmas = [sigmas_all[k] for k, _, _ in matches]
        est = estimate_with_rejection(body, inertial, sigmas)

        kept = est.kept
        rejected = [
            RejectedMatch(matches[i][0], matches[i][1], matches[i][2],
                          r * ARCSEC, "residual_outlier")
            for i, r in est.rejected
        ]

        if len(kept) < 3:
            return AttitudeSolution(
                status=STATUS_LOST_IN_SPACE,
                n_detected=len(detections), n_edge_excluded=n_edge,
                n_unmatched=len(unmatched), rejected_matches=rejected,
                detail="fewer than 3 matches survived residual verification",
            )

        # Consistency gate: standardized post-fit residuals must look like
        # unit-noise (rms(z) < 2.5). Absolute-arcsec gating would fail on
        # legitimately dim fields; this catches a garbage match set instead.
        z_rms = math.sqrt(
            sum((r / sigmas[i]) ** 2 for i, r in zip(kept, est.residuals_rad))
            / len(kept))
        if z_rms > 2.5:
            return AttitudeSolution(
                status=STATUS_LOST_IN_SPACE,
                n_detected=len(detections), n_edge_excluded=n_edge,
                n_unmatched=len(unmatched), rejected_matches=rejected,
                detail=(f"standardized residual rms {z_rms:.2f} > 2.5; "
                        f"match set rejected as inconsistent"),
            )
        cov = est.covariance_rad2
        singular = all(cov[i][i] == 0.0 for i in range(3))
        sig_b = tuple(math.sqrt(max(cov[i][i], 0.0)) * ARCSEC for i in range(3))
        match_reports = [
            MatchReport(matches[i][0], matches[i][1], matches[i][2],
                        r * ARCSEC)
            for i, r in zip(kept, est.residuals_rad)
        ]

        if len(kept) >= 4 and not singular:
            status = STATUS_OK
        else:
            status = STATUS_DEGRADED
        return AttitudeSolution(
            status=status, quaternion=est.quaternion,
            n_detected=len(detections), n_edge_excluded=n_edge,
            n_unmatched=len(unmatched),
            matches=match_reports, rejected_matches=rejected,
            rms_residual_arcsec=est.rms_residual_rad * ARCSEC,
            sigma_body_arcsec=sig_b, covariance_singular=singular,
            detail=f"{len(kept)} matches used, {len(rejected)} rejected",
        )


def validate(solution: AttitudeSolution, q_true):
    """Post-hoc truth comparison - simulation verification only.

    The estimator never calls this; it exists so a simulation can score the
    estimate against the (separately held) true attitude.
    """
    if solution.quaternion is None:
        return {"attitude_error_arcsec": math.inf,
                "note": "no attitude solution to validate"}
    err = attitude_error_rad(solution.quaternion, q_true) * ARCSEC
    return {"attitude_error_arcsec": err,
            "note": "validation only; not an estimator output"}
