"""Tests for the star sensor simulation (pytest-compatible, also runnable:

    PYTHONPATH=src python tests/test_star_sensor.py
)
"""
from __future__ import annotations

import inspect
import math
import random

from orbitforge.attitude.quaternion import Quaternion
from orbitforge.core.vector import Vec3
from orbitforge.attitude.starsensor import (
    CameraModel, CatalogStar, StarCatalog, StarSensorSim,
    STATUS_OK, STATUS_DEGRADED, STATUS_INSUFFICIENT_STARS, STATUS_LOST_IN_SPACE,
    detect_spots, dcm_apply, dcm_to_quat, quat_to_dcm, random_quaternion,
    quat_from_two_vectors, solve_wahba, validate, estimate_with_rejection,
)

ARCSEC = 206264.80624709636


# Shared deterministic scene catalog; the pair DB (dominant cost) is built
# once and injected into every sim so each test gets an independent RNG.
CATALOG = StarCatalog.synthetic(n_stars=3000, seed=20260926)
CAM = CameraModel()
PDB = CATALOG.build_pair_database(
    max_sep_rad=math.radians(CAM.fov_deg) * math.sqrt(2.0) * 1.05,
    mag_limit=CAM.limiting_magnitude)


def make_sim(seed, camera=CAM):
    # The shared DB is only valid for the default camera's magnitude limit.
    pdb = PDB if camera is CAM else None
    return StarSensorSim(CATALOG, camera, seed=seed, pair_db=pdb)


# ------------------------------------------------------------- math conventions

def test_dcm_matches_codebase_quaternion_and_roundtrip():
    rng = random.Random(1)
    v = (0.31, -0.72, 0.62)
    for _ in range(20):
        q = random_quaternion(rng)
        A = quat_to_dcm(q)
        via_dcm = dcm_apply(A, v)
        via_q = q.rotate(Vec3(*v)).as_tuple()
        assert max(abs(a - b) for a, b in zip(via_dcm, via_q)) < 1e-12
        q2 = dcm_to_quat(A)
        A2 = quat_to_dcm(q2)
        assert max(abs(A[i][j] - A2[i][j]) for i in range(3) for j in range(3)) < 1e-10


def test_wahba_exact_recovery_and_against_noise():
    rng = random.Random(2)
    for _trial in range(5):
        q_true = random_quaternion(rng)
        A = quat_to_dcm(q_true)
        inert, body, sig = [], [], []
        for _ in range(8):
            r = _random_unit_in_cone(rng, (0.0, 0.0, 1.0), math.radians(8))
            b0 = dcm_apply(A, r)
            b = (b0[0] + rng.gauss(0, 2e-5), b0[1] + rng.gauss(0, 2e-5),
                 b0[2] + rng.gauss(0, 2e-5))
            n = math.sqrt(sum(x * x for x in b))
            inert.append(r)
            body.append(tuple(x / n for x in b))
            sig.append(2e-5)
        est = estimate_with_rejection(body, inert, sig, reject_sigma=4.0)
        # Exact-noise-free case first:
        body_exact = [dcm_apply(A, r) for r in inert]
        est0 = estimate_with_rejection(body_exact, inert, [1.0] * len(inert))
        from orbitforge.attitude.starsensor.rotations import attitude_error_rad
        assert attitude_error_rad(est0.quaternion, q_true) < 1e-6  # float64 floor
        assert attitude_error_rad(est.quaternion, q_true) < 3e-4
        assert not est.rejected


# ------------------------------------------------------------------ camera model

def test_camera_project_unproject_roundtrip():
    rng = random.Random(3)
    for _ in range(20):
        u = rng.uniform(0, CAM.n_pixels - 1)
        v = rng.uniform(0, CAM.n_pixels - 1)
        b = CAM.unproject(u, v)
        u2, v2 = CAM.project(b)
        assert abs(u - u2) < 1e-9 and abs(v - v2) < 1e-9


# ------------------------------------------------------------- image generation

def test_magnitude_threshold_is_explicit():
    """Stars above limiting magnitude are counted and never rendered."""
    cam = CameraModel(limiting_magnitude=4.0)
    sim = StarSensorSim(CATALOG, cam, seed=4)
    q = random_quaternion(random.Random(10))
    img, info = sim.render(q)

    A = quat_to_dcm(q)
    n, m = cam.n_pixels, cam.edge_margin_px
    expect_rendered, expect_mag_skip = set(), 0
    for s in CATALOG.stars:
        proj = cam.project(dcm_apply(A, s.unit))
        if proj is None:
            continue
        u, v = proj
        if not (-m <= u < n + m and -m <= v < n + m):
            continue
        if s.mag > 4.0:
            expect_mag_skip += 1
        else:
            expect_rendered.add(s.index)

    assert info.n_skipped_magnitude == expect_mag_skip
    assert {r.catalog_index for r in info.records if r.rendered} == expect_rendered
    assert all(r.mag <= 4.0 for r in info.records if r.rendered)
    assert info.n_rendered == len(expect_rendered)


def test_centroids_track_true_centers():
    """Sub-pixel centroiding: offsets are consistent with the noise model."""
    sim = make_sim(101)
    q = random_quaternion(random.Random(11))
    img, info = sim.render(q)
    dets, _ = detect_spots(img, CAM)
    ratios = []
    for r in info.visible_records():
        if r.edge_clipped or not r.inside_sensor:
            continue
        near = min(dets, key=lambda d: (d.u - r.u) ** 2 + (d.v - r.v) ** 2)
        du = math.hypot(near.u - r.u, near.v - r.v)
        if du < 2.0:
            ratios.append(du / near.sigma_px)
    assert len(ratios) >= 5, f"only {len(ratios)} true spots recovered"
    ratios.sort()
    # 2-D offset of a 1-sigma-per-axis centroid has median ~1.18 sigma.
    assert ratios[len(ratios) // 2] < 2.0, ratios
    assert ratios[-1] < 5.0, ratios


def test_edge_stars_flagged_from_image_and_excluded_from_attitude():
    """A star parked in the edge band is flagged by the CENTROIDER (image-side),
    excluded from the attitude solution, and the exclusion is reported."""
    bright = min(CATALOG.stars, key=lambda s: s.mag)
    u_t = CAM.edge_margin_px + 0.5
    sim = make_sim(102)
    q = quat_from_two_vectors(bright.unit, CAM.unproject(u_t, CAM.center_px))
    img, info = sim.render(q)
    assert any(r.catalog_index == bright.index and r.edge_clipped
               for r in info.records)

    dets, _ = detect_spots(img, CAM)
    near = min(dets, key=lambda d: (d.u - u_t) ** 2 + (d.v - CAM.center_px) ** 2)
    assert math.hypot(near.u - u_t, near.v - CAM.center_px) < 3.0
    assert near.edge_clipped, "centroider must flag the truncated PSF"

    sol = sim.estimate(img)
    assert sol.status in (STATUS_OK, STATUS_DEGRADED)
    assert sol.n_edge_excluded >= 1
    assert bright.index not in {m.catalog_index for m in sol.matches}
    err = validate(sol, q)["attitude_error_arcsec"]
    # The estimate must be consistent with its OWN formal uncertainty
    # (this scene is dim-star dominated, so the honest sigma is large).
    assert err < 3.0 * max(sol.sigma_body_arcsec), (err, sol.sigma_body_arcsec)
    assert err < 300.0, err


# --------------------------------------------------------------- identification

def test_end_to_end_recovery_with_spurious_blobs_and_hot_pixels():
    sim = make_sim(103)
    q_true = random_quaternion(random.Random(12))
    img, info = sim.render(q_true, n_spurious=3, n_hot_pixels=40)
    sol = sim.estimate(img)

    assert sol.status == STATUS_OK, sol.detail
    assert len(sol.matches) >= 4
    # The three non-catalog blobs must not obtain catalog identities: they
    # end up unmatched or explicitly rejected as group-inconsistent.
    dets, _ = detect_spots(img, CAM)
    spurious_det = set()
    for (su, sv) in info.spurious_uv:
        k = min(range(len(dets)),
                key=lambda i: (dets[i].u - su) ** 2 + (dets[i].v - sv) ** 2)
        spurious_det.add(k)
    assert {m.detection_index for m in sol.matches}.isdisjoint(spurious_det)
    # Non-catalog blobs end up unmatched (no catalog prediction near them)
    # or explicitly rejected by the residual screen.
    n_resid_rejected = len(sol.rejected_matches)
    assert sol.n_unmatched + n_resid_rejected >= 3, \
        (sol.n_unmatched, n_resid_rejected)

    # Residual scatter must be unit-noise: standardized residual rms < 2.
    z = [m.residual_arcsec /
         (dets[m.detection_index].sigma_px / CAM.focal_px * ARCSEC)
         for m in sol.matches]
    z_rms = math.sqrt(sum(v * v for v in z) / len(z))
    assert z_rms < 2.0, z_rms
    err = validate(sol, q_true)["attitude_error_arcsec"]
    # The achieved error must be compatible with the tracker's own formal
    # uncertainty (this field happens to be dim-star dominated).
    assert err < 3.0 * max(sol.sigma_body_arcsec), \
        (err, sol.sigma_body_arcsec)


def test_nominal_accuracy_over_many_pointings():
    """Good fields reach ~arcsec class; every solution is consistent with
    its own reported covariance (the honesty property that matters)."""
    errs_good = []
    n_solved = 0
    for seed in range(12):
        sim = StarSensorSim(CATALOG, CAM, seed=200 + seed, pair_db=PDB)
        q_true = random_quaternion(random.Random(600 + seed))
        img, _ = sim.render(q_true, n_spurious=1, n_hot_pixels=20)
        sol = sim.estimate(img)
        if sol.quaternion is None:
            continue
        n_solved += 1
        err = validate(sol, q_true)["attitude_error_arcsec"]
        # The reported formal uncertainty must bracket the achieved error.
        assert err < 3.5 * max(sol.sigma_body_arcsec), \
            (seed, err, sol.sigma_body_arcsec)
        if len(sol.matches) >= 5:
            errs_good.append(err)
    assert n_solved >= 8, f"only {n_solved}/12 fields solved"
    # Well-observed fields (>= 5 matches) are arcsec-class accurate.
    errs_good.sort()
    assert errs_good, "no well-observed fields"
    assert errs_good[len(errs_good) // 2] < 30.0, errs_good


def test_false_match_is_rejected_by_residual_screen():
    """One deliberately wrong association among 8 good ones gets caught."""
    rng = random.Random(5)
    q_true = random_quaternion(rng)
    A = quat_to_dcm(q_true)
    inert, body, sig = [], [], []
    for _ in range(8):
        r = _random_unit_in_cone(rng, (0, 0, 1), math.radians(8))
        b0 = dcm_apply(A, r)
        b = tuple(b0[i] + rng.gauss(0, 1.5e-5) for i in range(3))
        n = math.sqrt(sum(x * x for x in b))
        inert.append(r)
        body.append(tuple(x / n for x in b))
        sig.append(1.5e-5)
    # Corrupt association #3: point it at a random unrelated direction.
    bad = _random_unit_in_cone(rng, (1, 0, 0), 0.1)
    inert[3] = bad

    est = estimate_with_rejection(body, inert, sig)
    assert [k for k, _ in est.rejected] == [3]
    assert est.residuals_rad and all(r * ARCSEC < 30 for r in est.residuals_rad)
    from orbitforge.attitude.starsensor.rotations import attitude_error_rad
    assert attitude_error_rad(est.quaternion, q_true) * ARCSEC < 10.0


def test_spurious_only_scene_is_lost_not_attitude():
    """With no catalog-visible stars the estimator must NOT emit an attitude."""
    empty_cam = CameraModel(limiting_magnitude=-2.0)  # catalog starts at -1.5
    sim = make_sim(104, camera=empty_cam)
    q_true = random_quaternion(random.Random(13))
    img, info = sim.render(q_true, n_spurious=2)
    sol = sim.estimate(img)
    assert sol.status in (STATUS_INSUFFICIENT_STARS, STATUS_LOST_IN_SPACE)
    assert sol.quaternion is None, "no truth / no fabricating attitudes"
    assert validate(sol, q_true)["attitude_error_arcsec"] == math.inf


def test_two_star_scene_reported_insufficient():
    two = StarCatalog([
        CatalogStar(0, 1.0, (0.05, 0.0, 0.9987)),
        CatalogStar(1, 1.2, (-0.05, 0.02, 0.9985)),
        CatalogStar(2, 2.0, (-0.9, 0.1, 0.1)),
    ])
    sim = StarSensorSim(two, CameraModel(limiting_magnitude=3.0), seed=8)
    img, _ = sim.render(Quaternion(1.0, 0.0, 0.0, 0.0))
    sol = sim.estimate(img)
    assert sol.status == STATUS_INSUFFICIENT_STARS
    assert sol.quaternion is None
    assert "need >= 3" in sol.detail


def test_estimator_cannot_see_truth():
    """Signature check plus a tamper test: estimate follows the image only."""
    params = inspect.signature(StarSensorSim.estimate).parameters
    assert "q_true" not in params and "truth" not in params and "attitude" not in params

    sim = make_sim(105)
    q_true = random_quaternion(random.Random(14))
    img, _ = sim.render(q_true)
    sol = sim.estimate(img)
    assert sol.quaternion is not None, sol.detail
    assert validate(sol, q_true)["attitude_error_arcsec"] \
        < 3.0 * max(sol.sigma_body_arcsec)
    # A different claimed "truth" must produce a large disagreement - i.e.
    # the answer was not copied from any pre-existing attitude.
    other = random_quaternion(random.Random(99))
    assert validate(sol, other)["attitude_error_arcsec"] > 0.5 * 3600.0


def _random_unit_in_cone(rng, axis, half_angle):
    cos_a = math.cos(half_angle)
    z = rng.uniform(cos_a, 1.0)
    lam = rng.uniform(0, 2 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    local = (r * math.cos(lam), r * math.sin(lam), z)
    az = math.atan2(axis[1], axis[0])
    ca, sa = math.cos(az), math.sin(az)
    x = ca * local[0] - sa * local[1]
    y = sa * local[0] + ca * local[1]
    n = math.sqrt(x * x + y * y + local[2] ** 2)
    return (x / n, y / n, local[2] / n)


# --------------------------------------------------------------- standalone run

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")
           and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
