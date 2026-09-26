"""星敏感器模拟模块测试。

无 pytest 环境下可直接运行：PYTHONPATH=src python3 tests/test_star_sensor.py
"""
import math
import random
import sys

from orbitforge.core.vector import Vec3
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.starsensor import (
    StarCatalog, CameraParams, StarSensorSim,
    project_stars, render_image, extract_centroids, centroid_to_body,
    quat_to_dcm, dcm_to_quat, boresight_toward, triad, wahba, attitude_error_rad,
)

ARCSEC = 206264.80624709636
_CAM = CameraParams(n_pixels=320, fov_deg=12.0)
_CAT = StarCatalog.random(n_stars=4000, seed=7)
_SIM = None


def get_sim():
    global _SIM
    if _SIM is None:
        _SIM = StarSensorSim(catalog=_CAT, camera=_CAM, mag_limit=6.0, seed=11)
    return _SIM


def random_attitude(rng):
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    return Quaternion(
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    )


def random_vec(rng):
    z = rng.uniform(-1.0, 1.0)
    p = rng.uniform(0.0, 2 * math.pi)
    r = math.sqrt(1 - z * z)
    return Vec3(r * math.cos(p), r * math.sin(p), z)


def test_dcm_quat_roundtrip():
    rng = random.Random(1)
    for _ in range(10):
        q = random_attitude(rng)
        assert attitude_error_rad(dcm_to_quat(quat_to_dcm(q)), q) < 1e-12


def test_wahba_recovers_noiseless_attitude():
    rng = random.Random(2)
    for _ in range(10):
        refs = [random_vec(rng) for _ in range(8)]
        qt = random_attitude(rng)
        bods = [qt.rotate(r) for r in refs]
        assert attitude_error_rad(wahba(bods, refs), qt) < 1e-6


def test_triad_recovers_noiseless_attitude():
    rng = random.Random(3)
    refs = [random_vec(rng) for _ in range(2)]
    qt = random_attitude(rng)
    bods = [qt.rotate(r) for r in refs]
    assert attitude_error_rad(triad(bods[0], bods[1], refs[0], refs[1]), qt) < 1e-9


def test_centroid_recovers_star_position():
    sim = get_sim()
    star = min(_CAT.stars, key=lambda s: s.mag)
    q = boresight_toward(star.unit)
    proj = project_stars(_CAT, q, _CAM)
    p = next(p for p in proj if p.star_id == star.star_id)
    img, _ = render_image(proj, _CAM, random.Random(5))
    cents, _, _ = extract_centroids(img)
    c = min(cents, key=lambda c: math.hypot(c.px - p.px, c.py - p.py))
    assert math.hypot(c.px - p.px, c.py - p.py) < 0.5
    ang = centroid_to_body(c, _CAM).angle(q.rotate(star.unit)) * ARCSEC
    assert ang < 60.0


def test_pipeline_nominal_recovers_attitude():
    sim = get_sim()
    rng = random.Random(20)
    for k in range(3):
        r = sim.run(random_attitude(rng), seed=1000 + k)
        if r.estimated_attitude is None:
            continue  # 稀疏天区允许失败，见失败用例
        assert r.attitude_error_arcsec < 60.0
        assert r.n_inliers >= 3
        assert r.residual_rms_arcsec > 0.0
        assert len(r.residuals_arcsec) == r.n_inliers
        assert r.residual_rms_arcsec <= r.residual_max_arcsec + 1e-9
        return
    raise AssertionError('3 次标称运行全部失败，链路异常')


def test_estimate_is_computed_not_copied_from_truth():
    sim = get_sim()
    rng = random.Random(21)
    for k in range(5):
        r = sim.run(random_attitude(rng), seed=1100 + k)
        if r.estimated_attitude is None:
            continue
        est, tru = r.estimated_attitude, r.true_attitude
        assert est is not tru
        assert (est.w, est.x, est.y, est.z) != (tru.w, tru.x, tru.y, tru.z)
        assert 1e-6 < r.attitude_error_arcsec < 60.0
        return
    raise AssertionError('未获得成功的定姿运行')


def test_sparse_catalog_failure_returns_none_not_truth():
    sim_sparse = StarSensorSim(catalog=_CAT, camera=_CAM, mag_limit=4.0, seed=11)
    rng = random.Random(22)
    failures = 0
    for k in range(4):
        r = sim_sparse.run(random_attitude(rng), n_fake_stars=4, seed=1200 + k)
        if r.estimated_attitude is None:
            failures += 1
            assert r.attitude_error_arcsec is None
            assert any('定姿失败' in m for m in r.messages)
    assert failures >= 1  # 稀疏匹配库下必须出现明确失败，而不是回退真姿态


def test_false_matches_are_rejected():
    sim = get_sim()
    rng = random.Random(23)
    for k in range(5):
        r = sim.run(random_attitude(rng), n_fake_stars=6, seed=1300 + k)
        if r.estimated_attitude is None:
            continue
        assert r.n_fake_injected == 6
        assert r.n_outliers + r.n_unidentified >= 1  # 假星被识别为离群/未识别源
        assert r.attitude_error_arcsec < 60.0        # 且不影响最终定姿精度
        return
    raise AssertionError('含假星运行全部失败，离群剔除异常')


def test_magnitude_threshold_changes_match_catalog():
    sim6 = get_sim()
    sim5 = StarSensorSim(catalog=_CAT, camera=_CAM, mag_limit=5.0, seed=11)
    assert len(sim5.match_catalog) < len(sim6.match_catalog)
    rng = random.Random(24)
    q = random_attitude(rng)
    r6 = sim6.run(q, seed=1400)
    r5 = sim5.run(q, seed=1400)
    assert r5.n_dim_excluded > r6.n_dim_excluded
    n6 = r6.n_inliers if r6.estimated_attitude else 0
    n5 = r5.n_inliers if r5.estimated_attitude else 0
    assert n5 <= n6


def test_edge_star_is_flagged_and_absorbed():
    sim = get_sim()
    star = min(_CAT.stars, key=lambda s: s.mag)
    ref = Vec3(0.0, 0.0, 1.0) if abs(star.unit.z) < 0.9 else Vec3(1.0, 0.0, 0.0)
    perp = star.unit.cross(ref).unit()
    theta = math.atan((0.5 * _CAM.n_pixels - 3.0) / _CAM.focal_px)
    boresight = (star.unit * math.cos(theta) + perp * math.sin(theta)).unit()
    r = sim.run(boresight_toward(boresight), seed=1500)
    assert r.n_edge_truncated >= 1
    assert any('边缘' in m for m in r.messages)


def _main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL {name}: {e}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
