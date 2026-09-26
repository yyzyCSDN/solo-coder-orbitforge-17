"""Star sensor simulation demo: imaging -> identification -> attitude.

Run:  PYTHONPATH=src python3 demo_star_sensor.py

Covers the required explicit behaviors:
  * FOV-edge stars are flagged from the image and excluded from the fit
  * the magnitude threshold is applied and counted explicitly
  * false matches are rejected and reported, never silently absorbed
  * the estimator never sees the true attitude; truth is used afterwards
    (validation only), and unobservable scenes yield LOST_IN_SPACE with
    quaternion=None rather than any fabricated attitude
"""
from __future__ import annotations

import math
import random

from orbitforge.attitude.starsensor import (
    CameraModel, StarCatalog, StarSensorSim, validate,
    quat_from_two_vectors, random_quaternion,
)

ARCSEC = 206264.80624709636


def ascii_frame(image, n, out=48):
    """Downsampled ASCII rendering of the focal plane (block means)."""
    chars = " .:-=+*#%@"
    bg = sorted(image)[len(image) // 2]
    step = n // out
    lines = []
    for j in range(0, n - step + 1, step):
        row = ""
        for i in range(0, n - step + 1, step):
            tot = cnt = 0
            for jj in range(j, j + step):
                for ii in range(i, i + step):
                    tot += image[jj * n + ii]
                    cnt += 1
            mean = tot / cnt
            level = min(len(chars) - 1, int((mean - bg) / max(1.0, bg) * 6))
            row += chars[max(0, level)]
        lines.append(row)
    return "\n".join(lines)


def report(tag, sol, q_true):
    print(f"  status            : {sol.status}  ({sol.detail})")
    print(f"  detected spots    : {sol.n_detected}"
          f"   edge-excluded: {sol.n_edge_excluded}"
          f"   unmatched: {sol.n_unmatched}")
    if sol.quaternion is None:
        print("  attitude          : NONE (no solution emitted)")
        return
    q = sol.quaternion
    print(f"  attitude q (wxyz) : [{q.w:+.6f} {q.x:+.6f} {q.y:+.6f} {q.z:+.6f}]")
    print(f"  matches used      : {len(sol.matches)}"
          f"   rms residual: {sol.rms_residual_arcsec:.1f} asec")
    print(f"  formal 1-sigma    : [{sol.sigma_body_arcsec[0]:.1f},"
          f" {sol.sigma_body_arcsec[1]:.1f},"
          f" {sol.sigma_body_arcsec[2]:.1f}] asec (body x,y,z)")
    for m in sol.matches:
        print(f"    det {m.detection_index:2d} -> star {m.catalog_index:5d}"
              f"   residual {m.residual_arcsec:7.1f} asec")
    for r in sol.rejected_matches:
        print(f"    REJECTED det {r.detection_index:2d} -> star {r.catalog_index:5d}"
              f"   ({r.reason})")
    err = validate(sol, q_true)["attitude_error_arcsec"]
    print(f"  [validation only] error vs true attitude: {err:.1f} asec")


def main():
    rng = random.Random(2026)
    catalog = StarCatalog.synthetic(n_stars=3000, seed=20260926)
    cam = CameraModel()
    sim = StarSensorSim(catalog, cam, seed=42)
    print(f"camera: {cam.n_pixels}px, fov {cam.fov_deg} deg, "
          f"{cam.arcsec_per_pixel:.1f} asec/px, limiting mag "
          f"{cam.limiting_magnitude}")

    # ---------------------------------------------------------- 1. nominal
    print("\n=== 1. nominal scene (3 spurious blobs, 40 hot pixels) ===")
    q_true = random_quaternion(rng)
    img, info = sim.render(q_true, n_spurious=3, n_hot_pixels=40)
    print(ascii_frame(img, cam.n_pixels))
    print(f"  rendered stars: {info.n_rendered}  spurious blobs: "
          f"{len(info.spurious_uv)}  hot pixels: {len(info.hot_pixel_uv)}")
    sol = sim.estimate(img)
    report("nominal", sol, q_true)

    # ------------------------------------------------- 2. FOV edge handling
    print("\n=== 2. brightest star parked in the edge band ===")
    bright = min(catalog.stars, key=lambda s: s.mag)
    u_edge = cam.edge_margin_px + 0.5
    q_edge = quat_from_two_vectors(bright.unit,
                                   cam.unproject(u_edge, cam.center_px))
    img2, info2 = sim.render(q_edge)
    rec = next(r for r in info2.records if r.catalog_index == bright.index)
    print(f"  star {bright.index} (mag {bright.mag:.2f}) at u={rec.u:.1f}px: "
          f"rendered={rec.rendered}, edge_clipped={rec.edge_clipped}")
    sol2 = sim.estimate(img2)
    report("edge", sol2, q_edge)
    assert bright.index not in {m.catalog_index for m in sol2.matches}
    print(f"  -> star {bright.index} explicitly excluded from the fit")

    # ------------------------------------------- 3. magnitude threshold
    print("\n=== 3. limiting magnitude 5.5 (tighter detection limit) ===")
    cam_dim = CameraModel(limiting_magnitude=5.5)
    sim_dim = StarSensorSim(catalog, cam_dim, seed=43)
    img3, info3 = sim_dim.render(q_true)  # same attitude as scenario 1
    print(f"  in-FOV candidates : {info3.n_rendered + info3.n_skipped_magnitude}")
    print(f"  rendered (<=5.5)  : {info3.n_rendered}")
    print(f"  skipped (>5.5)    : {info3.n_skipped_magnitude}"
          f"   [explicitly counted, never imaged]")
    sol3 = sim_dim.estimate(img3)
    report("dim", sol3, q_true)

    # ------------------------------------------- 4. lost in space (no stars)
    print("\n=== 4. no catalog stars (limiting mag -2.0), 4 glints only ===")
    sim_blind = StarSensorSim(catalog, CameraModel(limiting_magnitude=-2.0),
                              seed=44)
    img4, info4 = sim_blind.render(q_true, n_spurious=4)
    print(f"  catalog stars rendered: {info4.n_rendered},"
          f" spurious blobs: {len(info4.spurious_uv)}")
    sol4 = sim_blind.estimate(img4)
    report("lost", sol4, q_true)
    print(f"  [validation only] validate() -> "
          f"{validate(sol4, q_true)['attitude_error_arcsec']}"
          " (inf: nothing to validate)")

    # ------------------------------------------- 5. false match rejection
    print("\n=== 5. one deliberately corrupted association ===")
    from orbitforge.attitude.starsensor.estimation import (
        estimate_with_rejection)
    from orbitforge.attitude.starsensor.rotations import quat_to_dcm, dcm_apply
    rng5 = random.Random(5)
    q5 = random_quaternion(rng5)
    A5 = quat_to_dcm(q5)
    inert, body, sig = [], [], []
    for _ in range(8):
        z = rng5.uniform(math.cos(math.radians(8)), 1.0)
        lam = rng5.uniform(0, 2 * math.pi)
        r = math.sqrt(1 - z * z)
        rv = (r * math.cos(lam), r * math.sin(lam), z)
        b0 = dcm_apply(A5, rv)
        b = tuple(b0[i] + rng5.gauss(0, 1.5e-5) for i in range(3))
        n = math.sqrt(sum(x * x for x in b))
        inert.append(rv)
        body.append(tuple(x / n for x in b))
        sig.append(1.5e-5)
    inert[3] = (1.0, 0.0, 0.0)  # wrong identity for observation #3
    est = estimate_with_rejection(body, inert, sig)
    print(f"  rejected matches  : {est.rejected}  (index, residual_rad)")
    from orbitforge.attitude.starsensor.rotations import attitude_error_rad
    err5 = attitude_error_rad(est.quaternion, q5) * ARCSEC
    print(f"  attitude error after rejection: {err5:.2f} asec")


if __name__ == "__main__":
    main()
