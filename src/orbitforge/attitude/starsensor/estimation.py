"""Wahba-problem attitude determination and match verification.

Given measured body-frame unit vectors b_i and matched inertial catalog
unit vectors r_i, weights w_i, solve

    min  sum_i w_i | b_i - A r_i |^2

for the rotation A (Davenport's q-method: largest eigenvector of the
4x4 K matrix, eigen-decomposed with Jacobi rotations - no numpy needed).

Then the solution is verified observation by observation. Any match whose
post-fit residual exceeds a robust threshold is reported as a rejected
(false) match and the attitude is re-solved without it. Nothing here ever
sees the true attitude; rejection is driven solely by fit residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from orbitforge.attitude.quaternion import Quaternion
from .rotations import quat_to_dcm, dcm_apply, vdot


@dataclass
class AttitudeEstimate:
    quaternion: Quaternion
    dcm: tuple
    weights: list
    kept: list          # indices into the input match lists
    rejected: list      # list of (index, residual_rad) - failed verification
    residuals_rad: list  # aligned with kept
    covariance_rad2: tuple  # 3x3 small-angle covariance in the body frame
    rms_residual_rad: float


# ---------------------------------------------------------------- Jacobi 3x3/4x4

def _sym_eig(a):
    """Eigen-decomposition of a small symmetric matrix by cyclic Jacobi sweeps.

    Returns (eigenvalues, eigenvectors-as-columns).
    """
    n = len(a)
    a = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(60):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += a[p][q] * a[p][q]
        if off < 1e-26:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-16:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    if k != p and k != q:
                        akp, akq = a[k][p], a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[p][k] = a[k][p]
                        a[k][q] = s * akp + c * akq
                        a[q][k] = a[k][q]
                app, aqq, apq0 = a[p][p], a[q][q], a[p][q]
                a[p][p] = c * c * app - 2.0 * s * c * apq0 + s * s * aqq
                a[q][q] = s * s * app + 2.0 * s * c * apq0 + c * c * aqq
                a[p][q] = (c * c - s * s) * apq0 + s * c * (app - aqq)
                a[q][p] = a[p][q]
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


# ---------------------------------------------------------------- Wahba solver

def solve_wahba(body_vecs, inertial_vecs, weights):
    """Optimal quaternion (scalar-first, body-from-inertial)."""
    B = [[0.0, 0.0, 0.0] for _ in range(3)]
    for (bx, by, bz), (rx, ry, rz), w in zip(body_vecs, inertial_vecs, weights):
        B[0][0] += w * bx * rx; B[0][1] += w * bx * ry; B[0][2] += w * bx * rz
        B[1][0] += w * by * rx; B[1][1] += w * by * ry; B[1][2] += w * by * rz
        B[2][0] += w * bz * rx; B[2][1] += w * bz * ry; B[2][2] += w * bz * rz
    sig = B[0][0] + B[1][1] + B[2][2]
    # Sign of z is flipped relative to the common Davenport convention:
    # the codebase quaternion (orbitforge.attitude.quaternion) uses the
    # opposite cross-product term in its DCM, equivalent to K built with B^T.
    z1 = B[2][1] - B[1][2]
    z2 = B[0][2] - B[2][0]
    z3 = B[1][0] - B[0][1]
    S = [[B[i][j] + B[j][i] for j in range(3)] for i in range(3)]
    K = [
        [S[0][0] - sig, S[0][1], S[0][2], z1],
        [S[1][0], S[1][1] - sig, S[1][2], z2],
        [S[2][0], S[2][1], S[2][2] - sig, z3],
        [z1, z2, z3, sig],
    ]
    vals, vecs = _sym_eig(K)
    k = max(range(4), key=lambda i: vals[i])
    # K uses scalar-last (x, y, z, w); convert to the codebase scalar-first
    # quaternion (w, x, y, z).
    q = Quaternion(vecs[3][k], vecs[0][k], vecs[1][k], vecs[2][k]).normalized()
    return q, vals[k]


# ---------------------------------------------------------------- covariance

def _inv3_safe(m):
    """Inverse of a symmetric positive-(semi)definite 3x3; zero block if singular."""
    a, b, c = m[0]
    _, d, e = m[1]
    _, _, f = m[2]
    det = (a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c))
    if abs(det) < 1e-30 * max(1.0, a * d * f):
        pinv = [[0.0] * 3 for _ in range(3)]
        return pinv, True
    inv = [
        [(d * f - e * e), (c * e - b * f), (b * e - c * d)],
        [(c * e - b * f), (a * f - c * c), (b * c - a * e)],
        [(b * e - c * d), (b * c - a * e), (a * d - b * b)],
    ]
    return [[x / det for x in row] for row in inv], False


def attitude_covariance(body_vecs, weights):
    """Fisher information sum w_i (I - b_i b_i^T); small-angle covariance."""
    H = [[0.0] * 3 for _ in range(3)]
    for b, w in zip(body_vecs, weights):
        for i in range(3):
            for j in range(3):
                H[i][j] += w * ((1.0 if i == j else 0.0) - b[i] * b[j])
    return _inv3_safe(H)


def residual_rad(A, body, inertial):
    pred = dcm_apply(A, inertial)
    return math.acos(max(-1.0, min(1.0, vdot(body, pred))))


def robust_scale_z(standardized, floor=1.0, cap=2.5):
    """Scale for standardized-residual rejection, bounded against masking.

    Good matches have z ~ N(0, 1) by construction, so the rejection scale
    belongs near 1. The MAD may raise it slightly to tolerate an
    underestimated noise model, but it is CAPPED: an unbounded MAD scale
    lets one gross outlier inflate the fit scatter until its own z falls
    below the cutoff (masking).
    """
    if not standardized:
        return floor
    s = sorted(abs(z) for z in standardized)
    mad = 1.4826 * s[len(s) // 2]
    return min(cap, max(floor, mad))


def estimate_with_rejection(body_vecs, inertial_vecs, sigmas_rad,
                            reject_sigma=4.0, max_iter=5):
    """Solve Wahba, reject matches with excessive *standardized* residuals.

    Each match's post-fit residual is divided by its own measurement sigma
    before the outlier test, so a dim star with a large but statistically
    consistent residual is kept (weighted Wahba already downweights it),
    while a wrong identity - residual far beyond any measurement sigma - is
    rejected regardless of star brightness. A match is only dropped while at
    least two matches remain.
    """
    n = len(body_vecs)
    weights = [1.0 / max(s, 1e-9) ** 2 for s in sigmas_rad]
    active = list(range(n))
    rejected = []
    for _ in range(max_iter):
        bv = [body_vecs[i] for i in active]
        rv = [inertial_vecs[i] for i in active]
        wv = [weights[i] for i in active]
        q, _gain = solve_wahba(bv, rv, wv)
        A = quat_to_dcm(q)
        res = [residual_rad(A, body_vecs[i], inertial_vecs[i]) for i in active]
        z = [r / max(sigmas_rad[i], 1e-9) for i, r in zip(active, res)]
        cutoff = reject_sigma * robust_scale_z(z)
        drop = [k for k, zz in zip(active, z) if zz > cutoff]
        if not drop:
            break
        if len(drop) > len(active) - 2:
            # Masking case: one (or few) gross outliers corrupted the fit so
            # badly that everything looks bad. Trim only the single worst
            # and re-solve, never dropping below two matches.
            zi = dict(zip(active, z))
            drop = [max(active, key=lambda k: zi[k])]
        for k in drop:
            rejected.append((k, residual_rad(A, body_vecs[k], inertial_vecs[k])))
        drop_set = set(drop)
        active = [k for k in active if k not in drop_set]

    # Final solve on the surviving set.
    bv = [body_vecs[i] for i in active]
    rv = [inertial_vecs[i] for i in active]
    wv = [weights[i] for i in active]
    q, _gain = solve_wahba(bv, rv, wv)
    A = quat_to_dcm(q)
    res = [residual_rad(A, body_vecs[i], inertial_vecs[i]) for i in active]
    cov, singular = attitude_covariance(bv, wv)
    rms = math.sqrt(sum(r * r for r in res) / len(res)) if res else math.inf
    return AttitudeEstimate(
        quaternion=q, dcm=A, weights=wv, kept=active,
        rejected=sorted(rejected, key=lambda t: t[0]),
        residuals_rad=res, covariance_rad2=cov, rms_residual_rad=rms,
    )
