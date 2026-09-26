"""Direction-cosine-matrix helpers for the star sensor pipeline.

Convention (matches orbitforge.attitude.quaternion.Quaternion):
the attitude quaternion *q* is the body-from-inertial rotation, so an
inertial catalog unit vector ``r`` is seen in the tracker body frame as

    b = A(q) r

with q stored scalar-first ``(w, x, y, z)`` and active (vector) rotation.
"""
from __future__ import annotations

import math

from orbitforge.attitude.quaternion import Quaternion


def quat_to_dcm(q: Quaternion):
    """Rotation matrix A such that b = A r (tuple-of-tuples, row major)."""
    w, x, y, z = q.w, q.x, q.y, q.z
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def dcm_to_quat(A) -> Quaternion:
    """Shepperd's method; exact inverse of :func:`quat_to_dcm` for proper matrices."""
    tr = A[0][0] + A[1][1] + A[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0  # s = 4 w
        w = 0.25 * s
        x = (A[2][1] - A[1][2]) / s
        y = (A[0][2] - A[2][0]) / s
        z = (A[1][0] - A[0][1]) / s
    elif A[0][0] > A[1][1] and A[0][0] > A[2][2]:
        s = math.sqrt(1.0 + A[0][0] - A[1][1] - A[2][2]) * 2.0
        w = (A[2][1] - A[1][2]) / s
        x = 0.25 * s
        y = (A[1][0] + A[0][1]) / s
        z = (A[0][2] + A[2][0]) / s
    elif A[1][1] > A[2][2]:
        s = math.sqrt(1.0 + A[1][1] - A[0][0] - A[2][2]) * 2.0
        w = (A[0][2] - A[2][0]) / s
        x = (A[1][0] + A[0][1]) / s
        y = 0.25 * s
        z = (A[2][1] + A[1][2]) / s
    else:
        s = math.sqrt(1.0 + A[2][2] - A[0][0] - A[1][1]) * 2.0
        w = (A[1][0] - A[0][1]) / s
        x = (A[0][2] + A[2][0]) / s
        y = (A[2][1] + A[1][2]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return Quaternion(w / n, x / n, y / n, z / n)


def dcm_apply(A, v):
    return (
        A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
    )


def vdot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vnorm(v) -> float:
    return math.sqrt(vdot(v, v))


def vunit(v):
    n = vnorm(v)
    return (v[0] / n, v[1] / n, v[2] / n)


def quat_from_two_vectors(a, b) -> Quaternion:
    """Minimal-rotation quaternion mapping unit vector *a* onto unit vector *b*."""
    d = max(-1.0, min(1.0, vdot(a, b)))
    if d > 1.0 - 1e-12:
        return Quaternion(1.0, 0.0, 0.0, 0.0)
    if d < -1.0 + 1e-12:
        ref = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = vunit(vcross(a, ref))
        return Quaternion(0.0, axis[0], axis[1], axis[2])
    axis = vunit(vcross(a, b))
    half = 0.5 * math.acos(d)
    s = math.sin(half)
    return Quaternion(math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)


def random_quaternion(rng) -> Quaternion:
    """Uniformly distributed unit quaternion (Gaussian 4-vector, normalized)."""
    return Quaternion(
        rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0),
        rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0),
    ).normalized()


def attitude_error_rad(q_est: Quaternion, q_true: Quaternion) -> float:
    """Smallest rotation angle taking q_true to q_est, in radians."""
    dq = q_est * q_true.conj()
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(dq.w))))
