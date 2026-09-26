from __future__ import annotations
import math
from orbitforge.core.vector import Vec3
from orbitforge.attitude.quaternion import Quaternion

def quat_to_dcm(q: Quaternion):
    """姿态四元数 -> 方向余弦阵，使 R*v == q.rotate(v)。"""
    q = q.normalized()
    w, x, y, z = q.w, q.x, q.y, q.z
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]

def dcm_to_quat(m) -> Quaternion:
    """Shepperd 法：方向余弦阵 -> 四元数。"""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = Quaternion(0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        q = Quaternion((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        q = Quaternion((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        q = Quaternion((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    return q.normalized()

def boresight_toward(pointing: Vec3, reference=Vec3(0.0, 0.0, 1.0)) -> Quaternion:
    """构造使光轴（本体 +Z）指向惯性向量 pointing 的姿态（滚转任取）。"""
    row3 = pointing.unit()
    ref = reference if abs(row3.dot(reference)) < 0.9 else Vec3(1.0, 0.0, 0.0)
    row1 = row3.cross(ref).unit()
    row2 = row3.cross(row1)
    return dcm_to_quat([
        [row1.x, row1.y, row1.z],
        [row2.x, row2.y, row2.z],
        [row3.x, row3.y, row3.z],
    ])

def triad(b1: Vec3, b2: Vec3, r1: Vec3, r2: Vec3) -> Quaternion:
    """TRIAD：由两对 本体/惯性 向量求姿态（惯性 -> 本体）。"""
    t1b = b1.unit()
    t2b = b1.cross(b2).unit()
    t3b = t1b.cross(t2b)
    t1r = r1.unit()
    t2r = r1.cross(r2).unit()
    t3r = t1r.cross(t2r)
    tb = ((t1b.x, t1b.y, t1b.z), (t2b.x, t2b.y, t2b.z), (t3b.x, t3b.y, t3b.z))
    tr = ((t1r.x, t1r.y, t1r.z), (t2r.x, t2r.y, t2r.z), (t3r.x, t3r.y, t3r.z))
    m = [[sum(tb[k][i] * tr[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return dcm_to_quat(m)

def _jacobi_dominant_eigenvector(k4, sweeps=64, tol=1e-15):
    """对称 4x4 矩阵最大特征值对应的特征向量（Jacobi 旋转法）。"""
    a = [row[:] for row in k4]
    v = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for _ in range(sweeps):
        p, q, mx = 0, 1, 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                if abs(a[i][j]) > mx:
                    mx = abs(a[i][j])
                    p, q = i, j
        if mx < tol:
            break
        theta = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(theta), math.sin(theta)
        for k in range(4):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(4):
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(4):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    idx = max(range(4), key=lambda i: a[i][i])
    return [v[i][idx] for i in range(4)]

def wahba(body_vecs, ref_vecs, weights=None) -> Quaternion:
    """Davenport q 方法解 Wahba 问题：求 q 使 sum w_i|b_i - R(q) r_i|^2 最小。"""
    n = len(body_vecs)
    if n < 2:
        raise ValueError('wahba needs at least 2 vector pairs')
    if weights is None:
        weights = [1.0] * n
    b_mat = [[0.0] * 3 for _ in range(3)]
    for w, b, r in zip(weights, body_vecs, ref_vecs):
        bv = (b.x, b.y, b.z)
        rv = (r.x, r.y, r.z)
        for i in range(3):
            wi = w * bv[i]
            row = b_mat[i]
            row[0] += wi * rv[0]
            row[1] += wi * rv[1]
            row[2] += wi * rv[2]
    sigma = b_mat[0][0] + b_mat[1][1] + b_mat[2][2]
    z = [b_mat[1][2] - b_mat[2][1], b_mat[2][0] - b_mat[0][2], b_mat[0][1] - b_mat[1][0]]
    k = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            k[i][j] = b_mat[i][j] + b_mat[j][i] - (sigma if i == j else 0.0)
        k[i][3] = z[i]
        k[3][i] = z[i]
    k[3][3] = sigma
    vec = _jacobi_dominant_eigenvector(k)
    # Davenport K 的特征向量采用 Shuster 约定，其矢量部与主动转动四元数差一个符号
    q = Quaternion(vec[3], -vec[0], -vec[1], -vec[2]).normalized()
    if q.w < 0.0:
        q = Quaternion(-q.w, -q.x, -q.y, -q.z)
    return q

def attitude_residuals_rad(q: Quaternion, body_vecs, ref_vecs):
    """每对向量的残差角：angle(R(q)*r_i, b_i)。"""
    return [b.angle(q.rotate(r)) for b, r in zip(body_vecs, ref_vecs)]

def attitude_error_rad(q_est: Quaternion, q_true: Quaternion):
    """两姿态间转角，2*acos(|<q1,q2>|)。真姿态仅在此用于评估，不参与估计。"""
    a = q_est.normalized()
    b = q_true.normalized()
    d = abs(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z)
    return 2.0 * math.acos(min(1.0, d))

def error_components_rad(q_est: Quaternion, q_true: Quaternion):
    """误差四元数分解：(绕X, 绕Y, 绕Z) 小角近似。Z 分量即滚转误差。"""
    dq = q_true.normalized().conj() * q_est.normalized()
    if dq.w < 0.0:
        dq = Quaternion(-dq.w, -dq.x, -dq.y, -dq.z)
    return (2.0 * dq.x, 2.0 * dq.y, 2.0 * dq.z)
