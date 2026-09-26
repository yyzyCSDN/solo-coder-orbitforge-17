from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.core.errors import GeometryError
from .catalog import StarCatalog, PairDatabase
from .camera import CameraParams, project_stars, render_image
from .centroid import extract_centroids, centroid_to_body
from .identify import vote_matches
from .estimate import triad, wahba, attitude_residuals_rad, attitude_error_rad, error_components_rad

ARCSEC = 206264.80624709636

@dataclass
class SimResult:
    true_attitude: Quaternion
    estimated_attitude: Quaternion | None      # 仅由图像链路解算；失败时为 None
    attitude_error_arcsec: float | None        # 总姿态误差角
    cross_boresight_error_arcsec: float | None # 横向（光轴指向）误差
    roll_error_arcsec: float | None            # 滚转误差
    n_catalog_in_fov: int                      # 视场内（含边缘渗入）被渲染的星表星
    n_dim_excluded: int                        # 视场内但暗于 mag_limit、不在匹配库中的星
    n_edge_truncated: int                      # PSF 被视场边缘截断的星
    n_detected: int                            # 质心提取得到的星点数
    n_fake_injected: int                       # 注入的假星数
    n_matched: int                             # 投票后得到的对应关系数
    n_inliers: int                             # 姿态一致性检验保留的内点数
    n_outliers: int                            # 被剔除的错误匹配数
    n_unidentified: int                        # 未识别源（假星/暗星/截断星）
    residual_rms_arcsec: float | None
    residual_max_arcsec: float | None
    residuals_arcsec: list[float]
    messages: list[str] = field(default_factory=list)
    image: list | None = None

class StarSensorSim:
    """星敏感器端到端模拟：星表+姿态+噪声 -> 图像 -> 识别 -> 定姿 -> 残差。

    估计链路（render -> centroid -> identify -> wahba）不接触真姿态；
    真姿态只用于生成图像和最后评估误差。识别/定姿失败时
    estimated_attitude 为 None，绝不回退为真姿态。
    """

    def __init__(self, catalog=None, camera=CameraParams(), mag_limit=6.0, seed=1,
                 angle_tol_rad=1.5e-4, inlier_tol_rad=1.5e-4, min_inliers=3, ransac_iters=300):
        self.catalog = catalog if catalog is not None else StarCatalog.random()
        self.camera = camera
        self.mag_limit = mag_limit
        self.seed = seed
        self.angle_tol_rad = angle_tol_rad
        self.inlier_tol_rad = inlier_tol_rad
        self.min_inliers = min_inliers
        self.ransac_iters = ransac_iters
        self.match_catalog = self.catalog.brighter_than(mag_limit)
        half_diag = math.atan(math.sqrt(2.0) * math.tan(0.5 * camera.fov_rad))
        min_sep = 5.0 * camera.pixel_scale_rad
        self.pair_db = PairDatabase.build(self.match_catalog, min_sep, 2.0 * half_diag)

    def run(self, true_attitude: Quaternion, n_fake_stars=0, keep_image=False, seed=None) -> SimResult:
        rng = random.Random(self.seed if seed is None else seed)
        msgs = []
        cam = self.camera
        projected = project_stars(self.catalog, true_attitude, cam)
        n_dim = sum(1 for p in projected if p.mag > self.mag_limit)
        n_edge = sum(1 for p in projected if p.truncated)
        image, _truth = render_image(projected, cam, rng, n_fake_stars=n_fake_stars)
        cents, _bg, _sig = extract_centroids(image)
        obs = [centroid_to_body(c, cam) for c in cents]
        matches = vote_matches(obs, self.pair_db, self.angle_tol_rad)
        n_unidentified = len(obs) - len({m.obs_index for m in matches})
        est, inliers, outliers = self._robust_attitude(matches, obs)
        if est is None or len(inliers) < self.min_inliers:
            got = 0 if est is None else len(inliers)
            msgs.append(f'定姿失败：有效匹配 {got} 颗 < 最少 {self.min_inliers} 颗；'
                        f'估计置为 None（不回退真姿态）')
            return SimResult(true_attitude, None, None, None, None,
                             len(projected), n_dim, n_edge, len(cents), n_fake_stars,
                             len(matches), 0, len(matches), n_unidentified,
                             None, None, [], msgs, image if keep_image else None)
        res = attitude_residuals_rad(est,
                                     [obs[m.obs_index] for m in inliers],
                                     [self.match_catalog.by_id[m.star_id].unit for m in inliers])
        res_as = [r * ARCSEC for r in res]
        rms = math.sqrt(sum(r * r for r in res_as) / len(res_as))
        err = attitude_error_rad(est, true_attitude) * ARCSEC
        ex, ey, ez = error_components_rad(est, true_attitude)
        cross = math.hypot(ex, ey) * ARCSEC
        roll = abs(ez) * ARCSEC
        if outliers:
            msgs.append(f'剔除错误匹配 {len(outliers)} 个（姿态一致性检验）')
        if n_unidentified:
            msgs.append(f'未识别源 {n_unidentified} 个（假星/暗星/边缘截断），未参与定姿')
        if n_edge:
            msgs.append(f'视场边缘截断星 {n_edge} 颗，质心偏差由离群剔除机制吸收')
        return SimResult(true_attitude, est, err, cross, roll,
                         len(projected), n_dim, n_edge, len(cents), n_fake_stars,
                         len(matches), len(inliers), len(outliers), n_unidentified,
                         rms, max(res_as), res_as, msgs, image if keep_image else None)

    def _robust_attitude(self, matches, obs):
        """RANSAC(TRIAD) 粗解 + 内点集 Wahba 精化。只使用观测向量与星表。"""
        if len(matches) < 2:
            return None, [], list(matches)
        refs = [self.match_catalog.by_id[m.star_id].unit for m in matches]
        vecs = [obs[m.obs_index] for m in matches]
        rng = random.Random(self.seed + 1)
        best_in = []
        for _ in range(self.ransac_iters):
            i, j = rng.sample(range(len(matches)), 2)
            try:
                q = triad(vecs[i], vecs[j], refs[i], refs[j])
            except GeometryError:
                continue
            inl = [k for k in range(len(matches))
                   if vecs[k].angle(q.rotate(refs[k])) < self.inlier_tol_rad]
            if len(inl) > len(best_in):
                best_in = inl
        if len(best_in) < 2:
            return None, [], list(matches)
        for _ in range(3):  # 迭代精化：Wahba -> 重定内点
            q = wahba([vecs[k] for k in best_in], [refs[k] for k in best_in])
            inl = [k for k in range(len(matches))
                   if vecs[k].angle(q.rotate(refs[k])) < self.inlier_tol_rad]
            if inl == best_in:
                break
            if len(inl) < 2:
                return None, [], list(matches)
            best_in = inl
        inlier_matches = [matches[k] for k in best_in]
        outliers = [matches[k] for k in range(len(matches)) if k not in set(best_in)]
        return q, inlier_matches, outliers
