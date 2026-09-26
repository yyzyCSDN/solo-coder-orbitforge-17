"""星敏感器模拟演示：PYTHONPATH=src python -m orbitforge.starsensor.demo

四个场景：
1. 标称工况——随机姿态全链路定姿
2. 视场边缘——亮星落在视场边缘，PSF 截断
3. 星等阈值——匹配库星等上限对识别成功率的影响
4. 错误匹配——注入假星；以及星数不足时的失败行为（不返回真姿态）
"""
from __future__ import annotations
import math
import random
from orbitforge.core.vector import Vec3
from orbitforge.attitude.quaternion import Quaternion
from .catalog import StarCatalog
from .camera import CameraParams
from .estimate import boresight_toward
from .pipeline import StarSensorSim, ARCSEC

def random_attitude(rng):
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    return Quaternion(
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    )

def attitude_with_star_near_edge(star_unit, radius_px, camera):
    """构造姿态，使给定星落在距像面中心 radius_px 处（靠近视场边缘）。"""
    theta = math.atan(radius_px / camera.focal_px)
    ref = Vec3(0.0, 0.0, 1.0) if abs(star_unit.z) < 0.9 else Vec3(1.0, 0.0, 0.0)
    perp = star_unit.cross(ref).unit()
    boresight = (star_unit * math.cos(theta) + perp * math.sin(theta)).unit()
    return boresight_toward(boresight)

def show(tag, r):
    print(f'  [{tag}] 视场内星表星 {r.n_catalog_in_fov}（暗于阈值被排除 {r.n_dim_excluded}，'
          f'边缘截断 {r.n_edge_truncated}） 检测星点 {r.n_detected} 匹配 {r.n_matched} '
          f'内点 {r.n_inliers} 离群 {r.n_outliers} 未识别 {r.n_unidentified}')
    if r.estimated_attitude is None:
        print('  -> 定姿失败：estimated_attitude = None（未回退真姿态）')
    else:
        print(f'  -> 姿态误差 {r.attitude_error_arcsec:8.2f}" '
              f'(横向 {r.cross_boresight_error_arcsec:.2f}" 滚转 {r.roll_error_arcsec:.2f}") '
              f'残差 RMS {r.residual_rms_arcsec:.2f}" 最大 {r.residual_max_arcsec:.2f}"')
    for m in r.messages:
        print(f'  ! {m}')

def main():
    print('=' * 78)
    print('星敏感器端到端模拟（真姿态仅用于生成图像与评估误差，估计完全来自图像链路）')
    print('=' * 78)
    camera = CameraParams(n_pixels=512, fov_deg=12.0)
    catalog = StarCatalog.random(n_stars=6000, seed=7)
    sim = StarSensorSim(catalog=catalog, camera=camera, mag_limit=6.0, seed=11)
    print(f'星表 {len(catalog)} 颗，匹配库（mag<=6.0）{len(sim.match_catalog)} 颗，'
          f'角距库 {len(sim.pair_db)} 对')

    print('\n[1] 标称工况：3 个随机姿态')
    rng = random.Random(2024)
    for k in range(3):
        show(f'姿态{k + 1}', sim.run(random_attitude(rng), seed=100 + k))

    print('\n[2] 视场边缘：亮星置于距边缘约 3 像元处（PSF 被截断）')
    bright = min(catalog.stars, key=lambda s: s.mag)
    edge_radius = 0.5 * camera.n_pixels - 3.0
    q_edge = attitude_with_star_near_edge(bright.unit, edge_radius, camera)
    show('边缘星', sim.run(q_edge, seed=200))
    q_out = attitude_with_star_near_edge(bright.unit, 0.5 * camera.n_pixels + 2.0, camera)
    show('星心在阵列外2px', sim.run(q_out, seed=201))

    print('\n[3] 星等阈值：匹配库 mag_limit 对 8 个随机姿态的影响')
    for lim in (6.0, 5.2, 4.6):
        s = StarSensorSim(catalog=catalog, camera=camera, mag_limit=lim, seed=11)
        ok, errs = 0, []
        for k in range(8):
            r = s.run(random_attitude(rng), seed=300 + k)
            if r.estimated_attitude is not None:
                ok += 1
                errs.append(r.attitude_error_arcsec)
        mean_err = sum(errs) / len(errs) if errs else float('nan')
        print(f'  mag_limit={lim}: 库星 {len(s.match_catalog)} 颗，成功 {ok}/8，'
              f'平均姿态误差 {mean_err:.2f}"')

    print('\n[4] 错误匹配')
    print('  (a) 注入 6 颗假星（亮度假星，非星表源）：')
    show('含假星', sim.run(random_attitude(rng), n_fake_stars=6, seed=400))
    print('  (b) mag_limit=4.0 的稀疏匹配库 + 6 颗假星（星数不足以定姿）：')
    sim_sparse = StarSensorSim(catalog=catalog, camera=camera, mag_limit=4.0, seed=11)
    show('稀疏+假星', sim_sparse.run(random_attitude(rng), n_fake_stars=6, seed=401))

if __name__ == '__main__':
    main()
