# 星敏感器模拟模块（`orbitforge.attitude.starsensor`）

从星表、视场、姿态和噪声生成星点图像，再从图像中识别星向量、恢复姿态并给出定姿残差。
纯标准库实现（与 orbitforge 代码风格一致，无 numpy 依赖）。

## 数据流

```
星表 + 真姿态 ──► render() ──► 星点图像（PSF+泊松/读出噪声+伪斑+热像素）
                                   │
                                   ▼
                     detect_spots()  质心提取（仅图像）
                                   ▼
                     identify_stars() 三角形/金字塔种子 + 姿态引导匹配
                                   ▼
                     estimate_with_rejection() Wahba 定姿 + 残差剔除
                                   ▼
                     AttitudeSolution（姿态、残差、形式协方差、显式计数）
                                   ▼
                     validate()  与真姿态对比（仅事后验证）
```

## 诚实性设计（不把真姿态当答案）

- **真姿态只进入 `render()`**（传感器模型）。`estimate(image)` 的输入只有图像与星表，
  签名中没有任何姿态参数（有测试 `test_estimator_cannot_see_truth` 保证）。
- 定姿完全由测量驱动：质心 → 角距几何哈希识别 → Wahba 解算 → 标准化残差剔除。
- 不可观测的场景**不产出姿态**：返回 `INSUFFICIENT_STARS` / `LOST_IN_SPACE`，
  `quaternion=None`，并附带原因字符串。宁可报丢失，不给编造的姿态。
- 一致性闸门：解算后标准化残差 rms(z) > 2.5 的匹配集整体拒绝（防止"自信地错"）。
- `validate()` 仅用于仿真事后评分，估计器从不调用。

## 三个必须显式处理的情况

| 情况 | 机制 | 显式结果 |
|---|---|---|
| **视场边缘** | PSF 被传感器边界截断 → 质心有偏。质心器从图像上标记 `edge_clipped`，边缘星不参与识别与定姿 | `RenderInfo.n_edge_clipped`、`AttitudeSolution.n_edge_excluded`；测试验证最亮星被排除在匹配之外 |
| **星等阈值** | `limiting_magnitude` 以上的星根本不成像；机载角距库也按同一阈值截断 | `RenderInfo.n_skipped_magnitude` 精确计数；测试与独立重算逐一核对 |
| **错误匹配** | 三层防线：①金字塔（4 星）种子优先 + 姿态引导支持数裁决；②引导匹配含**光度门限**（修正质心测光 vs 星表星等，击杀"幽灵三角形"）；③Wahba 验后标准化残差逐步剔除（抗掩蔽） | `rejected_matches` 列表（含原因）；`n_unmatched` 计伪斑；测试验证人为错误关联被显式拒绝 |

## 关键算法

- **质心**：中值背景 + MAD 噪声 → 5σ 阈值 → 8 连通域 → 强度加权质心；
  热像素先去刺（孤立饱和像素）；σ 模型 = CRB × 1.6（蒙特卡洛标定）。
- **测光**：二维高斯阈值截断解析修正（捕获率 1 − T/P），残余星等误差 p90 ≈ 0.15 等。
- **识别**：星对角距库（二分查询）→ 三角形/金字塔种子（逐星对容差
  3.5√(σa²+σb²)）→ 种子姿态 → 全天星表反投影 → 位置门限 + 光度门限匹配，
  以支持数与 z_rms 裁决。
- **定姿**：Davenport q 方法（Jacobi 特征分解，无 numpy），加权最小二乘；
  残差按各自测量 σ 标准化后逐步剔除最差匹配（MAD 尺度封顶 2.5 防掩蔽）；
  形式协方差 P = [Σ wᵢ(I − bᵢbᵢᵀ)]⁻¹。

## 用法

```python
from orbitforge.attitude.starsensor import (
    StarCatalog, CameraModel, StarSensorSim, validate, random_quaternion)

catalog = StarCatalog.synthetic(n_stars=3000, seed=20260926)
sim = StarSensorSim(catalog, CameraModel(), seed=42)

q_true = random_quaternion(rng)          # 真姿态：仅用于成像与事后验证
image, info = sim.render(q_true, n_spurious=3, n_hot_pixels=40)
sol = sim.estimate(image)                # 估计器看不到 q_true
print(sol.status, sol.rms_residual_arcsec, sol.sigma_body_arcsec)
print(validate(sol, q_true))             # 事后验证（非估计器输出）
```

## 运行

```bash
PYTHONPATH=src python3 demo_star_sensor.py        # 五个场景演示
PYTHONPATH=src python3 tests/test_star_sensor.py  # 12 项测试（也可被 pytest 收集）
```

## 已知限制

- 合成星表（3000 星）较稀疏，星稀视场（<3 可用星）会诚实报 LOST；
  换更密星表可提高解出率。
- 金字塔搜索预算有限（`max_guided`），极端污染场景可能超时放弃而非误识别——这是设计意图。
- 256×256 纯 Python 成像约 0.3 s/帧，识别+定姿约 0.3–0.6 s/帧。
