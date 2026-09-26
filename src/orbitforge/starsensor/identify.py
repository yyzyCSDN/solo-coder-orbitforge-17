from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class StarMatch:
    obs_index: int   # 观测星（质心）序号
    star_id: int     # 星表星号
    votes: int

def vote_matches(obs_vectors, pair_db, angle_tol_rad, min_votes=3):
    """几何投票星图识别。

    对全部观测星对查角距库并投票；每颗观测星保留票数严格第一且达到
    min_votes 的星表候选（并列即放弃，留待下游姿态一致性检验兜底）；
    星表星号唯一（票数高者优先）。投票结果可能含错误匹配，需由下游
    RANSAC 检验剔除。
    """
    votes = {}
    n = len(obs_vectors)
    for i in range(n):
        vi = obs_vectors[i]
        for j in range(i + 1, n):
            ang = vi.angle(obs_vectors[j])
            for a, b in pair_db.query(ang, angle_tol_rad):
                votes[(i, a)] = votes.get((i, a), 0) + 1
                votes[(j, b)] = votes.get((j, b), 0) + 1
                votes[(i, b)] = votes.get((i, b), 0) + 1
                votes[(j, a)] = votes.get((j, a), 0) + 1
    best = {}
    second = {}
    for (i, sid), v in votes.items():
        cur = best.get(i)
        if cur is None or v > cur[1]:
            second[i] = cur[1] if cur else 0
            best[i] = (sid, v)
        elif v > second.get(i, 0):
            second[i] = v
    matches = []
    for i, (sid, v) in best.items():
        if v < min_votes or v <= second.get(i, 0):
            continue
        matches.append(StarMatch(i, sid, v))
    matches.sort(key=lambda m: -m.votes)
    seen = set()
    out = []
    for m in matches:
        if m.star_id in seen:
            continue
        seen.add(m.star_id)
        out.append(m)
    out.sort(key=lambda m: m.obs_index)
    return out
