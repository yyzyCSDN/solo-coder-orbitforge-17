"""Lost-in-space star identification (geometric hashing with pyramid seeds).

Strategy (mirrors on-board trackers):

1. For every observed pair of spots, query the catalog pair database for
   pairs at the measured angular separation (per-pair tolerance
   3.5 * sqrt(sigma_a^2 + sigma_b^2), so dim stars get the wide gate their
   centroid noise needs).
2. **Triangle/pyramid seeds** - a catalog pair candidate plus set
   intersections on the per-detection candidate star sets yields third
   (and, when possible, fourth) stars whose catalog separations reproduce
   every measured angle exactly. Four-star (pyramid) seeds are far more
   discriminating than triangles and are evaluated first.
3. **Attitude-guided verification** - each seed attitude projects the whole
   onboard catalog into the frame; the seed whose predicted positions
   attract the most detections (support) wins. A wrong seed attracts
   almost no support, which makes this step decisive.

The returned match set is the winning support set; the estimator's
residual screen remains the final arbiter for wrong associations.
"""
from __future__ import annotations

import math

from .catalog import angle_between, PairDatabase
from .estimation import solve_wahba
from .rotations import quat_to_dcm, dcm_apply


def _pair_tol(sig_a, sig_b, k_sigma):
    return k_sigma * math.sqrt(sig_a * sig_a + sig_b * sig_b)


def identify_stars(detections, pair_db: PairDatabase, sigmas_rad, catalog,
                   camera, k_sigma=3.5, max_guided=150, max_seeds=6000):
    """Identify non-edge detections. Returns (matches, unmatched, seed_ok).

    matches   : [(detection index, catalog index, 0)] - the support set
    unmatched : usable detection indices with no catalog identity
    seed_ok   : True when a seed attitude gained support >= 3
    """
    usable = [k for k, d in enumerate(detections) if not d.edge_clipped]
    m = len(usable)
    if m < 3:
        return [], usable, False

    sig = [sigmas_rad[k] for k in usable]
    vec = [detections[k].body_unit for k in usable]
    units = catalog.units()

    # Observed pairs, tightest (brightest) first.
    order = sorted(((sig[a] + sig[b], a, b)
                    for a in range(m) for b in range(a + 1, m)))
    ang, tol, pair_cands, star_sets = {}, {}, {}, {}
    for _, a, b in order:
        ang[(a, b)] = angle_between(vec[a], vec[b])
        tol[(a, b)] = _pair_tol(sig[a], sig[b], k_sigma)
        pc = pair_db.query(ang[(a, b)], tol[(a, b)])
        pair_cands[(a, b)] = pc
        ss = set()
        for ci, cj in pc:
            ss.add(ci)
            ss.add(cj)
        star_sets[(a, b)] = ss

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def sep_ok(cat_x, cat_y, a, b):
        return abs(angle_between(units[cat_x], units[cat_y])
                   - ang[key(a, b)]) <= tol[key(a, b)]

    # ---- collect triangle and pyramid seeds
    tri_seeds, pyr_seeds = [], []
    for _, a, b in order:
        if len(tri_seeds) + len(pyr_seeds) >= max_seeds:
            break
        others = [c for c in range(m) if c != a and c != b]
        for ca, cb in pair_cands[key(a, b)]:
            for x, y in ((ca, cb), (cb, ca)):
                for c in others:
                    inter = star_sets[key(a, c)] & star_sets[key(b, c)]
                    for s in inter:
                        if s == x or s == y:
                            continue
                        if not (sep_ok(x, s, a, c) and sep_ok(y, s, b, c)):
                            continue
                        # Degenerate (near-collinear) triangles are skipped.
                        if min(ang[key(a, b)], ang[key(a, c)],
                               ang[key(b, c)]) < math.radians(0.5):
                            continue
                        pyr = False
                        for d in others:
                            if d == c:
                                continue
                            inter4 = (star_sets[key(a, d)]
                                      & star_sets[key(b, d)]
                                      & star_sets[key(c, d)])
                            for t in inter4:
                                if t in (x, y, s):
                                    continue
                                if (sep_ok(x, t, a, d) and sep_ok(y, t, b, d)
                                        and sep_ok(s, t, c, d)):
                                    pyr_seeds.append(
                                        [(a, x), (b, y), (c, s), (d, t)])
                                    pyr = True
                                    break
                            if pyr:
                                break
                        if not pyr:
                            tri_seeds.append([(a, x), (b, y), (c, s)])

    # ---- evaluate seeds by guided support, best first
    best = None  # ((support, -z_rms), matches, unmatched)
    seen = set()
    budget = max_guided
    for assign in pyr_seeds + tri_seeds:
        if budget <= 0:
            break
        if best is not None and best[0][0] >= m - 1:
            break
        sig_key = tuple(sorted(assign))
        if sig_key in seen:
            continue
        seen.add(sig_key)
        idx = [u for u, _ in assign]
        q_seed, _ = solve_wahba(
            [vec[u] for u in idx], [units[c] for _, c in assign],
            [1.0 / sig[u] ** 2 for u in idx])
        budget -= 1
        matches, unmatched = attitude_guided_match(
            detections, q_seed, catalog, camera)
        support = len(matches)
        z_rms = math.inf
        if support >= 3:
            # Tie-break: photometric/angular fit quality of the support set.
            q_g, _ = solve_wahba(
                [detections[k].body_unit for k, _, _ in matches],
                [units[c] for _, c, _ in matches],
                [1.0 / sigmas_rad[k] ** 2 for k, _, _ in matches])
            A_g = quat_to_dcm(q_g)
            z2 = 0.0
            for k, c, _ in matches:
                pred = dcm_apply(A_g, units[c])
                d = detections[k].body_unit
                dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(d, pred))))
                z2 += (math.acos(dot) / sigmas_rad[k]) ** 2
            z_rms = math.sqrt(z2 / support)
        score = (support, -z_rms)
        if best is None or score > best[0]:
            best = (score, matches, unmatched)

    if best is not None and best[0][0] >= 3:
        return best[1], best[2], True
    return [], usable, False


def attitude_guided_match(detections, q, catalog, camera, gate_px=None,
                          mag_gate=0.75):
    """Match detections against catalog positions predicted by *q*.

    The onboard catalog is projected into the frame; each non-edge
    detection takes the nearest predicted catalog star within its
    positional gate AND within a photometric gate, which kills
    geometrically-consistent-but-wrong 'ghost' identities. Non-catalog
    (spurious) detections have no prediction nearby and stay unmatched;
    a wrong early identity does not propagate, because matching restarts
    from the predicted geometry.
    """
    A = quat_to_dcm(q)
    projected = []
    for s in catalog.stars:
        if s.mag > camera.limiting_magnitude:
            continue
        proj = camera.project(dcm_apply(A, s.unit))
        if proj is not None and camera.on_sensor(*proj):
            projected.append((proj[0], proj[1], s.index, s.mag))

    matches = []
    used_cat = set()
    used_det = set()
    for k, d in enumerate(detections):
        if d.edge_clipped:
            continue
        g = gate_px if gate_px is not None else max(4.0 * d.sigma_px, 1.5)
        best, best_d = None, g
        for u, v, ci, cmag in projected:
            if ci in used_cat or abs(d.mag_est - cmag) > mag_gate:
                continue
            du, dv = u - d.u, v - d.v
            if du > best_d or du < -best_d or dv > best_d or dv < -best_d:
                continue
            dist = math.hypot(du, dv)
            if dist < best_d:
                best, best_d = ci, dist
        if best is not None:
            used_cat.add(best)
            used_det.add(k)
            matches.append((k, best, 0))  # 0 marks a guided (not voted) match
    unmatched = [k for k, d in enumerate(detections)
                 if not d.edge_clipped and k not in used_det]
    return matches, unmatched
