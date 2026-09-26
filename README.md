# OrbitForge Mission Lab

OrbitForge is a Python library and small HTTP service for spacecraft mission
analysis: orbital mechanics, mission geometry and resource products.

## Quick start

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src python -m pytest -q
orbitforge
```

The service listens on `127.0.0.1:8080` by default. `GET /live` and `GET /ready`
report service state, and the analysis endpoints live under `/v1/`.

## Star sensor simulation (`orbitforge.starsensor`)

End-to-end star tracker simulation, pure standard library:

- `catalog` — synthetic sky catalog (density ∝ 10^0.4·mag) and a pairwise
  angular-separation database for pattern matching.
- `camera` — pinhole projection of catalog stars for a given attitude, then
  image rendering: Gaussian PSF, background, shot/read noise, optional injected
  fake stars; stars at the FOV edge are truncated by the array boundary.
- `centroid` — thresholding, connected-component labeling and center-of-mass
  centroiding; centroids are back-projected to body-frame unit vectors.
- `identify` — geometric voting over the pair database (each observed star
  keeps only a strict vote winner).
- `estimate` — TRIAD, Davenport q-method Wahba solver (Jacobi eigensolver),
  per-star residuals and attitude-error decomposition (cross-boresight / roll).
- `pipeline` — `StarSensorSim.run(true_attitude)` drives the whole chain and
  returns a `SimResult` with detection/match/inlier counts, outlier rejections
  (RANSAC consistency check), residual RMS/max and attitude error in arcsec.

The estimated attitude is computed solely from the image chain; the true
attitude is only used to render the image and to score the result. When
identification fails (too few usable matches, e.g. a sparse catalog at a
bright magnitude limit), `estimated_attitude` is `None` — the truth is never
returned as a fallback.

```bash
PYTHONPATH=src python -m orbitforge.starsensor.demo   # 4 scenarios: nominal, FOV edge, magnitude threshold, false matches
PYTHONPATH=src python tests/test_star_sensor.py       # test suite (also pytest-compatible)
```

