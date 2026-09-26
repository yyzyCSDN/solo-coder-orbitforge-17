"""Star sensor (star tracker) simulation: imaging, identification, attitude.

Public API:
    StarCatalog.synthetic(...)   all-sky catalog with magnitudes
    CameraModel(...)             pinhole camera + detector noise parameters
    StarSensorSim(catalog, cam)  .render(q_true) -> (image, RenderInfo)
                                 .estimate(image) -> AttitudeSolution
    validate(solution, q_true)   post-hoc truth comparison (verification only)
"""
from .camera import CameraModel
from .catalog import StarCatalog, PairDatabase, CatalogStar
from .centroid import Detection, detect_spots
from .estimation import AttitudeEstimate, estimate_with_rejection, solve_wahba
from .identification import identify_stars
from .imaging import RenderInfo, StarRenderRecord, render_image
from .pipeline import (
    AttitudeSolution, MatchReport, RejectedMatch, StarSensorSim, validate,
    STATUS_OK, STATUS_DEGRADED, STATUS_INSUFFICIENT_STARS, STATUS_LOST_IN_SPACE,
)
from .rotations import (
    attitude_error_rad, dcm_apply, dcm_to_quat, quat_from_two_vectors,
    quat_to_dcm, random_quaternion,
)

__all__ = [
    "CameraModel", "StarCatalog", "PairDatabase", "CatalogStar",
    "Detection", "detect_spots",
    "AttitudeEstimate", "estimate_with_rejection", "solve_wahba",
    "identify_stars",
    "RenderInfo", "StarRenderRecord", "render_image",
    "AttitudeSolution", "MatchReport", "RejectedMatch", "StarSensorSim",
    "validate",
    "STATUS_OK", "STATUS_DEGRADED", "STATUS_INSUFFICIENT_STARS",
    "STATUS_LOST_IN_SPACE",
    "attitude_error_rad", "dcm_apply", "dcm_to_quat", "quat_from_two_vectors",
    "quat_to_dcm", "random_quaternion",
]
