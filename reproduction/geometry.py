"""Geometry helpers: pinhole projection and depth->point-cloud conversion."""
from __future__ import annotations

import numpy as np

import config


def project_points(points_3d: np.ndarray) -> np.ndarray:
    """Project (N,3) camera-space points to (N,2) pixel coordinates.

    Uses the pinhole model:  u = fx*X/Z + cx,  v = fy*Y/Z + cy.
    """
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    u = config.FX * x / z + config.CX
    v = config.FY * y / z + config.CY
    return np.stack([u, v], axis=1)


def depth_to_pointcloud(depth_mm: np.ndarray, mask: np.ndarray | None = None,
                        stride: int = 1) -> np.ndarray:
    """Convert a depth image (mm, uint16) into a camera-space point cloud (meters).

    Returns (N, 3) points. If `mask` is given (same shape, nonzero = foreground),
    only masked pixels are kept. `stride` subsamples the pixel grid.
    """
    h, w = depth_mm.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]

    if mask is not None:
        sel = mask[::stride, ::stride] > 0
        vv = vv[sel]
        uu = uu[sel]
    else:
        vv = vv.ravel()
        uu = uu.ravel()

    z = depth_mm[::stride, ::stride]
    if mask is not None:
        z = z[sel]
    else:
        z = z.ravel()
    z = z.astype(np.float64) / config.DEPTH_SCALE

    x = (uu - config.CX) * z / config.FX
    y = (vv - config.CY) * z / config.FY
    return np.stack([x, y, z], axis=1)
