"""Loss terms for SMIL registration (the inverse problem)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def point_to_mesh_distance(verts: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Distance from each observed point to the nearest mesh VERTEX.

    (Vertex proxy for surface distance — fast and sufficient for the
    controlled recovery test; a true point-to-surface term can be added later.)
    """
    tree = cKDTree(verts)
    d, _ = tree.query(points)
    return d


def scan_loss(verts: np.ndarray, points: np.ndarray) -> float:
    """Mean squared distance from observed points to the mesh."""
    d = point_to_mesh_distance(verts, points)
    return float(np.mean(d ** 2))


def pose_prior_loss(pose: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> float:
    """Mahalanobis pose-prior term: (theta - mean)^T precision (theta - mean)."""
    diff = pose - mean
    return float(diff.T @ precision @ diff)


def shape_prior_loss(betas: np.ndarray) -> float:
    """Simple L2 shape prior on beta."""
    return float(np.sum(betas ** 2))
