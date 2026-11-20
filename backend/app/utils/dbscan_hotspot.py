"""
dbscan_hotspot.py — SMCI
========================
Spatial hotspot detection using DBSCAN clustering.

Groups geo-located incidents into spatial clusters (hotspots).
Works with scikit-learn only — no PyTorch or OpenCV required.

Each hotspot represents a geographic area with elevated complaint density.
The module exposes two functions used by the admin API:

    compute_hotspots(incidents)  → list of Hotspot dicts
    classify_zone_risk(incidents) → per-cluster risk label

Algorithm:
  1. Project lat/lon to radians for Haversine-compatible DBSCAN.
  2. Run DBSCAN with earth_radius metric.
  3. For each cluster, compute:
       - centroid (mean lat/lon)
       - member incident list
       - dominant category
       - aggregate priority score
       - risk level (Critical / High / Moderate / Low)
  4. Noise points (label=-1) are returned as single-point "hotspots"
     with risk=Low.

Usage:
    from app.utils.dbscan_hotspot import compute_hotspots
    hotspots = compute_hotspots(incidents)
"""

import math
from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN

# ─── Constants ────────────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0

# eps_km: maximum distance between two incidents to be in the same cluster.
# 0.5 km (500 m) is a sensible neighbourhood for city-block hotspot detection.
DEFAULT_EPS_KM    = 0.5
DEFAULT_MIN_SAMPLES = 2  # at least 2 incidents to form a cluster

RISK_THRESHOLDS = {
    'Critical': 70,
    'High':     45,
    'Moderate': 20,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _haversine_km(a: np.ndarray, b: np.ndarray) -> float:
    """Haversine distance in km between two [lat_rad, lon_rad] points."""
    dlat = b[0] - a[0]
    dlon = b[1] - a[1]
    h = math.sin(dlat / 2) ** 2 + math.cos(a[0]) * math.cos(b[0]) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _risk_label(score: float, member_count: int) -> str:
    """Assign a risk label based on aggregate priority score and cluster size."""
    adjusted = score + (member_count - 1) * 5  # bonus per extra member
    for label, threshold in RISK_THRESHOLDS.items():
        if adjusted >= threshold:
            return label
    return 'Low'


# ─── Main function ────────────────────────────────────────────────────────────

def compute_hotspots(
    incidents: List[Any],
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[Dict]:
    """
    Run DBSCAN on a list of Incident ORM objects or dicts.
    Accepts both SQLAlchemy model instances and plain dicts.

    Returns a list of hotspot dicts, sorted by risk (Critical first).
    """
    if not incidents:
        return []

    # Normalise input: support ORM objects and plain dicts
    def _get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    # Build coordinate array
    valid = []
    for inc in incidents:
        lat = _get(inc, 'latitude')
        lon = _get(inc, 'longitude')
        if lat is None or lon is None:
            continue
        try:
            valid.append((float(lat), float(lon), inc))
        except (TypeError, ValueError):
            continue

    if not valid:
        return []

    lats = np.array([v[0] for v in valid])
    lons = np.array([v[1] for v in valid])

    # Convert to radians for Haversine metric
    coords_rad = np.column_stack([np.radians(lats), np.radians(lons)])

    # DBSCAN epsilon in radians
    eps_rad = eps_km / EARTH_RADIUS_KM

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm='ball_tree',
        metric='haversine',
    ).fit(coords_rad)

    labels = db.labels_

    # Group by cluster label
    clusters: Dict[int, List] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(valid[idx])

    hotspots = []

    for cluster_label, members in clusters.items():
        member_lats = [m[0] for m in members]
        member_lons = [m[1] for m in members]
        member_incs = [m[2] for m in members]

        centroid_lat = float(np.mean(member_lats))
        centroid_lon = float(np.mean(member_lons))
        is_noise = bool(cluster_label == -1)

        # Category distribution
        categories = [
            str(_get(inc, 'category') or _get(inc, 'ml_category') or 'Unknown')
            for inc in member_incs
        ]
        cat_counter = Counter(categories)
        dominant_cat = cat_counter.most_common(1)[0][0]

        # Aggregate priority score
        scores = []
        for inc in member_incs:
            ps = _get(inc, 'priority_score')
            try:
                scores.append(float(ps or 0))
            except (TypeError, ValueError):
                scores.append(0.0)
        avg_score = float(np.mean(scores)) if scores else 0.0
        max_score = float(max(scores)) if scores else 0.0

        # Status breakdown
        statuses = [str(_get(inc, 'status') or 'Unknown') for inc in member_incs]
        status_counter = Counter(statuses)

        # Risk level
        risk = 'Low' if is_noise else _risk_label(avg_score, len(members))

        # Incident IDs for the frontend
        incident_ids = [
            str(_get(inc, 'id', '') or '')
            for inc in member_incs
        ]
        incident_codes = [
            str(_get(inc, 'incident_code', '') or '')
            for inc in member_incs
        ]

        # Bounding box for the cluster
        bbox = {
            'min_lat': float(min(member_lats)),
            'max_lat': float(max(member_lats)),
            'min_lon': float(min(member_lons)),
            'max_lon': float(max(member_lons)),
        }

        hotspots.append({
            'cluster_id'        : int(cluster_label),
            'is_noise'          : is_noise,
            'centroid_lat'      : centroid_lat,
            'centroid_lon'      : centroid_lon,
            'member_count'      : int(len(members)),
            'incident_ids'      : incident_ids,
            'incident_codes'    : incident_codes,
            'dominant_category' : dominant_cat,
            'category_breakdown': {k: int(v) for k, v in cat_counter.items()},
            'status_breakdown'  : {k: int(v) for k, v in status_counter.items()},
            'avg_priority_score': round(avg_score, 2),
            'max_priority_score': round(max_score, 2),
            'risk_level'        : risk,
            'bbox'              : bbox,
            'eps_km'            : eps_km,
            'min_samples'       : min_samples,
        })

    # Sort: Critical > High > Moderate > Low; within level by member count desc
    risk_order = {'Critical': 0, 'High': 1, 'Moderate': 2, 'Low': 3}
    hotspots.sort(key=lambda h: (risk_order.get(h['risk_level'], 4), -h['member_count']))

    return hotspots

