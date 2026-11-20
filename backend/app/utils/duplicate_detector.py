"""
Duplicate Detection Engine
Combines geographic proximity + visual similarity + temporal factors
"""
import json
from datetime import datetime
from .image_utils import haversine_distance, hash_similarity, cosine_similarity

# Thresholds
GEO_DUPLICATE_THRESHOLD_KM = 0.5      # 500m
GEO_POSSIBLE_THRESHOLD_KM = 1.0       # 1km
VISUAL_EXACT_THRESHOLD = 0.90         # >90% similar = likely duplicate
VISUAL_PROBABLE_THRESHOLD = 0.70      # 70-90% = probable
COMBINED_DUPLICATE_THRESHOLD = 0.80   # combined score threshold

# Bug 1 fix: Minimum visual similarity required before auto-merging (exact/probable).
# When visual data IS available but falls below this floor, downgrade to 'possible'
# even if the combined score would otherwise qualify as exact/probable.
# This prevents nearby-GPS but visually-different complaints from being auto-merged.
VISUAL_FLOOR_FOR_AUTO_MERGE = 0.65


def compute_geo_score(distance_km):
    """Convert geographic distance to similarity score (0-1)."""
    if distance_km <= 0.1:
        return 1.0
    elif distance_km <= GEO_DUPLICATE_THRESHOLD_KM:
        return 0.9 - ((distance_km - 0.1) / (GEO_DUPLICATE_THRESHOLD_KM - 0.1)) * 0.3
    elif distance_km <= GEO_POSSIBLE_THRESHOLD_KM:
        return 0.6 - ((distance_km - GEO_DUPLICATE_THRESHOLD_KM) / (GEO_POSSIBLE_THRESHOLD_KM - GEO_DUPLICATE_THRESHOLD_KM)) * 0.4
    else:
        return max(0.0, 0.2 - distance_km / 10.0)


def compute_visual_score(hash1, hash2, features1_json, features2_json):
    """Compute visual similarity score (0-1) from hashes and features."""
    scores = []

    # Perceptual hash similarity
    if hash1 and hash2:
        h_sim = hash_similarity(hash1, hash2)
        scores.append(h_sim)

    # Feature vector cosine similarity
    if features1_json and features2_json:
        try:
            f1 = json.loads(features1_json)
            f2 = json.loads(features2_json)
            if f1 and f2 and len(f1) == len(f2):
                c_sim = cosine_similarity(f1, f2)
                scores.append(c_sim)
        except (json.JSONDecodeError, ValueError):
            pass

    if not scores:
        return None  # Cannot compute

    return sum(scores) / len(scores)


def classify_match(geo_score, visual_score, same_category, incident_resolved):
    """
    Classify the type of duplicate match.
    Returns: 'exact', 'probable', 'possible', 'manual_review', 'not_duplicate'

    Bug 1 fix: When visual data IS available but the visual similarity is below
    VISUAL_FLOOR_FOR_AUTO_MERGE (0.65), the match is capped at 'possible' and
    will never be auto-merged, regardless of how close the GPS coordinates are.
    This prevents a visually-different complaint at the same location from being
    incorrectly merged into an existing incident.
    """
    # Different category — cannot be duplicate regardless of location
    if not same_category:
        return 'not_duplicate', 0.0

    # Resolved incident at similar location — could be new incident
    if incident_resolved and geo_score >= 0.7:
        return 'possible', 0.5  # Flag for review as potential re-occurrence

    # Compute combined score
    if visual_score is not None:
        # Weight: geo 40%, visual 60%
        combined = (geo_score * 0.4) + (visual_score * 0.6)

        # Visual floor guard: if visual similarity is too low, do not auto-merge
        # even if the combined score crosses the exact/probable threshold.
        # Downgrade to 'possible' so an admin must review it.
        if visual_score < VISUAL_FLOOR_FOR_AUTO_MERGE:
            if combined >= 0.40:
                return 'possible', combined
            elif combined >= 0.25:
                return 'manual_review', combined
            else:
                return 'not_duplicate', combined
    else:
        # Only geo available — be more conservative
        combined = geo_score * 0.7  # penalize for missing visual

    if combined >= 0.85:
        return 'exact', combined
    elif combined >= 0.65:
        return 'probable', combined
    elif combined >= 0.40:
        return 'possible', combined
    elif combined >= 0.25:
        return 'manual_review', combined
    else:
        return 'not_duplicate', combined


def find_duplicate_incidents(new_complaint, all_incidents, upload_folder):
    """
    Find potential duplicate incidents for a new complaint.

    Returns list of matches sorted by combined score descending.
    Each match: {incident_id, incident_code, geo_distance_km, visual_similarity,
                 combined_score, match_type, incident_status}
    """
    matches = []

    for incident in all_incidents:
        # Skip incidents from a different category
        if incident.category != new_complaint['category']:
            continue

        # Geographic distance
        distance_km = haversine_distance(
            new_complaint['latitude'], new_complaint['longitude'],
            incident.latitude, incident.longitude
        )

        # Skip if too far (> 5km can't be same incident)
        if distance_km > 5.0:
            continue

        geo_score = compute_geo_score(distance_km)

        # Visual similarity
        visual_score = compute_visual_score(
            new_complaint.get('image_hash'),
            incident.image_hash,
            new_complaint.get('image_features'),
            incident.image_features
        )

        is_resolved = incident.status == 'Resolved'
        same_category = incident.category == new_complaint['category']

        match_type, combined_score = classify_match(
            geo_score, visual_score, same_category, is_resolved
        )

        if match_type != 'not_duplicate':
            matches.append({
                'incident_id': incident.id,
                'incident_code': incident.incident_code,
                'incident_title': incident.title,
                'incident_status': incident.status,
                'geo_distance_km': round(distance_km, 3),
                'geo_score': round(geo_score, 3),
                'visual_similarity': round(visual_score, 3) if visual_score is not None else None,
                'combined_score': round(combined_score, 3),
                'match_type': match_type,
                'is_resolved': is_resolved,
                'complaint_count': incident.complaint_count,
            })

    matches.sort(key=lambda x: x['combined_score'], reverse=True)
    return matches

