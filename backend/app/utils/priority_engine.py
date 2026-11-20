"""
Priority Scoring Engine - Explainable Rule-Based Model
Weights: Severity 35%, Public Impact 30%, Location 20%, Time/History 15%

NOTE: This is NOT a trained ML model. It is an explainable weighted scoring system
designed to provide transparent, auditable priority assessments.
"""
import json
from datetime import datetime

WEIGHTS = {
    'severity': 0.35,
    'public_impact': 0.30,
    'location_importance': 0.20,
    'time_history': 0.15,
}

CATEGORY_SEVERITY = {
    'Potholes': 7.0,
    'Garbage Accumulation': 6.5,
    'Waterlogging': 8.5,
    'Damaged Roads': 7.5,
    'Drainage Problems': 7.0,
    'Street Lights': 5.0,
    'Other': 5.0,
}

CATEGORY_IMPACT = {
    'Potholes': 7.5,
    'Garbage Accumulation': 7.0,
    'Waterlogging': 9.0,
    'Damaged Roads': 7.5,
    'Drainage Problems': 7.0,
    'Street Lights': 5.5,
    'Other': 5.0,
}


def calculate_priority_score(severity, public_impact, location_importance, time_history):
    """Calculate weighted priority score (0-10)."""
    score = (
        severity * WEIGHTS['severity'] +
        public_impact * WEIGHTS['public_impact'] +
        location_importance * WEIGHTS['location_importance'] +
        time_history * WEIGHTS['time_history']
    )
    return round(min(10.0, max(0.0, score)), 2)


def classify_priority(score):
    """Classify priority category from score."""
    if score >= 7.5:
        return 'Critical'
    elif score >= 4.0:
        return 'Urgent'
    else:
        return 'Normal'


def compute_time_history_score(complaint_count, days_since_first, is_resolved):
    """
    Compute time/history factor:
    - More complaints = higher score
    - Older unresolved = higher score
    - Resolved incident gets lower score
    """
    if is_resolved:
        return 2.0

    base = min(10.0, complaint_count * 1.5)  # 1 complaint = 1.5, 7 = 10 (capped)
    age_factor = min(5.0, days_since_first / 10.0)  # older = higher
    score = (base * 0.6) + (age_factor * 0.4)
    return round(min(10.0, score), 2)


def compute_location_importance(latitude, longitude, address=None):
    """
    Heuristic location importance based on address keywords.
    Returns score 0-10.
    """
    high_importance_keywords = [
        'hospital', 'school', 'station', 'airport', 'junction', 'underpass',
        'market', 'mall', 'main road', 'highway', 'national highway', 'nh-',
        'silk board', 'mg road', 'brigade', 'church street'
    ]
    medium_importance_keywords = [
        'road', 'street', 'avenue', 'nagar', 'colony', 'layout'
    ]

    if not address:
        return 5.0

    address_lower = address.lower()

    for kw in high_importance_keywords:
        if kw in address_lower:
            return 8.5

    for kw in medium_importance_keywords:
        if kw in address_lower:
            return 6.0

    return 5.0


def build_explanation(severity, public_impact, location_importance, time_history,
                      category, complaint_count, priority_category):
    """Build human-readable explanation of priority factors."""
    explanations = []

    if severity >= 8.0:
        explanations.append("⚠️ High issue severity posing immediate safety risk")
    elif severity >= 6.0:
        explanations.append("🔶 Moderate to high severity affecting usability")
    else:
        explanations.append("🔷 Lower severity but requires attention")

    if public_impact >= 8.0:
        explanations.append("👥 High public impact — large number of citizens affected")
    elif public_impact >= 6.0:
        explanations.append("👥 Moderate public impact")

    if location_importance >= 8.0:
        explanations.append("📍 Located near critical infrastructure or high-traffic area")
    elif location_importance >= 6.0:
        explanations.append("📍 Located in moderately important area")

    if complaint_count >= 5:
        explanations.append(f"📢 {complaint_count} citizens reported the same incident")
    elif complaint_count >= 2:
        explanations.append(f"📢 {complaint_count} complaints linked to this incident")

    if time_history >= 7.0:
        explanations.append("🕒 Incident has remained unresolved for extended period")
    elif time_history >= 4.0:
        explanations.append("🕒 Incident reported recently — monitoring phase")

    explanations.append(f"📊 Priority: {priority_category} (Score: {calculate_priority_score(severity, public_impact, location_importance, time_history)}/10)")

    return explanations


def analyze_incident_priority(incident):
    """Full priority analysis for an Incident model object."""
    # created_at may be None for newly constructed (not yet flushed) objects
    if incident.created_at:
        days_since_first = (datetime.utcnow() - incident.created_at).days
    else:
        days_since_first = 0

    is_resolved = (incident.status == 'Resolved') if incident.status else False

    # Use stored scores if available, otherwise compute defaults
    severity = incident.severity or CATEGORY_SEVERITY.get(incident.category, 5.0)
    public_impact = incident.public_impact or CATEGORY_IMPACT.get(incident.category, 5.0)
    location_importance = incident.location_importance or compute_location_importance(
        incident.latitude, incident.longitude, incident.location_address
    )
    time_history = compute_time_history_score(
        incident.complaint_count, days_since_first, is_resolved
    )

    score = calculate_priority_score(severity, public_impact, location_importance, time_history)
    category = classify_priority(score)
    explanations = build_explanation(
        severity, public_impact, location_importance, time_history,
        incident.category, incident.complaint_count, category
    )

    return {
        'severity': severity,
        'public_impact': public_impact,
        'location_importance': location_importance,
        'time_history': time_history,
        'priority_score': score,
        'priority_category': category,
        'explanation': explanations,
        'weights': WEIGHTS,
        'breakdown': {
            'severity_contribution': round(severity * WEIGHTS['severity'], 2),
            'impact_contribution': round(public_impact * WEIGHTS['public_impact'], 2),
            'location_contribution': round(location_importance * WEIGHTS['location_importance'], 2),
            'time_contribution': round(time_history * WEIGHTS['time_history'], 2),
        }
    }

