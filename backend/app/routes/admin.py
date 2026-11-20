"""
Admin Dashboard Routes
"""
import json
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from ..database import db, Incident, Complaint, DuplicateMatch, PriorityAnalysis
from ..utils.dbscan_hotspot import compute_hotspots
from ..utils.priority_engine import analyze_incident_priority
from sqlalchemy import func

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/stats', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics."""
    total_complaints = Complaint.query.count()
    total_incidents = Incident.query.count()
    critical = Incident.query.filter_by(priority_category='Critical').count()
    urgent = Incident.query.filter_by(priority_category='Urgent').count()
    normal = Incident.query.filter_by(priority_category='Normal').count()
    resolved = Incident.query.filter_by(status='Resolved').count()
    in_progress = Incident.query.filter_by(status='In Progress').count()
    pending_review = DuplicateMatch.query.filter_by(reviewed=False).count()

    # Category breakdown
    category_counts = db.session.query(
        Incident.category, func.count(Incident.id).label('count')
    ).group_by(Incident.category).all()

    # Status breakdown
    status_counts = db.session.query(
        Incident.status, func.count(Incident.id).label('count')
    ).group_by(Incident.status).all()

    # Trend data (last 30 days)
    trend_data = []
    for days_ago in range(29, -1, -1):
        date = datetime.utcnow() - timedelta(days=days_ago)
        date_str = date.strftime('%Y-%m-%d')
        count = Complaint.query.filter(
            func.date(Complaint.submitted_at) == date.date()
        ).count()
        trend_data.append({'date': date_str, 'count': count})

    return jsonify({
        'total_complaints': total_complaints,
        'total_incidents': total_incidents,
        'critical_incidents': critical,
        'urgent_incidents': urgent,
        'normal_incidents': normal,
        'resolved_incidents': resolved,
        'in_progress_incidents': in_progress,
        'pending_duplicate_reviews': pending_review,
        'category_breakdown': [{'category': c[0], 'count': c[1]} for c in category_counts],
        'status_breakdown': [{'status': s[0], 'count': s[1]} for s in status_counts],
        'complaint_trend': trend_data,
    })


@admin_bp.route('/incidents', methods=['GET'])
def admin_list_incidents():
    """Admin incident list with full details."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    status = request.args.get('status')
    category = request.args.get('category')
    priority = request.args.get('priority_category')
    sort_by = request.args.get('sort_by', 'priority_score')
    sort_dir = request.args.get('sort_dir', 'desc')

    query = Incident.query

    if status:
        query = query.filter_by(status=status)
    if category:
        query = query.filter_by(category=category)
    if priority:
        query = query.filter_by(priority_category=priority)

    sort_col = getattr(Incident, sort_by, Incident.priority_score)
    if sort_dir == 'desc':
        query = query.order_by(sort_col.desc())
    else:
        query = query.order_by(sort_col.asc())

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'incidents': [_incident_admin_dict(i) for i in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


@admin_bp.route('/incidents/<incident_uuid>', methods=['PATCH'])
def admin_update_incident(incident_uuid):
    """Admin update incident fields."""
    incident = db.session.get(Incident, incident_uuid)
    if not incident:
        return jsonify({'error': 'Incident not found'}), 404

    data = request.get_json()
    priority_affecting_change = False   # track if re-analysis is needed

    if 'status' in data:
        valid = ['Submitted', 'AI Analysed', 'Under Review', 'Assigned', 'In Progress', 'Resolved']
        if data['status'] not in valid:
            return jsonify({'error': 'Invalid status'}), 400
        incident.status = data['status']
        priority_affecting_change = True   # status changes time_history_score
        if data['status'] == 'Resolved':
            incident.resolved_at = datetime.utcnow()
    if 'assigned_to' in data:
        incident.assigned_to = data['assigned_to']
    if 'severity' in data:
        incident.severity = float(data['severity'])
        priority_affecting_change = True
    if 'public_impact' in data:
        incident.public_impact = float(data['public_impact'])
        priority_affecting_change = True
    if 'location_importance' in data:
        incident.location_importance = float(data['location_importance'])
        priority_affecting_change = True

    # Bug C fix: recalculate and persist priority whenever a priority-affecting
    # field changes (status, severity, public_impact, location_importance).
    # Previously only complaint_count-driven changes recalculated priority.
    if priority_affecting_change:
        priority = analyze_incident_priority(incident)
        incident.priority_score = priority['priority_score']
        incident.priority_category = priority['priority_category']
        incident.priority_explanation = json.dumps(priority['explanation'])
        incident.time_history_score = priority['time_history']

        # Update the most recent PriorityAnalysis audit row
        pa = PriorityAnalysis.query.filter_by(incident_id=incident.id) \
            .order_by(PriorityAnalysis.computed_at.desc()).first()
        if pa:
            pa.time_history_score = priority['time_history']
            pa.final_priority_score = priority['priority_score']
            pa.priority_category = priority['priority_category']
            pa.explanation_factors = json.dumps(priority['explanation'])
            pa.computed_at = datetime.utcnow()
            pa.computed_by = 'admin_status_change'

    incident.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True, 'incident': _incident_admin_dict(incident)})


@admin_bp.route('/complaints', methods=['GET'])
def admin_list_complaints():
    """Admin complaint list."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    status = request.args.get('status')
    category = request.args.get('category')

    query = Complaint.query
    if status:
        query = query.filter_by(status=status)
    if category:
        query = query.filter_by(category=category)
    query = query.order_by(Complaint.submitted_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'complaints': [_complaint_admin_dict(c) for c in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


@admin_bp.route('/duplicate-matches', methods=['GET'])
def admin_duplicate_matches():
    """Get all duplicate matches for review."""
    reviewed = request.args.get('reviewed', 'false').lower() == 'true'
    matches = DuplicateMatch.query.filter_by(reviewed=reviewed).order_by(
        DuplicateMatch.combined_score.desc()
    ).all()
    result = []
    for m in matches:
        complaint = db.session.get(Complaint, m.complaint_id)
        incident = db.session.get(Incident, m.incident_id)
        if complaint and incident:
            result.append({
                'match_id': m.id,
                'complaint_id': complaint.complaint_id,
                'complaint_uuid': m.complaint_id,
                'complaint_category': complaint.category,
                'incident_code': incident.incident_code,
                'incident_uuid': m.incident_id,
                'incident_title': incident.title,
                'incident_status': incident.status,
                'geo_distance_km': m.geo_distance_km,
                'visual_similarity': m.visual_similarity,
                'combined_score': m.combined_score,
                'match_type': m.match_type,
                'reviewed': m.reviewed,
                'review_action': m.review_action,
                'reviewed_by': m.reviewed_by,
                'reviewed_at': m.reviewed_at.isoformat() if m.reviewed_at else None,
                'created_at': m.created_at.isoformat(),
            })
    return jsonify({'matches': result, 'total': len(result)})


def _incident_admin_dict(i):
    import json
    priority_exp = []
    if i.priority_explanation:
        try:
            priority_exp = json.loads(i.priority_explanation)
        except Exception:
            priority_exp = [str(i.priority_explanation)]
    return {
        'id': i.id,
        'incident_code': i.incident_code,
        'category': i.category,
        'title': i.title,
        'description': i.description,
        'latitude': i.latitude,
        'longitude': i.longitude,
        'location_address': i.location_address,
        'status': i.status,
        'severity': i.severity,
        'public_impact': i.public_impact,
        'location_importance': i.location_importance,
        'time_history_score': i.time_history_score,
        'priority_score': i.priority_score,
        'priority_category': i.priority_category,
        'priority_explanation': priority_exp,
        'complaint_count': i.complaint_count,
        'image_url': f'/api/complaints/uploads/{i.image_path}' if i.image_path else None,
        'ml_category': i.ml_category,
        'ml_confidence': i.ml_confidence,
        'ml_status': i.ml_status,
        'reported_category': getattr(i, 'reported_category', None),
        'assigned_to': i.assigned_to,
        'resolved_at': i.resolved_at.isoformat() if i.resolved_at else None,
        'created_at': i.created_at.isoformat() if i.created_at else None,
        'updated_at': i.updated_at.isoformat() if i.updated_at else None,
    }


def _complaint_admin_dict(c):
    reported  = getattr(c, 'reported_category', None)
    predicted = getattr(c, 'predicted_category', None) or c.ml_category
    mismatch  = bool(predicted and reported and predicted != reported)
    return {
        'id': c.id,
        'complaint_id': c.complaint_id,
        'reporter_name': c.reporter_name,
        'reporter_email': c.reporter_email,
        'reporter_phone': getattr(c, 'reporter_phone', None),
        'category': c.category,
        'reported_category': reported or c.category,
        'predicted_category': predicted,
        'category_mismatch': mismatch,
        'description': c.description,
        'latitude': c.latitude,
        'longitude': c.longitude,
        'location_address': c.location_address,
        'image_url': f'/api/complaints/uploads/{c.image_path}' if c.image_path else None,
        'ml_category': c.ml_category,
        'ml_confidence': c.ml_confidence,
        'ml_model_type': getattr(c, 'ml_model_type', None),
        'ml_dataset_label': getattr(c, 'ml_dataset_label', None),
        'ml_status': c.ml_status,
        'is_duplicate': c.is_duplicate,
        'incident_id': c.incident_id,
        'status': c.status,
        'submitted_at': c.submitted_at.isoformat() if c.submitted_at else None,
    }



# ─── Hotspot API ──────────────────────────────────────────────────────────────

@admin_bp.route('/hotspots', methods=['GET'])
def get_hotspots():
    """
    Run DBSCAN on all incidents with valid GPS coordinates.

    Query params:
      eps_km       float  (default 0.5) — neighbourhood radius in km
      min_samples  int    (default 2)   — minimum incidents per cluster
      status       str    — filter by incident status before clustering
      category     str    — filter by incident category before clustering
    """
    eps_km      = request.args.get('eps_km',      0.5,  type=float)
    min_samples = request.args.get('min_samples', 2,    type=int)
    status      = request.args.get('status',      None)
    category    = request.args.get('category',    None)

    eps_km      = max(0.1, min(eps_km, 20.0))
    min_samples = max(1,   min(min_samples, 20))

    query = Incident.query.filter(
        Incident.latitude.isnot(None),
        Incident.longitude.isnot(None),
    )
    if status:
        query = query.filter_by(status=status)
    if category:
        query = query.filter_by(category=category)

    incidents = query.all()

    if not incidents:
        return jsonify({
            'hotspots': [],
            'total_incidents_clustered': 0,
            'cluster_count': 0,
            'noise_count': 0,
            'eps_km': eps_km,
            'min_samples': min_samples,
        })

    from ..utils.dbscan_hotspot import compute_hotspots
    hotspots = compute_hotspots(incidents, eps_km=eps_km, min_samples=min_samples)

    clusters = [h for h in hotspots if not h['is_noise']]
    noise    = [h for h in hotspots if h['is_noise']]

    return jsonify({
        'hotspots': hotspots,
        'total_incidents_clustered': len(incidents),
        'cluster_count': len(clusters),
        'noise_count': len(noise),
        'eps_km': eps_km,
        'min_samples': min_samples,
        'risk_summary': {
            level: sum(1 for h in clusters if h['risk_level'] == level)
            for level in ('Critical', 'High', 'Moderate', 'Low')
        },
    })
