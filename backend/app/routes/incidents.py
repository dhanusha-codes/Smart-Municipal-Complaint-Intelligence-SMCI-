"""
Incident Routes - Incident management and geo-visual analysis
"""
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from ..database import db, Incident, Complaint, DuplicateMatch, PriorityAnalysis
from ..utils.priority_engine import analyze_incident_priority
incidents_bp = Blueprint('incidents', __name__)


@incidents_bp.route('', methods=['GET'])
def list_incidents():
    """List all incidents with filters."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    status = request.args.get('status')
    category = request.args.get('category')
    priority = request.args.get('priority_category')

    query = Incident.query

    if status:
        query = query.filter_by(status=status)
    if category:
        query = query.filter_by(category=category)
    if priority:
        query = query.filter_by(priority_category=priority)

    query = query.order_by(Incident.priority_score.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'incidents': [_incident_to_dict(i) for i in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


@incidents_bp.route('/map', methods=['GET'])
def get_map_data():
    """Get all incident locations for map rendering."""
    incidents = Incident.query.all()
    return jsonify({
        'incidents': [
            {
                'id': i.id,
                'incident_code': i.incident_code,
                'title': i.title,
                'category': i.category,
                'latitude': i.latitude,
                'longitude': i.longitude,
                'location_address': i.location_address,
                'status': i.status,
                'priority_category': i.priority_category,
                'priority_score': i.priority_score,
                'complaint_count': i.complaint_count,
                'image_url': f'/api/complaints/uploads/{i.image_path}' if i.image_path else None,
            }
            for i in incidents
        ]
    })


@incidents_bp.route('/<incident_id>', methods=['GET'])
def get_incident(incident_id):
    """Get incident details."""
    incident = Incident.query.filter(
        (Incident.id == incident_id) | (Incident.incident_code == incident_id)
    ).first()
    if not incident:
        return jsonify({'error': 'Incident not found'}), 404

    data = _incident_to_dict(incident)
    # Include related complaints
    data['complaints'] = [
        {
            'complaint_id': c.complaint_id,
            'reporter_name': c.reporter_name,
            'submitted_at': c.submitted_at.isoformat() if c.submitted_at else None,
            'status': c.status,
        }
        for c in incident.complaints
    ]
    # Priority analysis
    pa = PriorityAnalysis.query.filter_by(incident_id=incident.id).order_by(PriorityAnalysis.computed_at.desc()).first()
    if pa:
        data['priority_analysis'] = {
            'severity_score': pa.severity_score,
            'public_impact_score': pa.public_impact_score,
            'location_importance_score': pa.location_importance_score,
            'time_history_score': pa.time_history_score,
            'final_priority_score': pa.final_priority_score,
            'priority_category': pa.priority_category,
            'explanation': json.loads(pa.explanation_factors) if pa.explanation_factors else [],
            'computed_at': pa.computed_at.isoformat(),
        }
    return jsonify(data)


@incidents_bp.route('/<incident_uuid>/status', methods=['PATCH'])
def update_incident_status(incident_uuid):
    """Update incident status."""
    incident = db.session.get(Incident, incident_uuid)
    if not incident:
        return jsonify({'error': 'Incident not found'}), 404

    data = request.get_json()
    new_status = data.get('status')
    assigned_to = data.get('assigned_to')
    valid_statuses = ['Submitted', 'AI Analysed', 'Under Review', 'Assigned', 'In Progress', 'Resolved']

    if new_status and new_status not in valid_statuses:
        return jsonify({'error': f'Invalid status'}), 400

    if new_status:
        incident.status = new_status
        if new_status == 'Resolved':
            incident.resolved_at = datetime.utcnow()
    if assigned_to is not None:
        incident.assigned_to = assigned_to
    incident.updated_at = datetime.utcnow()

    db.session.commit()
    return jsonify({'success': True, 'status': incident.status})


@incidents_bp.route('/<incident_uuid>/priority', methods=['POST'])
def recompute_priority(incident_uuid):
    """Recompute priority for an incident."""
    incident = db.session.get(Incident, incident_uuid)
    if not incident:
        return jsonify({'error': 'Incident not found'}), 404

    data = request.get_json() or {}
    # Allow manual overrides
    if 'severity' in data:
        incident.severity = float(data['severity'])
    if 'public_impact' in data:
        incident.public_impact = float(data['public_impact'])
    if 'location_importance' in data:
        incident.location_importance = float(data['location_importance'])

    priority = analyze_incident_priority(incident)
    incident.priority_score = priority['priority_score']
    incident.priority_category = priority['priority_category']
    incident.time_history_score = priority['time_history']
    incident.priority_explanation = json.dumps(priority['explanation'])

    pa = PriorityAnalysis(
        incident_id=incident.id,
        severity_score=priority['severity'],
        public_impact_score=priority['public_impact'],
        location_importance_score=priority['location_importance'],
        time_history_score=priority['time_history'],
        final_priority_score=priority['priority_score'],
        priority_category=priority['priority_category'],
        explanation_factors=json.dumps(priority['explanation']),
        computed_by='admin_manual',
    )
    db.session.add(pa)
    db.session.commit()

    return jsonify({
        'success': True,
        'priority': priority,
    })


@incidents_bp.route('/duplicates/pending', methods=['GET'])
def get_pending_duplicates():
    """Get duplicate matches pending review."""
    matches = DuplicateMatch.query.filter_by(reviewed=False).all()
    result = []
    for m in matches:
        complaint = db.session.get(Complaint, m.complaint_id)
        incident = db.session.get(Incident, m.incident_id)
        if complaint and incident:
            result.append({
                'match_id': m.id,
                'complaint_id': complaint.complaint_id,
                'complaint_uuid': m.complaint_id,
                'incident_code': incident.incident_code,
                'incident_uuid': m.incident_id,
                'incident_title': incident.title,
                'geo_distance_km': m.geo_distance_km,
                'visual_similarity': m.visual_similarity,
                'combined_score': m.combined_score,
                'match_type': m.match_type,
                'created_at': m.created_at.isoformat(),
            })
    return jsonify({'matches': result, 'total': len(result)})


@incidents_bp.route('/duplicates/<match_id>/review', methods=['PATCH'])
def review_duplicate(match_id):
    """Review a duplicate match (confirm or reject)."""
    match = db.session.get(DuplicateMatch, match_id)
    if not match:
        return jsonify({'error': 'Match not found'}), 404

    data = request.get_json()
    action = data.get('action')  # 'confirmed' or 'rejected'
    reviewed_by = data.get('reviewed_by', 'admin')

    if action not in ('confirmed', 'rejected'):
        return jsonify({'error': 'Action must be confirmed or rejected'}), 400

    match.reviewed = True
    match.review_action = action
    match.reviewed_by = reviewed_by
    match.reviewed_at = datetime.utcnow()

    if action == 'confirmed':
        # Update complaint's duplicate status
        complaint = db.session.get(Complaint, match.complaint_id)
        if complaint:
            complaint.is_duplicate = True
            complaint.duplicate_of = match.incident_id
            complaint.incident_id = match.incident_id
            # Increment incident complaint count and recalculate priority.
            # Bug 3 fix: previously only complaint_count was incremented;
            # updated_at and priority fields were not updated.
            incident = db.session.get(Incident, match.incident_id)
            if incident:
                incident.complaint_count += 1
                incident.updated_at = datetime.utcnow()

                # Recalculate priority with the new complaint count
                priority = analyze_incident_priority(incident)
                incident.priority_score = priority['priority_score']
                incident.priority_category = priority['priority_category']
                incident.priority_explanation = json.dumps(priority['explanation'])

                # Persist an updated PriorityAnalysis record
                pa = PriorityAnalysis.query.filter_by(incident_id=incident.id)\
                    .order_by(PriorityAnalysis.computed_at.desc()).first()
                if pa:
                    pa.time_history_score = priority['time_history']
                    pa.final_priority_score = priority['priority_score']
                    pa.priority_category = priority['priority_category']
                    pa.explanation_factors = json.dumps(priority['explanation'])
                    pa.computed_at = datetime.utcnow()
                    pa.computed_by = 'admin_confirm'

    db.session.commit()
    return jsonify({'success': True, 'action': action})


def _incident_to_dict(i):
    priority_exp = []
    if i.priority_explanation:
        try:
            priority_exp = json.loads(i.priority_explanation)
        except Exception:
            priority_exp = [i.priority_explanation]
    return {
        'id': i.id,
        'incident_code': i.incident_code,
        'category': i.category,
        'sub_category': i.sub_category,
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
        'image_path': i.image_path,
        'image_url': f'/api/complaints/uploads/{i.image_path}' if i.image_path else None,
        'ml_category': i.ml_category,
        'ml_confidence': i.ml_confidence,
        'ml_status': i.ml_status,
        'assigned_to': i.assigned_to,
        'resolved_at': i.resolved_at.isoformat() if i.resolved_at else None,
        'created_at': i.created_at.isoformat() if i.created_at else None,
        'updated_at': i.updated_at.isoformat() if i.updated_at else None,
    }

