"""
Complaint Routes - Citizen complaint submission and tracking
"""
import os
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from ..database import db, Complaint, Incident, DuplicateMatch, PriorityAnalysis, generate_complaint_id
from ..utils.image_utils import (
    compute_image_hash, compute_image_features, allowed_file, save_uploaded_image
)
from ..utils.duplicate_detector import find_duplicate_incidents
from ..utils.priority_engine import (
    analyze_incident_priority, CATEGORY_SEVERITY, CATEGORY_IMPACT,
    compute_location_importance, compute_time_history_score
)
from ..ml.classifier import classify_image

complaints_bp = Blueprint('complaints', __name__)


@complaints_bp.route('', methods=['POST'])
def submit_complaint():
    """Submit a new citizen complaint."""
    try:
        # Parse form data
        reporter_name = request.form.get('reporter_name', '').strip()
        reporter_email = request.form.get('reporter_email', '').strip()
        reporter_phone = request.form.get('reporter_phone', '').strip()
        category = request.form.get('category', '').strip()
        description = request.form.get('description', '').strip()
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        location_address = request.form.get('location_address', '').strip()

        # Validation
        errors = []
        if not reporter_name:
            errors.append('Reporter name is required')
        if not category:
            errors.append('Category is required')
        if not description:
            errors.append('Description is required')
        if not latitude or not longitude:
            errors.append('Location coordinates are required')

        if errors:
            return jsonify({'success': False, 'errors': errors}), 400

        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'errors': ['Invalid coordinates']}), 400

        # Handle image upload
        image_filename = None
        image_path = None
        image_hash = None
        image_features_json = None
        ml_result = None

        file = request.files.get('image')
        if file and file.filename and allowed_file(file.filename):
            filename, filepath = save_uploaded_image(file, current_app.config['UPLOAD_FOLDER'])
            image_filename = filename
            image_path = filepath

            # Compute image hash and features
            image_hash = compute_image_hash(filepath)
            features = compute_image_features(filepath)
            if features:
                image_features_json = json.dumps(features)

            # ML Classification — independent of citizen-selected category
            ml_result = classify_image(filepath, description)

        # ── Category resolution ───────────────────────────────────────────────
        # reported_category : what the citizen manually selected
        # predicted_category: what the ML model independently predicted from image
        # category          : authoritative value used for incident grouping
        #                     → ML prediction when available, else citizen pick
        reported_category  = category                                    # citizen pick
        predicted_category = ml_result['category'] if ml_result else None  # ML result
        # Use ML prediction as the authoritative category when available
        authoritative_category = predicted_category if predicted_category else reported_category
        category_mismatch = (
            predicted_category is not None and
            predicted_category != reported_category
        )

        # Create complaint record
        complaint = Complaint(
            complaint_id=generate_complaint_id(),
            reporter_name=reporter_name,
            reporter_email=reporter_email or None,
            reporter_phone=reporter_phone or None,
            category=authoritative_category,          # ML prediction or citizen pick
            reported_category=reported_category,       # always citizen's selection
            predicted_category=predicted_category,     # always ML's independent result
            description=description,
            latitude=latitude,
            longitude=longitude,
            location_address=location_address or None,
            image_path=image_filename,
            image_hash=image_hash,
            image_features=image_features_json,
            ml_category=ml_result['category'] if ml_result else None,
            ml_confidence=ml_result['confidence'] if ml_result else None,
            ml_model_type=ml_result.get('model_type') if ml_result else None,
            ml_dataset_label=ml_result.get('dataset_label') if ml_result else None,
            ml_status=ml_result['status'] if ml_result else 'no_image',
            status='Submitted',
        )
        db.session.add(complaint)
        db.session.flush()  # get complaint.id

        # Duplicate detection — search against authoritative category
        all_incidents = Incident.query.filter_by(category=authoritative_category).all()
        complaint_data = {
            'category': authoritative_category,
            'latitude': latitude,
            'longitude': longitude,
            'image_hash': image_hash,
            'image_features': image_features_json,
        }
        duplicate_matches = find_duplicate_incidents(complaint_data, all_incidents, current_app.config['UPLOAD_FOLDER'])

        incident = None
        is_duplicate = False
        duplicate_match_data = None

        # Find best matching unresolved incident
        best_match = next(
            (m for m in duplicate_matches if m['match_type'] in ('exact', 'probable') and not m['is_resolved']),
            None
        )

        if best_match:
            # Link to existing incident
            incident = db.session.get(Incident, best_match['incident_id'])
            if incident:
                incident.complaint_count += 1
                incident.updated_at = datetime.utcnow()
                complaint.incident_id = incident.id
                complaint.is_duplicate = True
                complaint.duplicate_of = incident.id
                complaint.status = 'AI Analysed'
                is_duplicate = True

                # Record the match — mark as already reviewed (auto-merged by system)
                # Bug 4 fix: exact/probable matches that were auto-merged should NOT
                # appear in the Admin "Pending Review" queue, because the decision was
                # already made automatically. Mark them reviewed=True with action='auto_merged'.
                dm = DuplicateMatch(
                    complaint_id=complaint.id,
                    incident_id=incident.id,
                    geo_distance_km=best_match['geo_distance_km'],
                    visual_similarity=best_match['visual_similarity'],
                    combined_score=best_match['combined_score'],
                    match_type=best_match['match_type'],
                    reviewed=True,
                    review_action='auto_merged',
                    reviewed_by='system',
                    reviewed_at=datetime.utcnow(),
                )
                db.session.add(dm)
                duplicate_match_data = best_match
        else:
            # Create new incident
            incident_count = Incident.query.count() + 1
            incident_code = f"INC-{incident_count:03d}"
            # Ensure uniqueness
            while Incident.query.filter_by(incident_code=incident_code).first():
                incident_count += 1
                incident_code = f"INC-{incident_count:03d}"

            severity = CATEGORY_SEVERITY.get(authoritative_category, 5.0)
            public_impact = CATEGORY_IMPACT.get(authoritative_category, 5.0)
            loc_importance = compute_location_importance(latitude, longitude, location_address)
            time_score = compute_time_history_score(1, 0, False)

            incident = Incident(
                incident_code=incident_code,
                category=authoritative_category,          # ML prediction or citizen pick
                reported_category=reported_category,       # citizen's original selection
                title=f"{authoritative_category} at {location_address or 'reported location'}",
                description=description,
                latitude=latitude,
                longitude=longitude,
                location_address=location_address or None,
                status='AI Analysed' if ml_result else 'Submitted',
                severity=severity,
                public_impact=public_impact,
                location_importance=loc_importance,
                time_history_score=time_score,
                image_path=image_filename,
                image_hash=image_hash,
                image_features=image_features_json,
                ml_category=ml_result['category'] if ml_result else None,
                ml_confidence=ml_result['confidence'] if ml_result else None,
                ml_status=ml_result['status'] if ml_result else 'no_image',
                complaint_count=1,
            )

            # Priority analysis
            priority = analyze_incident_priority(incident)
            incident.priority_score = priority['priority_score']
            incident.priority_category = priority['priority_category']
            incident.priority_explanation = json.dumps(priority['explanation'])

            db.session.add(incident)
            db.session.flush()

            complaint.incident_id = incident.id

            # Record priority analysis
            pa = PriorityAnalysis(
                incident_id=incident.id,
                severity_score=severity,
                public_impact_score=public_impact,
                location_importance_score=loc_importance,
                time_history_score=time_score,
                final_priority_score=priority['priority_score'],
                priority_category=priority['priority_category'],
                explanation_factors=json.dumps(priority['explanation']),
            )
            db.session.add(pa)

        # Also save possible/manual_review matches
        for match in duplicate_matches:
            if match['match_type'] in ('possible', 'manual_review') and match['incident_id'] != (incident.id if incident else None):
                dm = DuplicateMatch(
                    complaint_id=complaint.id,
                    incident_id=match['incident_id'],
                    geo_distance_km=match['geo_distance_km'],
                    visual_similarity=match['visual_similarity'],
                    combined_score=match['combined_score'],
                    match_type=match['match_type'],
                )
                db.session.add(dm)

        db.session.commit()

        response = {
            'success': True,
            'complaint_id': complaint.complaint_id,
            'complaint_uuid': complaint.id,
            'incident_code': incident.incident_code if incident else None,
            'is_duplicate': is_duplicate,
            'ml_result': ml_result,
            # Category breakdown for the frontend
            'reported_category': reported_category,
            'predicted_category': predicted_category,
            'authoritative_category': authoritative_category,
            'category_mismatch': category_mismatch,
            'duplicate_matches': duplicate_match_data,
            'message': (
                f"Complaint {complaint.complaint_id} submitted successfully. "
                + (f"Linked to existing incident {incident.incident_code}." if is_duplicate
                   else f"New incident {incident.incident_code} created.")
            )
        }
        return jsonify(response), 201

    except Exception as e:
        db.session.rollback()
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@complaints_bp.route('', methods=['GET'])
def list_complaints():
    """List all complaints with optional filters."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    status = request.args.get('status')
    category = request.args.get('category')
    complaint_id = request.args.get('complaint_id')

    query = Complaint.query

    if status:
        query = query.filter_by(status=status)
    if category:
        query = query.filter_by(category=category)
    if complaint_id:
        query = query.filter_by(complaint_id=complaint_id)

    query = query.order_by(Complaint.submitted_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'complaints': [_complaint_to_dict(c) for c in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
    })


@complaints_bp.route('/<complaint_id_str>', methods=['GET'])
def get_complaint(complaint_id_str):
    """Get complaint by complaint_id or UUID."""
    complaint = Complaint.query.filter(
        (Complaint.complaint_id == complaint_id_str) | (Complaint.id == complaint_id_str)
    ).first()

    if not complaint:
        return jsonify({'error': 'Complaint not found'}), 404

    data = _complaint_to_dict(complaint)
    if complaint.incident_id:
        incident = db.session.get(Incident, complaint.incident_id)
        if incident:
            data['incident'] = {
                'incident_code': incident.incident_code,
                'category': incident.category,
                'status': incident.status,
                'priority_category': incident.priority_category,
                'complaint_count': incident.complaint_count,
            }
    return jsonify(data)


@complaints_bp.route('/<complaint_uuid>/status', methods=['PATCH'])
def update_complaint_status(complaint_uuid):
    """Update complaint status."""
    complaint = Complaint.query.filter(
        (Complaint.complaint_id == complaint_uuid) | (Complaint.id == complaint_uuid)
    ).first()
    if not complaint:
        return jsonify({'error': 'Complaint not found'}), 404

    data = request.get_json() or {}
    new_status = data.get('status')
    valid_statuses = ['Submitted', 'AI Analysed', 'Under Review', 'Assigned', 'In Progress', 'Resolved']

    if new_status not in valid_statuses:
        return jsonify({'error': f'Invalid status. Valid: {valid_statuses}'}), 400

    complaint.status = new_status
    complaint.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True, 'status': new_status})


@complaints_bp.route('/<complaint_uuid>', methods=['DELETE'])
def delete_complaint(complaint_uuid):
    """Delete a complaint (admin removal)."""
    complaint = Complaint.query.filter(
        (Complaint.complaint_id == complaint_uuid) | (Complaint.id == complaint_uuid)
    ).first()
    if not complaint:
        return jsonify({'error': 'Complaint not found'}), 404

    cid = complaint.complaint_id

    # Remove linked duplicate matches
    DuplicateMatch.query.filter_by(complaint_id=complaint.id).delete()

    # If linked to an incident, decrement complaint count and recalculate priority.
    # Bug 2 fix: previously only complaint_count was updated; the incident's
    # priority_score and priority_category were left stale after deletion.
    if complaint.incident_id:
        incident = db.session.get(Incident, complaint.incident_id)
        if incident:
            incident.complaint_count = max(0, incident.complaint_count - 1)
            incident.updated_at = datetime.utcnow()

            # Recalculate priority with the new complaint count
            priority = analyze_incident_priority(incident)
            incident.priority_score = priority['priority_score']
            incident.priority_category = priority['priority_category']
            incident.priority_explanation = json.dumps(priority['explanation'])

            # Update the most recent PriorityAnalysis record for this incident
            pa = PriorityAnalysis.query.filter_by(incident_id=incident.id)\
                .order_by(PriorityAnalysis.computed_at.desc()).first()
            if pa:
                pa.time_history_score = priority['time_history']
                pa.final_priority_score = priority['priority_score']
                pa.priority_category = priority['priority_category']
                pa.explanation_factors = json.dumps(priority['explanation'])
                pa.computed_at = datetime.utcnow()
                pa.computed_by = 'system_on_delete'

    db.session.delete(complaint)
    db.session.commit()
    return jsonify({'success': True, 'message': f'Complaint {cid} removed successfully.'})


@complaints_bp.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


def _complaint_to_dict(c):
    reported  = getattr(c, 'reported_category', None)
    predicted = getattr(c, 'predicted_category', None) or c.ml_category
    mismatch  = bool(predicted and reported and predicted != reported)
    return {
        'id': c.id,
        'complaint_id': c.complaint_id,
        'reporter_name': c.reporter_name,
        'reporter_email': c.reporter_email,
        'reporter_phone': c.reporter_phone,
        # Authoritative category (what the system acts on)
        'category': c.category,
        # Citizen's original manual selection
        'reported_category': reported or c.category,
        # ML model's independent prediction
        'predicted_category': predicted,
        # Whether they differ
        'category_mismatch': mismatch,
        'description': c.description,
        'latitude': c.latitude,
        'longitude': c.longitude,
        'location_address': c.location_address,
        'image_path': c.image_path,
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
        'updated_at': c.updated_at.isoformat() if c.updated_at else None,
    }

