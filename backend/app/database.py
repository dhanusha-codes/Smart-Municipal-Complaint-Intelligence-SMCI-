"""
Database Models - SQLAlchemy ORM
"""
import uuid
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def generate_uuid():
    return str(uuid.uuid4())


def generate_complaint_id():
    import random
    import string
    prefix = "SMCI"
    suffix = ''.join(random.choices(string.digits, k=6))
    return f"{prefix}-{suffix}"


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), default='citizen')  # citizen, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    complaints = db.relationship('Complaint', backref='user', lazy=True)


class Incident(db.Model):
    __tablename__ = 'incidents'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    incident_code = db.Column(db.String(20), unique=True, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    sub_category = db.Column(db.String(100), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    location_address = db.Column(db.String(300), nullable=True)
    status = db.Column(db.String(30), default='Submitted')  # Submitted, AI Analysed, Under Review, Assigned, In Progress, Resolved
    severity = db.Column(db.Float, default=5.0)
    public_impact = db.Column(db.Float, default=5.0)
    location_importance = db.Column(db.Float, default=5.0)
    time_history_score = db.Column(db.Float, default=5.0)
    priority_score = db.Column(db.Float, default=0.0)
    priority_category = db.Column(db.String(20), default='Normal')  # Critical, Urgent, Normal
    priority_explanation = db.Column(db.Text, nullable=True)
    complaint_count = db.Column(db.Integer, default=1)
    image_path = db.Column(db.String(300), nullable=True)
    image_hash = db.Column(db.String(100), nullable=True)
    image_features = db.Column(db.Text, nullable=True)  # JSON encoded features
    ml_category = db.Column(db.String(50), nullable=True)
    ml_confidence = db.Column(db.Float, nullable=True)
    ml_status = db.Column(db.String(30), default='pending')  # pending, completed, unavailable
    # reported_category: citizen's manual selection when first complaint was filed
    reported_category = db.Column(db.String(50), nullable=True)
    assigned_to = db.Column(db.String(100), nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    complaints = db.relationship('Complaint', backref='incident', lazy=True)
    duplicate_matches = db.relationship('DuplicateMatch', foreign_keys='DuplicateMatch.incident_id', backref='incident', lazy=True)
    priority_analyses = db.relationship('PriorityAnalysis', backref='incident', lazy=True)


class Complaint(db.Model):
    __tablename__ = 'complaints'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    complaint_id = db.Column(db.String(20), unique=True, nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)
    incident_id = db.Column(db.String(36), db.ForeignKey('incidents.id'), nullable=True)
    reporter_name = db.Column(db.String(100), nullable=False)
    reporter_email = db.Column(db.String(150), nullable=True)
    reporter_phone = db.Column(db.String(20), nullable=True)
    # category = authoritative category (ML prediction if available, else citizen pick)
    category = db.Column(db.String(50), nullable=False)
    # reported_category = what the citizen manually selected in the form
    reported_category = db.Column(db.String(50), nullable=True)
    # predicted_category = what the ML model independently predicted from the image
    predicted_category = db.Column(db.String(50), nullable=True)
    description = db.Column(db.Text, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    location_address = db.Column(db.String(300), nullable=True)
    image_path = db.Column(db.String(300), nullable=True)
    image_hash = db.Column(db.String(100), nullable=True)
    image_features = db.Column(db.Text, nullable=True)
    ml_category = db.Column(db.String(50), nullable=True)
    ml_confidence = db.Column(db.Float, nullable=True)
    ml_model_type = db.Column(db.String(50), nullable=True)   # pytorch / sklearn / heuristic
    ml_dataset_label = db.Column(db.String(50), nullable=True)  # raw dataset class
    ml_status = db.Column(db.String(30), default='pending')
    is_duplicate = db.Column(db.Boolean, default=False)
    duplicate_of = db.Column(db.String(36), nullable=True)  # incident_id it's a duplicate of
    status = db.Column(db.String(30), default='Submitted')
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    duplicate_matches = db.relationship('DuplicateMatch', foreign_keys='DuplicateMatch.complaint_id', backref='complaint', lazy=True)


class DuplicateMatch(db.Model):
    __tablename__ = 'duplicate_matches'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    complaint_id = db.Column(db.String(36), db.ForeignKey('complaints.id'), nullable=False)
    incident_id = db.Column(db.String(36), db.ForeignKey('incidents.id'), nullable=False)
    geo_distance_km = db.Column(db.Float, nullable=True)
    visual_similarity = db.Column(db.Float, nullable=True)
    combined_score = db.Column(db.Float, nullable=True)
    match_type = db.Column(db.String(30), nullable=True)  # exact, probable, possible, manual_review
    reviewed = db.Column(db.Boolean, default=False)
    review_action = db.Column(db.String(20), nullable=True)  # confirmed, rejected
    reviewed_by = db.Column(db.String(100), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PriorityAnalysis(db.Model):
    __tablename__ = 'priority_analyses'

    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    incident_id = db.Column(db.String(36), db.ForeignKey('incidents.id'), nullable=False)
    severity_score = db.Column(db.Float, nullable=False)
    public_impact_score = db.Column(db.Float, nullable=False)
    location_importance_score = db.Column(db.Float, nullable=False)
    time_history_score = db.Column(db.Float, nullable=False)
    final_priority_score = db.Column(db.Float, nullable=False)
    priority_category = db.Column(db.String(20), nullable=False)
    explanation_factors = db.Column(db.Text, nullable=True)  # JSON
    computed_at = db.Column(db.DateTime, default=datetime.utcnow)
    computed_by = db.Column(db.String(50), default='system')


def init_db():
    """Initialize database and create tables."""
    db.create_all()
    _seed_demo_data()


def _seed_demo_data():
    """Seed database with demo data if empty."""
    if Incident.query.count() > 0:
        return  # Already seeded

    import json
    from datetime import timedelta

    demo_incidents = [
        {
            'incident_code': 'INC-001',
            'category': 'Potholes',
            'title': 'Large pothole on MG Road',
            'description': 'Deep pothole causing vehicle damage near bus stop. Multiple vehicles affected daily.',
            'latitude': 12.9716,
            'longitude': 77.5946,
            'location_address': 'MG Road, Bangalore, Karnataka',
            'status': 'In Progress',
            'severity': 8.5,
            'public_impact': 9.0,
            'location_importance': 8.0,
            'time_history_score': 7.0,
            'priority_score': 8.45,
            'priority_category': 'Critical',
            'complaint_count': 7,
            'ml_status': 'completed',
            'ml_category': 'Potholes',
            'ml_confidence': 0.87,
        },
        {
            'incident_code': 'INC-002',
            'category': 'Garbage Accumulation',
            'title': 'Garbage pile near residential area',
            'description': 'Large garbage pile near school zone. Health hazard for children.',
            'latitude': 12.9352,
            'longitude': 77.6245,
            'location_address': 'Koramangala, Bangalore, Karnataka',
            'status': 'Under Review',
            'severity': 7.0,
            'public_impact': 8.5,
            'location_importance': 7.5,
            'time_history_score': 6.0,
            'priority_score': 7.5,
            'priority_category': 'Critical',
            'complaint_count': 4,
            'ml_status': 'completed',
            'ml_category': 'Garbage Accumulation',
            'ml_confidence': 0.92,
        },
        {
            'incident_code': 'INC-003',
            'category': 'Waterlogging',
            'title': 'Severe waterlogging at underpass',
            'description': 'Underpass floods after every rain. Traffic halts for hours.',
            'latitude': 12.9810,
            'longitude': 77.5520,
            'location_address': 'Silk Board Junction, Bangalore',
            'status': 'Assigned',
            'severity': 9.0,
            'public_impact': 9.5,
            'location_importance': 9.0,
            'time_history_score': 8.5,
            'priority_score': 9.1,
            'priority_category': 'Critical',
            'complaint_count': 12,
            'ml_status': 'completed',
            'ml_category': 'Waterlogging',
            'ml_confidence': 0.94,
        },
        {
            'incident_code': 'INC-004',
            'category': 'Damaged Roads',
            'title': 'Road surface collapse near hospital',
            'description': 'Road surface has collapsed, causing dangerous driving conditions.',
            'latitude': 12.9630,
            'longitude': 77.5900,
            'location_address': 'Jayanagar, Bangalore',
            'status': 'Submitted',
            'severity': 6.5,
            'public_impact': 7.0,
            'location_importance': 7.5,
            'time_history_score': 4.0,
            'priority_score': 6.6,
            'priority_category': 'Urgent',
            'complaint_count': 3,
            'ml_status': 'completed',
            'ml_category': 'Damaged Roads',
            'ml_confidence': 0.78,
        },
        {
            'incident_code': 'INC-005',
            'category': 'Drainage Problems',
            'title': 'Blocked storm drain causing flooding',
            'description': 'Storm drain blocked with debris, causing street flooding during rains.',
            'latitude': 12.9950,
            'longitude': 77.6020,
            'location_address': 'Hennur Main Road, Bangalore',
            'status': 'Resolved',
            'severity': 5.0,
            'public_impact': 5.5,
            'location_importance': 5.0,
            'time_history_score': 3.0,
            'priority_score': 4.95,
            'priority_category': 'Urgent',
            'complaint_count': 2,
            'ml_status': 'completed',
            'ml_category': 'Drainage Problems',
            'ml_confidence': 0.81,
            'resolved_at': datetime.utcnow() - timedelta(days=5),
        },
        {
            'incident_code': 'INC-006',
            'category': 'Street Lights',
            'title': 'Multiple street lights non-functional',
            'description': 'Stretch of 200 meters has no working streetlights. Safety concern at night.',
            'latitude': 13.0100,
            'longitude': 77.5700,
            'location_address': 'Yeshwanthpur, Bangalore',
            'status': 'AI Analysed',
            'severity': 4.0,
            'public_impact': 5.0,
            'location_importance': 4.5,
            'time_history_score': 2.0,
            'priority_score': 4.0,
            'priority_category': 'Urgent',
            'complaint_count': 1,
            'ml_status': 'completed',
            'ml_category': 'Street Lights',
            'ml_confidence': 0.73,
        },
    ]

    incident_objs = []
    for data in demo_incidents:
        resolved_at = data.pop('resolved_at', None)
        incident = Incident(**data)
        if resolved_at:
            incident.resolved_at = resolved_at
        incident.priority_explanation = json.dumps([
            f"Category: {incident.category}",
            f"Severity score: {incident.severity}/10",
            f"Complaint count: {incident.complaint_count} reports",
        ])
        db.session.add(incident)
        incident_objs.append(incident)

    db.session.flush()

    # Demo complaints
    reporter_names = ['Rahul Sharma', 'Priya Patel', 'Amit Kumar', 'Sneha Reddy', 'Vijay Nair']
    for i, incident in enumerate(incident_objs):
        complaint = Complaint(
            complaint_id=generate_complaint_id(),
            reporter_name=reporter_names[i % len(reporter_names)],
            reporter_email=f'citizen{i+1}@example.com',
            category=incident.category,
            description=incident.description,
            latitude=incident.latitude,
            longitude=incident.longitude,
            location_address=incident.location_address,
            incident_id=incident.id,
            ml_category=incident.ml_category,
            ml_confidence=incident.ml_confidence,
            ml_status=incident.ml_status,
            status=incident.status,
            submitted_at=datetime.utcnow() - timedelta(days=30-i*5),
        )
        db.session.add(complaint)

    db.session.commit()
