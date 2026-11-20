"""
Flask Application Factory
"""
import os
from flask import Flask
from flask_cors import CORS
from .database import db, init_db


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Configuration
    app.config['SECRET_KEY'] = 'smci-secret-key-2024'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'instance', 'smci.db'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'uploads'
    )
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'instance'), exist_ok=True)

    # Initialize extensions
    CORS(app, origins=['http://localhost:5173', 'http://localhost:3000'])
    db.init_app(app)

    # Register blueprints
    from .routes.complaints import complaints_bp
    from .routes.incidents import incidents_bp
    from .routes.admin import admin_bp
    from .routes.ml_routes import ml_bp

    app.register_blueprint(complaints_bp, url_prefix='/api/complaints')
    app.register_blueprint(incidents_bp, url_prefix='/api/incidents')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(ml_bp, url_prefix='/api/ml')

    # ── Health & dashboard aliases ────────────────────────────────────────────
    from flask import jsonify as _jsonify

    @app.route('/api/health')
    def health():
        return _jsonify({
            'status': 'ok',
            'service': 'Smart Municipal Complaint Intelligence API',
            'version': '1.0.0',
        })

    @app.route('/api/dashboard/stats')
    def dashboard_stats_alias():
        """Alias so the frontend can call /api/dashboard/stats."""
        from .routes.admin import get_dashboard_stats
        return get_dashboard_stats()

    # Initialize database
    with app.app_context():
        init_db()

    return app

