"""
ML Routes - Model status and direct inference
"""
from flask import Blueprint, request, jsonify, current_app
from ..ml.classifier import classify_image, get_ml_status
from ..utils.image_utils import allowed_file, save_uploaded_image
import os

ml_bp = Blueprint('ml', __name__)


@ml_bp.route('/status', methods=['GET'])
def ml_status():
    """Get ML system status."""
    return jsonify(get_ml_status())


@ml_bp.route('/classify', methods=['POST'])
def classify():
    """Classify an uploaded image directly."""
    file = request.files.get('image')
    description = request.form.get('description', '')

    if not file or not file.filename:
        return jsonify({'error': 'No image provided'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename, filepath = save_uploaded_image(file, current_app.config['UPLOAD_FOLDER'])
    result = classify_image(filepath, description)

    return jsonify({
        'filename': filename,
        'result': result,
    })


@ml_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get supported civic issue categories."""
    from ..ml.classifier import SMCI_CATEGORIES
    return jsonify({'categories': SMCI_CATEGORIES})

