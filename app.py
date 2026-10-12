"""
app.py
------
Flask Backend Application for "An ML-Based Thyroid Disorder Prediction".
Serves:
1. One-frame interactive web application (index.html)
2. Patient diagnosis history management (history.html)
3. Printable medical diagnostic report (report.html)
4. REST API endpoints for image loading, preprocessing, feature extraction, training, prediction, and SQLite database operations.
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import database
import ml_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SAMPLES_FOLDER = os.path.join(BASE_DIR, "sample_images")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "static", "processed")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

database.init_db()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Renders the unified one-frame ML pipeline application."""
    samples = []
    if os.path.exists(SAMPLES_FOLDER):
        for f in sorted(os.listdir(SAMPLES_FOLDER)):
            if allowed_file(f):
                cat = "normal"
                if "hypo" in f.lower(): cat = "hypothyroid"
                elif "hyper" in f.lower(): cat = "hyperthyroid"
                elif "nodule" in f.lower(): cat = "nodule"
                samples.append({"filename": f, "category": cat, "title": cat.capitalize()})
                
    metrics_file = os.path.join(MODELS_FOLDER, "metrics.json")
    cached_metrics = None
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as mf:
                cached_metrics = json.load(mf)
        except Exception:
            cached_metrics = None

    stats = database.get_statistics()
    return render_template('index.html', samples=samples, metrics=cached_metrics, stats=stats)

@app.route('/history')
def history_page():
    records = database.get_all_predictions()
    stats = database.get_statistics()
    return render_template('history.html', records=records, stats=stats)

@app.route('/report/<int:record_id>')
def report_page(record_id):
    record = database.get_prediction_by_id(record_id)
    if not record:
        return "Record not found", 404
        
    features = {}
    if record.get("features_summary"):
        try:
            features = json.loads(record["features_summary"])
        except Exception:
            features = {}
            
    return render_template('report.html', record=record, features=features)

@app.route('/api/load-sample', methods=['POST'])
def api_load_sample():
    data = request.get_json() or {}
    sample_name = data.get('filename')
    if not sample_name:
        return jsonify({"success": False, "error": "Filename is required"}), 400
        
    sample_path = os.path.join(SAMPLES_FOLDER, sample_name)
    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": "Sample file not found"}), 404
        
    unique_name = f"sample_{uuid.uuid4().hex[:8]}_{secure_filename(sample_name)}"
    dest_path = os.path.join(UPLOAD_FOLDER, unique_name)
    
    with open(sample_path, 'rb') as f_src, open(dest_path, 'wb') as f_dst:
        f_dst.write(f_src.read())

    img = ml_pipeline.load_image(dest_path)
    b64_uri = ml_pipeline.encode_img_to_base64(img)

    return jsonify({
        "success": True,
        "filename": unique_name,
        "image_url": f"/static/uploads/{unique_name}",
        "image_b64": b64_uri,
        "width": img.shape[1],
        "height": img.shape[0]
    })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid image format"}), 400

    filename = f"scan_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = ml_pipeline.load_image(filepath)
    b64_uri = ml_pipeline.encode_img_to_base64(img)

    return jsonify({
        "success": True,
        "filename": filename,
        "image_url": f"/static/uploads/{filename}",
        "image_b64": b64_uri,
        "width": img.shape[1],
        "height": img.shape[0]
    })

@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    data = request.get_json() or {}
    filename = data.get('filename')
    if not filename:
        return jsonify({"success": False, "error": "Image filename required"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not found"}), 404

    try:
        img = ml_pipeline.load_image(filepath)
        _, previews = ml_pipeline.preprocess_image(img)
        return jsonify({"success": True, "previews": previews})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/extract-features', methods=['POST'])
def api_extract_features():
    data = request.get_json() or {}
    filename = data.get('filename')
    if not filename:
        return jsonify({"success": False, "error": "Image filename required"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not found"}), 404

    try:
        img = ml_pipeline.load_image(filepath)
        prep_dict, _ = ml_pipeline.preprocess_image(img)
        features_dict, _ = ml_pipeline.extract_features(prep_dict)
        return jsonify({"success": True, "features_all": features_dict})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        metrics_payload = ml_pipeline.train_all_models()
        best_name = metrics_payload["best_model"]
        best_summary = metrics_payload["models_evaluated"][best_name]
        database.save_model_metrics(
            best_name,
            best_summary["accuracy"],
            best_summary["precision"],
            best_summary["recall"],
            best_summary["f1_score"],
            best_summary["confusion_matrix"]
        )
        return jsonify({
            "success": True,
            "metrics": metrics_payload,
            "message": f"Successfully trained! Best: {best_name} ({best_summary['accuracy']}%)"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    filename = data.get('filename')
    model_choice = data.get('model', 'best')
    patient_id = data.get('patient_id')
    patient_name = data.get('patient_name', 'Anonymous Patient')
    patient_age = data.get('patient_age', 38)
    gender = data.get('gender', 'Female')
    doctor_notes = data.get('doctor_notes', '')

    if not filename:
        return jsonify({"success": False, "error": "Image filename required"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not found"}), 404

    try:
        result = ml_pipeline.predict_image(filepath, model_choice=model_choice)
        record_id = database.save_prediction(
            patient_id=patient_id,
            name=patient_name,
            age=patient_age,
            gender=gender,
            image_name=filename,
            prediction=result["display_title"],
            confidence=result["confidence"],
            severity=result["severity"],
            notes=doctor_notes,
            features_dict=result["features"]
        )
        result["record_id"] = record_id
        result["success"] = True
        result["patient_name"] = patient_name
        result["patient_id"] = patient_id
        result["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def api_get_history():
    records = database.get_all_predictions()
    stats = database.get_statistics()
    return jsonify({"success": True, "records": records, "stats": stats})

@app.route('/api/history/delete/<int:record_id>', methods=['POST', 'DELETE'])
def api_delete_history(record_id):
    database.delete_prediction(record_id)
    return jsonify({"success": True, "message": "Record deleted"})

@app.route('/sample_images/<filename>')
def serve_sample(filename):
    return send_from_directory(SAMPLES_FOLDER, filename)

if __name__ == '__main__':
    print("=" * 70)
    print("Starting ThyroScan AI - ML-Based Thyroid Disorder Prediction")
    print("Open your browser at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host='127.0.0.1', port=5000, debug=True)