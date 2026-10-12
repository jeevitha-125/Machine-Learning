"""
database.py
-----------
SQLite database management for ThyroScan AI.
Stores patient diagnosis records, extracted features, and model training metrics.
"""

import sqlite3
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "thyroid_records.db")

def get_connection():
    """Returns a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database tables if they do not already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: Patient Diagnostic Predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT NOT NULL,
            patient_age INTEGER,
            gender TEXT,
            image_name TEXT,
            disorder_predicted TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT,
            doctor_notes TEXT,
            features_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table 2: Model Training Metrics History
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL NOT NULL,
            precision_score REAL NOT NULL,
            recall_score REAL NOT NULL,
            f1_score REAL NOT NULL,
            confusion_matrix_json TEXT,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully at:", DB_PATH)

def save_prediction(patient_id, name, age, gender, image_name, prediction, confidence, severity, notes, features_dict):
    """Inserts a new diagnosis record into the SQLite database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO patient_predictions 
        (patient_id, patient_name, patient_age, gender, image_name, disorder_predicted, confidence, severity, doctor_notes, features_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id or f"THY-{int(datetime.now().timestamp())%100000}",
        name or "Anonymous Patient",
        int(age) if age else 35,
        gender or "Unknown",
        image_name,
        prediction,
        float(confidence),
        severity,
        notes or "Clinical correlation recommended.",
        json.dumps(features_dict) if features_dict else "{}"
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_all_predictions():
    """Retrieves all patient records ordered by newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_prediction_by_id(record_id):
    """Retrieves a single patient record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_prediction(record_id):
    """Deletes a record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patient_predictions WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return True

def save_model_metrics(model_name, acc, prec, rec, f1, cm_list):
    """Saves evaluation metrics for a trained machine learning model."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_metrics (model_name, accuracy, precision_score, recall_score, f1_score, confusion_matrix_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (model_name, acc, prec, rec, f1, json.dumps(cm_list)))
    conn.commit()
    conn.close()

def get_latest_metrics():
    """Retrieves the latest training metrics for each model."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM training_metrics ORDER BY id DESC LIMIT 5
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_statistics():
    """Returns aggregated stats for dashboard counters."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total_patients FROM patient_predictions")
    total_patients = cursor.fetchone()["total_patients"]
    
    cursor.execute("SELECT disorder_predicted, COUNT(*) as count FROM patient_predictions GROUP BY disorder_predicted")
    distribution = {row["disorder_predicted"]: row["count"] for row in cursor.fetchall()}
    
    conn.close()
    return {
        "total_patients": total_patients,
        "distribution": distribution
    }

if __name__ == "__main__":
    init_db()