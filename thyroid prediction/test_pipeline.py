"""
test_pipeline.py
----------------
Automated unit verification for Flask routes and ML pipeline.
"""
import unittest
import json
from app import app
import database

class ThyroScanTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_01_index_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ThyroScan", response.data)

    def test_02_load_sample(self):
        response = self.client.post('/api/load-sample', json={'filename': 'sample_nodule_1.png'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_03_preprocess(self):
        res_sample = self.client.post('/api/load-sample', json={'filename': 'sample_normal_1.png'})
        sample_data = json.loads(res_sample.data)
        response = self.client.post('/api/preprocess', json={'filename': sample_data['filename']})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_04_extract_features(self):
        res_sample = self.client.post('/api/load-sample', json={'filename': 'sample_hypothyroid_1.png'})
        sample_data = json.loads(res_sample.data)
        response = self.client.post('/api/extract-features', json={'filename': sample_data['filename']})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_05_predict_and_sqlite_save(self):
        res_sample = self.client.post('/api/load-sample', json={'filename': 'sample_nodule_1.png'})
        sample_data = json.loads(res_sample.data)
        payload = {
            'filename': sample_data['filename'],
            'model': 'best',
            'patient_id': 'TEST-PATIENT-01',
            'patient_name': 'Test Medical Patient',
            'patient_age': 45,
            'gender': 'Female',
            'doctor_notes': 'Automated validation check'
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_06_report_page(self):
        records = database.get_all_predictions()
        if records:
            last_id = records[0]['id']
            response = self.client.get(f'/report/{last_id}')
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()