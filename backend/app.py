import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import re
from sklearn.ensemble import RandomForestClassifier
import joblib
import tempfile
from datetime import datetime
from utils.image_utils import process_pan_image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model
try:
    model = joblib.load('ml_model/pan_card_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/verify_pan', methods=['POST'])
def verify_pan():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        
        # Process image and extract features
        features, extracted_text, pan_number = process_pan_image(temp_path)
        
        if features is None:
            return jsonify({'status': 'error', 'error': 'Error processing image'}), 500
            
        # Predict authenticity
        if model:
            prediction = model.predict([[features[f] for f in sorted(features)]])[0]
            proba = model.predict_proba([[features[f] for f in sorted(features)]])[0]
            authenticity = "Authentic" if prediction == 1 else "Fake"
            confidence = round(max(proba) * 100, 2)
        else:
            authenticity = "Unknown"
            confidence = 0.0
        
        # Prepare response
        response = {
            'status': 'success',
            'data': {
                'authenticity': authenticity,
                'confidence': confidence,
                'details': {
                    'Format Validation': 'Valid' if features['pan_format_valid'] else 'Invalid',
                    'Image Quality': 'Good' if features['blur'] > 100 else 'Poor',
                    'Structural Elements': 'Complete' if sum([features['has_name'], 
                                                          features['has_father_name'], 
                                                          features['has_dob']]) >= 2 else 'Incomplete',
                    'Edge Consistency': 'Consistent' if features['edge_density'] > 0.1 else 'Inconsistent',
                    'Color Profile': 'Normal' if 50 < features['color_variation'] < 150 else 'Abnormal'
                },
                'extracted_data': {
                    'PAN Number': pan_number if pan_number else 'Not detected',
                    'Name': extract_field(extracted_text, 'name'),
                    'Father Name': extract_field(extracted_text, 'father'),
                    'Date of Birth': extract_field(extracted_text, 'dob')
                }
            }
        }
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'status': 'error', 'error': str(e)}), 500

def extract_field(text, field_type):
    """Extract specific fields from OCR text"""
    text = text.lower()
    if field_type == 'name':
        # Simple pattern for demo - would be more robust in production
        if 'name' in text:
            return text.split('name')[-1].split('\n')[0].strip().title()
    elif field_type == 'father':
        if 'father' in text:
            return text.split('father')[-1].split('\n')[0].strip().title()
    elif field_type == 'dob':
        # Simple date pattern matching
        date_pattern = re.compile(r'\d{2}[\/\-]\d{2}[\/\-]\d{4}')
        match = date_pattern.search(text)
        return match.group(0) if match else 'Not detected'
    return 'Not detected'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
