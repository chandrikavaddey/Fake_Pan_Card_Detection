import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

# PAN card pattern validation
PAN_PATTERN = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$')

def process_pan_image(image_path):
    """Process PAN card image and extract features"""
    features = {}
    
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Image quality metrics
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['blur'] = blur_value
        
        # 2. Color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        features['color_variation'] = np.std(hsv)
        
        # 3. Edge density
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.sum(edges) / (img.shape[0] * img.shape[1])
        
        # 4. Text extraction and validation
        text = pytesseract.image_to_string(Image.open(image_path))
        pan_number = None
        
        # Search for PAN number pattern
        matches = PAN_PATTERN.findall(text.upper().replace(' ', ''))
        if matches:
            pan_number = matches[0]
            features['pan_format_valid'] = 1
        else:
            features['pan_format_valid'] = 0
            
        # 5. Structural analysis
        text_upper = text.upper()
        features['has_name'] = 1 if "NAME" in text_upper else 0
        features['has_father_name'] = 1 if "FATHER" in text_upper else 0
        features['has_dob'] = 1 if any(word in text_upper for word in ["DOB", "DATE", "YEAR"]) else 0
        
        return features, text, pan_number
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None
