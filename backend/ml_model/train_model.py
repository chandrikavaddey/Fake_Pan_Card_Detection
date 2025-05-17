import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_sample_data(num_samples=1000):
    """Generate synthetic training data"""
    np.random.seed(42)
    
    # Authentic PAN cards (class 1)
    authentic_data = {
        'blur': np.random.normal(200, 30, num_samples//2),
        'color_variation': np.random.normal(100, 20, num_samples//2),
        'edge_density': np.random.normal(0.15, 0.03, num_samples//2),
        'pan_format_valid': np.ones(num_samples//2),
        'has_name': np.ones(num_samples//2),
        'has_father_name': np.ones(num_samples//2),
        'has_dob': np.ones(num_samples//2),
        'label': np.ones(num_samples//2)
    }
    
    # Fake PAN cards (class 0)
    fake_data = {
        'blur': np.random.normal(80, 40, num_samples//2),
        'color_variation': np.random.normal(180, 40, num_samples//2),
        'edge_density': np.random.normal(0.08, 0.05, num_samples//2),
        'pan_format_valid': np.random.choice([0, 1], num_samples//2, p=[0.7, 0.3]),
        'has_name': np.random.choice([0, 1], num_samples//2, p=[0.3, 0.7]),
        'has_father_name': np.random.choice([0, 1], num_samples//2, p=[0.5, 0.5]),
        'has_dob': np.random.choice([0, 1], num_samples//2, p=[0.4, 0.6]),
        'label': np.zeros(num_samples//2)
    }
    
    # Combine and shuffle
    df_authentic = pd.DataFrame(authentic_data)
    df_fake = pd.DataFrame(fake_data)
    df = pd.concat([df_authentic, df_fake]).sample(frac=1).reset_index(drop=True)
    
    return df

def train_and_save_model():
    """Train and save the PAN card classifier"""
    data = generate_sample_data(2000)
    X = data.drop('label', axis=1)
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'pan_card_model.pkl')
    print("Model trained and saved successfully.")

if __name__ == '__main__':
    train_and_save_model()
