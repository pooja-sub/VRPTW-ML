# Wrapper for ML model used in pricing
# Load your trained model and provide predict() function

import os
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

_model = None
_scaler = None
_loaded = False


def _load_model():
    """Load the trained model from MLModels directory."""
    global _model, _scaler, _loaded
    if _loaded:
        return
    
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, 'MLModels')
    
    # Try to load model (rf_model.joblib, model.joblib, etc.)
    model_path = None
    for fname in ['rf_model.joblib', 'model.joblib', 'rf.joblib']:
        candidate = os.path.join(model_dir, fname)
        if os.path.exists(candidate):
            model_path = candidate
            break
    
    # Try to load scaler if available
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    if model_path and joblib:
        try:
            _model = joblib.load(model_path)
            print(f"[model_wrapper] Loaded model from {model_path}")
        except Exception as e:
            print(f"[model_wrapper] Failed to load model: {e}")
            _model = None
    
    if os.path.exists(scaler_path) and joblib:
        try:
            _scaler = joblib.load(scaler_path)
            print(f"[model_wrapper] Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"[model_wrapper] Failed to load scaler: {e}")
            _scaler = None
    
    _loaded = True


def predict(features):
    """
    Predict whether an arc should be included (1) or pruned (0).
    
    Args:
        features: dict of arc features from CSV row
    
    Returns:
        0 or 1 (binary prediction)
    """
    _load_model()
    
    if _model is None:
        # Fallback: include all arcs if model not available
        return 1
    
    try:
        # Extract feature values in a consistent order
        # Exclude non-feature columns like 'from', 'to', 'label', etc.
        exclude_cols = {'from', 'to', 'From', 'To', 'label', 'Label'}
        feature_dict = {k: v for k, v in features.items() if k not in exclude_cols}
        
        # Convert to numeric values
        X = []
        for key in sorted(feature_dict.keys()):
            try:
                X.append(float(feature_dict[key]))
            except (ValueError, TypeError):
                X.append(0.0)
        
        X = np.array(X).reshape(1, -1)
        
        # Scale if scaler available
        if _scaler is not None:
            X = _scaler.transform(X)
        
        # Predict
        prediction = _model.predict(X)[0]
        return int(prediction)
    
    except Exception as e:
        print(f"[model_wrapper] Prediction error: {e}")
        return 1  # Default to including arc on error
