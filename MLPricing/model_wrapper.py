# Lightweight wrapper for the user's ML model.
# Loads `rf.joblib` (RandomForestClassifier or similar) and optional `scaler.joblib`
# from the same folder and uses them to predict a binary label (0/1) for a single
# arc-feature dict (row from CSV). If loading or prediction fails, falls back to
# a simple heuristic.

import os
import joblib
import numpy as np

_model = None
_scaler = None
_feature_order = None
_loaded = False


def _load_model_and_scaler():
    global _model, _scaler, _feature_order, _loaded
    if _loaded:
        return
    base = os.path.dirname(__file__)
    # Candidate locations (priority): env var -> same folder -> nearby MLModels/ dir
    def _find_mlmodels_dir(start_dir):
        cur = start_dir
        for _ in range(5):
            candidate = os.path.join(cur, 'MLModels')
            if os.path.isdir(candidate):
                return candidate
            cur = os.path.dirname(cur)
            if not cur:
                break
        return None

    env_model = os.environ.get('MLP_MODEL_PATH')
    env_scaler = os.environ.get('MLP_SCALER_PATH')
    # allow selecting model name: 'rf' or 'nn' via env var MLP_MODEL_NAME
    model_name_hint = os.environ.get('MLP_MODEL_NAME', '').lower()
    same_model = os.path.join(base, 'rf.joblib')
    same_nn = os.path.join(base, 'nn.joblib')
    same_scaler = os.path.join(base, 'scaler.joblib')

    mlmodels_dir = _find_mlmodels_dir(base)
    mlmodels_model = os.path.join(mlmodels_dir, 'rf.joblib') if mlmodels_dir else None
    mlmodels_nn = os.path.join(mlmodels_dir, 'nn.joblib') if mlmodels_dir else None
    mlmodels_scaler = os.path.join(mlmodels_dir, 'scaler.joblib') if mlmodels_dir else None

    # determine model path with optional hint
    if env_model:
        model_path = env_model
    else:
        # prefer hinted model if exists
        if model_name_hint == 'nn' and os.path.exists(same_nn):
            model_path = same_nn
        elif model_name_hint == 'nn' and mlmodels_nn and os.path.exists(mlmodels_nn):
            model_path = mlmodels_nn
        else:
            # default to rf locations
            model_path = same_model if os.path.exists(same_model) else (mlmodels_model or same_model)

    scaler_path = env_scaler or (same_scaler if os.path.exists(same_scaler) else mlmodels_scaler or same_scaler)

    try:
        if os.path.exists(model_path):
            _model = joblib.load(model_path)
        else:
            _model = None
    except Exception:
        _model = None

    try:
        if os.path.exists(scaler_path):
            _scaler = joblib.load(scaler_path)
        else:
            _scaler = None
    except Exception:
        _scaler = None

    # Try to obtain feature order from model or scaler (sklearn exposes feature_names_in_)
    if _model is not None and hasattr(_model, 'feature_names_in_'):
        _feature_order = list(getattr(_model, 'feature_names_in_'))
    elif _scaler is not None and hasattr(_scaler, 'feature_names_in_'):
        _feature_order = list(getattr(_scaler, 'feature_names_in_'))
    else:
        _feature_order = None

    _loaded = True


def _build_vector_from_dict(features):
    """
    Build numeric vector from features dict using _feature_order when available,
    otherwise use deterministic ordering of keys excluding common non-feature columns.
    """
    # keys to ignore if present in CSV
    ignore_keys = {'from', 'to', 'label'}
    if _feature_order:
        keys = [k for k in _feature_order if k not in ignore_keys]
    else:
        keys = [k for k in features.keys() if k.lower() not in ignore_keys]
        # deterministic order
        keys = sorted(keys)

    vec = []
    for k in keys:
        v = features.get(k, 0.0)
        # try to coerce to float, fallback to 0.0
        try:
            fv = float(v)
        except Exception:
            # If value is boolean-like
            if isinstance(v, bool):
                fv = 1.0 if v else 0.0
            else:
                fv = 0.0
        vec.append(fv)
    return np.array(vec, dtype=float).reshape(1, -1)


def predict(features):
    """Predict 0/1 label for a single arc feature dict.

    features: dict mapping feature_name -> value (strings from CSV rows are fine).

    Returns: integer 0 or 1.
    """
    _load_model_and_scaler()

    # fallback heuristic (original behavior) in case model not available or prediction fails
    def fallback():
        try:
            c = float(features.get('cost', 0.0))
            return 1 if c < 50.0 else 0
        except Exception:
            return 0

    # prediction threshold (use probabilities if available). Default 0.5
    try:
        threshold = float(os.environ.get('MLP_PRED_THRESHOLD', '0.5'))
    except Exception:
        threshold = 0.5

    if _model is None:
        return fallback()

    try:
        x = _build_vector_from_dict(features)
        if _scaler is not None:
            try:
                x = _scaler.transform(x)
            except Exception:
                # scaler failed; continue with unscaled features
                pass

        # If model supports predict_proba, use probability + threshold to reduce false positives
        if hasattr(_model, 'predict_proba'):
            try:
                proba = _model.predict_proba(x)
                # assume positive class is column 1
                p1 = float(np.asarray(proba).ravel()[1]) if proba.shape[1] > 1 else float(np.asarray(proba).ravel()[0])
                return 1 if p1 >= threshold else 0
            except Exception:
                pass

        # fallback to predict()
        pred = _model.predict(x)
        if hasattr(pred, '__iter__'):
            label = int(np.asarray(pred).ravel()[0])
        else:
            label = int(pred)
        return 1 if label != 0 else 0
    except Exception:
        return fallback()
