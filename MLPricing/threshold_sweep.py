"""Threshold sweep helper.

Usage:
  python MLPricing/threshold_sweep.py --features ../merged_features_vrptw.csv

This script:
- Loads a features CSV with columns including 'from','to','label' and other features.
- Loads `rf.joblib`, `nn.joblib`, and `scaler.joblib` (searches MLModels/ and MLPricing/ folders).
- Computes predicted probabilities for RF and NN (when available), then evaluates a range
  of thresholds and prints confusion matrices, precision, recall, F1 for each threshold.
- Also evaluates averaged probability ensemble (RF+NN).

Results are printed and saved to `threshold_sweep_results.csv` in the current folder.
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def find_candidate(path, names):
    # try explicit path
    if path and os.path.exists(path):
        return path
    # try local folder
    here = os.path.dirname(__file__)
    for n in names:
        p = os.path.join(here, n)
        if os.path.exists(p):
            return p
    # try workspace MLModels folder upwards
    cur = here
    for _ in range(5):
        cand = os.path.join(cur, 'MLModels')
        if os.path.isdir(cand):
            for n in names:
                p = os.path.join(cand, n)
                if os.path.exists(p):
                    return p
        cur = os.path.dirname(cur)
    return None


def load_model_and_scaler(rf_path=None, nn_path=None, scaler_path=None):
    rf_p = find_candidate(rf_path, ['rf.joblib', 'rf_calib.joblib'])
    nn_p = find_candidate(nn_path, ['nn.joblib'])
    sc_p = find_candidate(scaler_path, ['scaler.joblib'])
    rf = joblib.load(rf_p) if rf_p else None
    nn = joblib.load(nn_p) if nn_p else None
    scaler = joblib.load(sc_p) if sc_p else None
    return rf, nn, scaler, rf_p, nn_p, sc_p


def make_proba(model, X):
    if model is None:
        return None
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
        if p.shape[1] > 1:
            return p[:, 1]
        else:
            return p.ravel()
    else:
        # fallback to predict -> hard 0/1 probabilities
        return model.predict(X)


def eval_thresholds(y_true, probs, thresholds):
    rows = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({'threshold': t, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                     'precision': prec, 'recall': rec, 'f1': f1})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='../merged_features_vrptw.csv', help='Features CSV with label column')
    p.add_argument('--rf', default=None, help='Path to rf.joblib (optional)')
    p.add_argument('--nn', default=None, help='Path to nn.joblib (optional)')
    p.add_argument('--scaler', default=None, help='Path to scaler.joblib (optional)')
    p.add_argument('--out', default='threshold_sweep_results.csv')
    args = p.parse_args()

    if not os.path.exists(args.features):
        print('Features file not found:', args.features)
        return

    df = pd.read_csv(args.features)
    if 'label' not in df.columns:
        print('Features CSV must contain a "label" column')
        return

    X = df.drop(columns=['from', 'to', 'label'], errors='ignore')
    y = df['label'].astype(int).values

    rf, nn, scaler, rf_p, nn_p, sc_p = load_model_and_scaler(args.rf, args.nn, args.scaler)
    print('RF model path:', rf_p)
    print('NN model path:', nn_p)
    print('Scaler path:', sc_p)

    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values

    proba_rf = make_proba(rf, Xs) if rf is not None else None
    proba_nn = make_proba(nn, Xs) if nn is not None else None

    thresholds = np.concatenate((np.linspace(0.0, 0.5, 11), np.linspace(0.55, 0.95, 9)))

    results = {}
    if proba_rf is not None:
        results['rf'] = eval_thresholds(y, proba_rf, thresholds)
    if proba_nn is not None:
        results['nn'] = eval_thresholds(y, proba_nn, thresholds)
    if proba_rf is not None and proba_nn is not None:
        proba_avg = 0.5 * (proba_rf + proba_nn)
        results['avg'] = eval_thresholds(y, proba_avg, thresholds)

    # Print best threshold by f1 and by precision >= 0.8 constraint
    out_frames = []
    for name, dfres in results.items():
        best_f1 = dfres.loc[dfres['f1'].idxmax()]
        print(f"\nModel {name} best F1: threshold={best_f1['threshold']} f1={best_f1['f1']:.4f} prec={best_f1['precision']:.4f} rec={best_f1['recall']:.4f}")
        # highest precision with recall >= 0.6 (example)
        cand = dfres[dfres['recall'] >= 0.6]
        if not cand.empty:
            best_prec = cand.loc[cand['precision'].idxmax()]
            print(f"Model {name} best precision@recall>=0.6: threshold={best_prec['threshold']} prec={best_prec['precision']:.4f} rec={best_prec['recall']:.4f} f1={best_prec['f1']:.4f}")
        out = dfres.copy()
        out['model'] = name
        out_frames.append(out)

    if out_frames:
        final = pd.concat(out_frames)
        final.to_csv(args.out, index=False)
        print('\nSaved sweep results to', args.out)
    else:
        print('No models found to evaluate')


if __name__ == '__main__':
    main()
