import os
import pickle
import numpy as np


class DummyModel:
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        score = -X.mean(axis=1)
        p1 = 1.0 / (1.0 + np.exp(-score))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def main():
    root = os.path.join(os.path.dirname(__file__), "MLModels")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "rf.joblib")
    with open(path, "wb") as f:
        pickle.dump(DummyModel(), f)
    print(f"Dummy model saved: {path}")


if __name__ == "__main__":
    main()
