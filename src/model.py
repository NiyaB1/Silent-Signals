## Train model and detect anomalies based on features extracted in feature_extraction.py
## This is the core ML logic for stress detection. 

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class BaselineModel:
    """Trainable baseline using Isolation Forest."""

    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.scaler = None
        self.model = None
        self.feature_cols = None

    def fit(self, features: pd.DataFrame, feature_cols=None):
        if feature_cols is None:
            # use numeric columns except identifiers
            feature_cols = [c for c in features.columns if c not in ("session_id", "window_id")]
        self.feature_cols = feature_cols

        X = features[self.feature_cols].fillna(0).values
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.model = IsolationForest(contamination=self.contamination, random_state=42)
        self.model.fit(Xs)
        return self

    def score(self, features: pd.DataFrame):
        X = features[self.feature_cols].fillna(0).values
        Xs = self.scaler.transform(X)
        # IsolationForest.score_samples gives higher = more normal
        return self.model.score_samples(Xs)


class StressDetector:
    """Evaluate windows against a baseline model."""

    def __init__(self, baseline: BaselineModel):
        self.baseline = baseline

    def evaluate(self, features: pd.DataFrame):
        scores = self.baseline.score(features)
        # transform score to anomaly strength: lower score -> higher risk
        # IsolationForest scores are roughly in (-inf, 0], so normalize
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        # invert so 0 = normal, 1 = anomalous
        risk = 1.0 - norm
        out = features.copy()
        out["anomaly_score"] = scores
        out["stress_risk"] = risk
        out["is_anomaly"] = out["stress_risk"] > 0.7
        return out


def train_baseline_from_features(feature_df: pd.DataFrame, contamination=0.05):
    model = BaselineModel(contamination=contamination)
    model.fit(feature_df)
    return model


if __name__ == "__main__":
    # simple self-test: load features and train
    try:
        from feature_extraction import extract_features
        print("Loading features...")
        feats = extract_features()
        if feats.empty:
            print("No features found. Run logger to collect data first.")
        else:
            print(f"Training on {len(feats)} windows...")
            m = train_baseline_from_features(feats)
            det = StressDetector(m)
            res = det.evaluate(feats)
            print(res[["session_id", "window_id", "stress_risk", "is_anomaly"]].head())
    except Exception as e:
        print("Model test failed:", e)
