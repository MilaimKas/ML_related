
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss

import numpy as np
import pandas as pd

from scipy import stats
from scipy.special import expit  # Sigmoid


def simulate_feature_drift(X, drift_feature_idx, drift_prop_max=2., time_period=10):
    """
    Simulates feature drift by shifting the mean of specific features over time.
    
    Parameters:
    - X: The feature matrix.
    - drift_feature_idx: List of feature indices to apply the drift to.
    - drift_prop_max: maximum increase (if > 1) or decrease (if < 1) in the feature
    - time_period: number of time period where drift occurs
    """
    X_drifted = X.copy()
    num_samples = X.shape[0]//time_period
    drift_amount = np.linspace(0, drift_prop_max, time_period)
    
    # Gradually increase the drift for the chosen features over time
    for i in drift_feature_idx:
        for j in range(time_period):
            X_drifted[j*num_samples:(j+1)*num_samples, i] *= drift_amount[j]
    
    return X_drifted

def generate_data_sklearn(num_samples=1000, num_features=10, weigths = [0.4, 0.6], drift=False, n_informative=8):
    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=n_informative, n_classes=2, random_state=42, weights=weigths)
    # add driftet features
    if drift:
        X = simulate_feature_drift(X, drift_feature_idx=[0, 1, 2, 3], drift_prop_max=2.0)
    return X, y

def generate_data(num_samples=1000, num_features=3, distribution_types=None, interaction_formula=None):
    
    # Initialize features dictionary to store feature arrays
    features = {}
    
    # Define some common distributions if not provided
    if distribution_types is None:
        distribution_types = {
            "weibull": lambda size: np.random.weibull(a=2, size=size),
            "normal": lambda size: np.random.normal(loc=0, scale=1, size=size),
            "skewed_normal": lambda size: stats.skewnorm.rvs(a=5, loc=0, scale=1, size=size),
            "poisson": lambda size: np.random.poisson(lam=1, size=size),
        }
    
    # Generate feature data
    for i in range(1, num_features + 1):
        dist_name = list(distribution_types.keys())[i % len(distribution_types)]
        features[f"x{i}"] = distribution_types[dist_name](num_samples)
    
    # Default complex interaction formula if not provided
    if interaction_formula is None:
        if num_features != 3:
            raise ValueError("Default interaction formula needs 3 features")
        interaction_formula = lambda features: (features["x1"] + features["x2"])**2 + features["x3"]

    # Add complex non-linear interactions for target variable y
    y = interaction_formula(features)
    
    # Combine features and target into a dictionary
    data = {**features, "y": y}
    
    return pd.DataFrame(data)

def generate_classification_data(num_samples=1000,
                                 num_features=5,
                                 distribution_types=None,
                                 interaction_formula=None,
                                 aleatoric_strength=1.0,
                                 sparse_region=False):
    np.random.seed(42)  # Reproducibility
    features = {}

    if distribution_types is None:
        distribution_types = {
            "weibull": lambda size: np.random.weibull(a=2, size=size),
            "normal": lambda size: np.random.normal(loc=0, scale=1, size=size),
            "skewed_normal": lambda size: stats.skewnorm.rvs(a=5, loc=0, scale=1, size=size),
            "poisson": lambda size: np.random.poisson(lam=1, size=size),
        }

    # Generate feature data
    for i in range(1, num_features + 1):
        dist_name = list(distribution_types.keys())[i % len(distribution_types)]
        features[f"x{i}"] = distribution_types[dist_name](num_samples)

    if interaction_formula is None:
        if num_features != 5:
            raise ValueError("Default interaction formula needs 3 features")
        interaction_formula = lambda f: np.sin(f["x1"]) + np.log1p(np.abs(f["x2"])) - f["x3"]**2*f["x4"] + np.sqrt(np.abs(f["x5"]))

    # Interaction-based score (logits before noise)
    logit = interaction_formula(features)

    # Inject heteroscedastic aleatoric noise (optional: based on a feature)
    noise = np.random.normal(loc=0, scale=aleatoric_strength * (1 + np.abs(features["x2"])), size=num_samples)
    noisy_logit = logit + noise

    # Convert to probability using sigmoid
    prob = expit(noisy_logit)

    # Binary label from Bernoulli
    y = np.random.binomial(n=1, p=prob)

    # Optional: create epistemic uncertainty by reducing density in specific region
    if sparse_region:
        mask = (features["x1"] > 0.5) & (features["x1"] < 1.5)
        for k in features:
            features[k] = features[k][~mask]
        y = y[~mask]
        prob = prob[~mask]

    # Final DataFrame
    df = pd.DataFrame({**features, "prob": prob, "y": y})
    return df

def evaluate_classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

    return metrics
class FeatureNormalizer:
    def __init__(self):
        self.means = None
        self.stds = None
        self.feature_names = None

    def fit(self, df, feature_names):
        self.feature_names = feature_names
        self.means = df[feature_names].mean()
        self.stds = df[feature_names].std(ddof=0).replace(0, 1.0)  # prevent div-by-zero

    def transform(self, df):
        df_copy = df.copy()
        for feat in self.feature_names:
            df_copy[feat] = (df_copy[feat] - self.means[feat]) / self.stds[feat]
        return df_copy

    def fit_transform(self, df, feature_names):
        self.fit(df, feature_names)
        return self.transform(df)

    def inverse_transform(self, df_norm):
        df_copy = df_norm.copy()
        for feat in self.feature_names:
            df_copy[feat] = df_copy[feat] * self.stds[feat] + self.means[feat]
        return df_copy
