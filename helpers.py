
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import numpy as np
import pandas as pd
import scipy.stats as stats


def early_stopping(model, X, y, epochs=100, patience=10, lr=0.001, weight_decay=0., pos_weight=None):
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_acc = 0  # Best accuracy seen so far
    patience_counter = 0  # Counts how many epochs without improvement
    
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])) #nn.BCELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        
        model.train()
        
        # Training step
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation step: calculate accuracy on the validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).round()  # Predict and round to binary
            #val_acc = accuracy_score(y_val.numpy(), val_outputs.numpy())
            val_acc = f1_score(y_val.numpy(), val_outputs.numpy())
        
        # Print accuracy and loss at each epoch
        #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Early stopping logic: check if accuracy improved
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
        
        # Stop training if patience has been exceeded
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best validation f1 score: {best_acc:.4f}")
            break

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

def make_data(weigths = [0.4, 0.6], drift=False, num_features=10, n_informative=8):
    X, y = make_classification(n_samples=50000, n_features=num_features, n_informative=n_informative, n_classes=2, random_state=42, weights=weigths)
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

# define boolean mapping
def to_bool(array, threshold=0):
    return 1*(array > threshold)

    
class SimpleNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer1 = nn.Linear(num_features, 60)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x
