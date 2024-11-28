"""
This script allow to investigate the SHAP value extracted from a pytorch model.
In addition to "first-order" shap value, higher-order SHAP values are obtain from feature masking.
The difference between first and higher order SHAP values should give some insight into features interaction.
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from itertools import combinations
import shap

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from helpers import *

# %%
# generate data using sklearn classification utilities

num_features = 10

X, y = make_data(num_features=num_features)

data = pd.DataFrame()
data["y"] = y
data[[f"feature {i}" for i in range(X.shape[1])]] = X

for col in data.columns:
    if col != "y":
        plt.hist(data[col], label=col, alpha=0.4)
plt.legend()
plt.show()

# %%

# Own data generation function

data = generate_data(
    num_samples=5000,
    num_features=num_features,
    interaction_formula=lambda features: to_bool((features["x1"] + features["x2"])**2 + np.sqrt(np.abs(features["x3"])) 
                                        + features["x4"] + features["x5"] + 0.2*features["x6"]
                                        + features["x7"]**2*features["x8"] + 3*features["x9"] + 0.1*features["x10"])
)

for x in data.columns:
    if x != "y":
        plt.hist(data[x], alpha=0.4, label=x, density=True)
plt.legend()
plt.show()

plt.hist(data["y"])
plt.show()

# %%
num_features = 10

model = SimpleNN(num_features=num_features)
pos_weight = data.y.mean()/(1-data.y.mean())

X, y = data.drop(columns="y").to_numpy(), data["y"].to_numpy()

# train model on all data
X_train_tensor = torch.tensor(X, dtype=torch.float32)
y_train_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

early_stopping(model, X_train_tensor, y_train_tensor, pos_weight=0.9/0.1)

outputs = model(X_train_tensor).round()

print("label 1 count = ", outputs.sum())
print("false_positive = ",  ((outputs == 1) & (y_train_tensor == 0)).sum().item())
print("false_negative = ", ((outputs == 0) & (y_train_tensor == 1)).sum().item())
print("f1 score = ", f1_score(y_true=y_train_tensor.detach().numpy(), y_pred=outputs.detach().numpy()))
print("accuracy = ", accuracy_score(y_true=y_train_tensor.detach().numpy(), y_pred=outputs.detach().numpy()))

# %%
# SHAP values first order
X_background = X_train_tensor[torch.randint(X_train_tensor.shape[0], [500])]  # Smaller sample for SHAP background
X_shap = X_train_tensor[torch.randint(X_train_tensor.shape[0], [100])]  # Sample 20 instances for SHAP calculation

explainer = shap.GradientExplainer(model, X_background)
shap_values_all_features = explainer.shap_values(X_shap)

# %%
# second order shap value

# Function to mask specific features by setting them to 0 (or another baseline) for interactions
def get_masked_shap_values(model, explainer, data, features_to_mask, baseline=0):
    masked_data = data.clone()  # Clone to avoid modifying the original data
    masked_data[:, features_to_mask] = baseline  # Set specified features to the baseline value (e.g., 0)
    shap_values_masked = explainer.shap_values(masked_data)
    return shap_values_masked

# Define feature combinations for pairwise and triple interactions
#feature_pairs = [(0, 1), (1, 2), (0, 2)]  # Example pairs
feature_pairs = list(combinations(range(X.shape[1]), 2))

# Analyze pairwise interactions
pairwise_interactions = {}
for pair in feature_pairs:
    # Compute SHAP values with the pair masked
    shap_values_pair = get_masked_shap_values(model, explainer, X_shap, pair)
    # Calculate interaction by comparing masked SHAP values with original SHAP values
    interaction_effect = shap_values_all_features - shap_values_pair
    pairwise_interactions[pair] = interaction_effect  # Store result

# %%

# Example data: pairwise_interactions dictionary where keys are pairs, and values are the interaction effects
# Assume that `pairwise_interactions` contains the mean interaction effect for each feature pair (for simplicity)
# Compute the mean interaction effect for each pair across all samples in X_shap
mean_interaction_effects = {pair: np.abs(interaction).mean() for pair, interaction in pairwise_interactions.items()}

# plot only 

# Prepare data for heatmap
num_features = np.array(list(mean_interaction_effects.keys())).flatten().max()+1
interaction_matrix = np.zeros((num_features, num_features))

# Calculate mean absolute SHAP values for each individual feature as a baseline
individual_shap_values = np.abs(shap_values_all_features).mean(axis=0)

# Compute relative interaction effect for each pair
relative_interaction_effects = {}
for (i, j), interaction in pairwise_interactions.items():
    # Mean absolute interaction effect for this pair
    mean_interaction = np.abs(interaction).mean()
    
    # Calculate the baseline effect by combining individual feature impacts
    baseline_effect = individual_shap_values[i] + individual_shap_values[j]
    
    # Calculate relative interaction strength as a proportion
    relative_interaction_strength = mean_interaction / baseline_effect if baseline_effect != 0 else 0
    relative_interaction_effects[(i, j)] = relative_interaction_strength

# Create matrix for heatmap
relative_interaction_matrix = np.zeros((num_features, num_features))
for (i, j), strength in relative_interaction_effects.items():
    relative_interaction_matrix[i, j] = strength
    relative_interaction_matrix[j, i] = strength  # Symmetric matrix

# Plot heatmap of relative interaction strengths
plt.figure(figsize=(20, 16))
sns.heatmap(relative_interaction_matrix, annot=True, fmt=".1%", cmap="coolwarm",
            xticklabels=[f'Feature {i}' for i in range(num_features)],
            yticklabels=[f'Feature {i}' for i in range(num_features)],
            cbar_kws={'label': 'Relative Interaction Strength'})
plt.title("Relative Pairwise Feature Interaction Strengths")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()



# %%
import matplotlib.pyplot as plt
import seaborn as sns

feature_1 = 'x3'  # replace with actual feature names
feature_2 = 'x7'

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data.drop(columns="y"), x=feature_1, y=feature_2, hue=data.y, palette="viridis")
plt.title(f"Interaction between {feature_1} and {feature_2} with target")
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.show()

plt.hist(data[data.y==0][feature_1], alpha=0.5)
plt.hist(data[data.y==1][feature_1], alpha=0.5)
plt.hist(data[data.y==0][feature_2], alpha=0.5)
plt.hist(data[data.y==1][feature_2], alpha=0.5)
plt.show()


