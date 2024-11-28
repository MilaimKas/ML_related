"""
Complex set of data are generated.
A Bayesian neural network is trained and compared to its determinitstic counterpart.
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

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

# %%

# Create data generated with noise and non-linearity

num_features = 10

data = generate_data(
    num_samples=10000,
    num_features=num_features,
    interaction_formula=lambda features: to_bool((features["x1"] + features["x2"])**2 + np.sqrt(np.abs(features["x3"])) 
                                        + features["x4"] + features["x5"] + 0.2*features["x6"]
                                        + features["x7"]**2*features["x8"] + 3*features["x9"] + 0.1*features["x10"],
                                        threshold=0)
)

for x in data.columns:
    if x != "y":
        plt.hist(data[x], alpha=0.4, label=x, density=True)
plt.legend()
plt.show()

plt.hist(data["y"])
plt.show()

# %%
# train Deterministic NN

num_features = 10
learning_rate = 0.001
#batch_size = 64 

model = SimpleNN(num_features=num_features)
pos_weight = None#data.y.mean()/(1-data.y.mean())

X, y = data.drop(columns="y").to_numpy(), data["y"].to_numpy()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model on all data
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

early_stopping(model, X_train_tensor, y_train_tensor, pos_weight=pos_weight, lr=learning_rate)

outputs = model(X_test_tensor).round()

print("label 1 ratio = ", outputs.mean())
print("false_positive = ",  ((outputs == 1) & (y_test_tensor == 0)).sum().item())
print("false_negative = ", ((outputs == 0) & (y_test_tensor == 1)).sum().item())
print("f1 score = ", f1_score(y_true=y_test_tensor.detach().numpy(), y_pred=outputs.detach().numpy()))
print("accuracy = ", accuracy_score(y_true=y_test_tensor.detach().numpy(), y_pred=outputs.detach().numpy()))

# %%
y_train_tensor.mean()

# %%
# define Bayesian NN from pre-trained DNN

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}
    
dnn_to_bnn(model, const_bnn_prior_parameters)


# %%
# train BNN

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), learning_rate)

#output = model(X_train_tensor)
#loss = criterion(output, y_train_tensor)

#loss.backward()
#optimizer.step()

early_stopping(model, X_train_tensor, y_train_tensor, pos_weight=pos_weight, lr=learning_rate)


# %%
# testing

num_monte_carlo_sample = 100

model.eval()
with torch.no_grad():
    output_mc = []
    for mc_run in range(num_monte_carlo_sample):
        probs = model(X_test_tensor)
        #probs = nn.functional.softmax(logits, dim=-1) # softmax already as
        output_mc.append(probs)
    output = torch.stack(output_mc)  
    pred_mean = output.mean(dim=0)
    pred_std = output.std(dim=0)
    y_pred = torch.argmax(pred_mean, axis=-1)
    test_acc = (y_pred.data.cpu().numpy() == y_test_tensor.data.cpu().numpy()).mean()

# %%
plt.hist(pred_std)
plt.show()


