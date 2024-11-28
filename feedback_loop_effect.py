"""
I wanted to analyse how feedback effect impact the outcome of a model.
A feedback loop is, here, defined as the "contamination" of training data with past model prediction. 
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

import shap
from BI_helper import plot_helper as plt_help

# %%
# 1. Generate a synthetic classification dataset

# add data drift
# Simulating feature drift by adding a time-dependent shift to some features
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

def make_data(weigths = [0.4, 0.6], drift=False):
    X, y = make_classification(n_samples=50000, n_features=20, n_informative=15, n_classes=2, random_state=42, weights=weigths)
    # add driftet features
    if drift:
        X = simulate_feature_drift(X, drift_feature_idx=[0, 1, 2, 3], drift_prop_max=2.0)
    return X, y



# %%
# make df

X, y = make_data()

data = pd.DataFrame()
data["y"] = y
data[[f"feature {i}" for i in range(X.shape[1])]] = X

for col in data.columns:
    if col != "y":
        plt.hist(data[col], label=col, alpha=0.4)
plt.legend()
plt.show()

#data.groupby("y").mean().transpose()

# %%
# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        r1 = torch.relu(self.fc1(x))  # ReLU activation
        r2 = torch.relu(self.fc2(r1))  # ReLU activation
        s1 = torch.sigmoid(self.fc3(r2))
        return s1  

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
            print(f"Early stopping at epoch {epoch+1}. Best validation accuracy: {best_acc:.4f}")
            break



# %%
clf = LogisticRegression().fit(X, y)
clf_pred = clf.predict(X)

print("label 1 count = ", clf_pred.sum())
print("false_positive = ",  ((clf_pred == 1) & (y == 0)).sum().item())
print("false_negative = ", ((clf_pred == 0) & (y == 1)).sum().item())
print("f1 score = ", f1_score(y_true=y, y_pred=clf_pred))
print("accuracy = ", accuracy_score(y_true=y, y_pred=clf_pred))

# %%
X, y = make_data(weigths=[0.9, 0.1])
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
def feedback_loop(model_type, X, y, iterations=20, retrain_epochs=100, ground_truth_prop_start_end=[1., 0.05]):
    
    # initilize model
    if model_type == "nn":
        model = SimpleNN()
    else:
        model = LogisticRegression()
    
    # Define the number of samples per iteration
    train_size = X.shape[0] // iterations
        
    # Lists for tracking accuracy
    acc = []
    acc_truth = []
    
    false_positive = []
    false_positive_truth = []
    
    false_negative = []
    false_negative_truth = []
    
    f1 = []
    f1_truth = []
    
    shap_values_over_time = []
    
    confidence = []
    
    # Define linear decreasing proportion of ground truth labels
    coeff = (ground_truth_prop_start_end[0] - ground_truth_prop_start_end[1]) / (iterations-1)
    ground_truth_prop = ground_truth_prop_start_end[0] - coeff * np.arange(iterations)
            
    for i in range(iterations):
        
        # Select the subset of data for the current iteration
        X_train = X[train_size * i : train_size * (i + 1)]
        y_truth = y[train_size * i : train_size * (i + 1)] 
        
        if not isinstance(model, LogisticRegression):
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_truth = torch.tensor(y_truth, dtype=torch.float32).unsqueeze(1)
        
        # get y_pseudo label from previous iteration
        if i==0:
            if isinstance(model, LogisticRegression):
                y_pseudo = y_truth.copy()
            else:
                y_pseudo = y_truth.clone().detach()
        else:
            if isinstance(model, LogisticRegression):
                y_pseudo = model.predict(X_train)
            else:
                with torch.no_grad():
                    y_pseudo = model(X_train).round()   
                    
            # Retrain the model using the mix of pseudo- and true labels
            idx_truth = np.random.choice(np.arange(len(y_truth)), size=int(ground_truth_prop[i] * len(y_truth)), replace=False)
            y_pseudo[idx_truth] = y_truth[idx_truth]  # Inject some true labels into pseudo-labels
                
        # weigth the loss in case of inbalance data
        pos_weight = (y_pseudo == 0).sum() / (y_pseudo == 1).sum()    
        
        if isinstance(model, LogisticRegression):
            model.fit(X_train, y_pseudo)
            outputs = model.predict_proba(X_train)
            y_pred = model.predict(X_train)
        
            # Accuracy metrics
            acc.append(accuracy_score(y_pseudo, y_pred))
            acc_truth.append(accuracy_score(y_truth, y_pred))
            
            f1.append(f1_score(y_pseudo, y_pred))
            f1_truth.append(f1_score(y_truth, y_pred))
            
            # False positives and false negatives
            fp_pseudo = ((y_pred == 1) & (y_pseudo == 0)).sum() # Predicted 1, but pseudo-label is 0
            fn_pseudo = ((y_pred == 0) & (y_pseudo == 1)).sum() # Predicted 0, but pseudo-label is 1
            fp_truth = ((y_pred == 1) & (y_truth == 0)).sum()    # Predicted 1, but true label is 0
            fn_truth = ((y_pred == 0) & (y_truth == 1)).sum()    # Predicted 0, but true label is 1
            
            confidence.append(outputs[:,1])  # Apply sigmoid for confidence interpretation
            
        else:
            # Train model with early stopping
            early_stopping(model, X_train, y_pseudo, epochs=retrain_epochs, pos_weight=pos_weight)
            with torch.no_grad():
                outputs = model(X_train)  # Model prediction (logits)
            y_pred = outputs.round()  # Predicted labels after rounding the logits
        
            # Accuracy metrics
            acc.append(accuracy_score(y_pseudo.numpy(), y_pred.numpy()))
            acc_truth.append(accuracy_score(y_truth.numpy(), y_pred.numpy()))
            
            f1.append(f1_score(y_pseudo, y_pred))
            f1_truth.append(f1_score(y_truth, y_pred))
            
            # False positives and false negatives
            fp_pseudo = ((y_pred == 1) & (y_pseudo == 0)).sum().item()  # Predicted 1, but pseudo-label is 0
            fn_pseudo = ((y_pred == 0) & (y_pseudo == 1)).sum().item()  # Predicted 0, but pseudo-label is 1
            fp_truth = ((y_pred == 1) & (y_truth == 0)).sum().item()    # Predicted 1, but true label is 0
            fn_truth = ((y_pred == 0) & (y_truth == 1)).sum().item()    # Predicted 0, but true label is 1
            
            confidence.append(outputs)  # Apply sigmoid for confidence interpretation
        
        false_positive.append(fp_pseudo)
        false_negative.append(fn_pseudo)
        false_positive_truth.append(fp_truth)
        false_negative_truth.append(fn_truth)
        
        # SHAP values
        if X_train.shape[0] > 100:  # Avoid index errors when sampling for SHAP
            X_background = X_train[torch.randint(X_train.shape[0], [100])]  # Smaller sample for SHAP background
        else:
            X_background = X_train  # Use full data if too small for 100 samples
        
        X_shap = X_train[torch.randint(X_train.shape[0], [20])]  # Sample 20 instances for SHAP calculation
        
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_background)
        else:
            explainer = shap.GradientExplainer(model, X_background)
        shap_values = explainer.shap_values(X_shap)
        shap_values_over_time.append(np.mean(np.abs(shap_values), axis=0))
                    
    return acc, acc_truth, f1, f1_truth, false_positive, false_positive_truth, false_negative, false_negative_truth, shap_values_over_time, confidence




# %%
# Run the feedback loop simulation for several class balance

weight_list = [0.9, 0.6, 0.5]
acc_list = []
acc_truth_list = []
f1_list = []
f1_truth_list = []
false_positive_list = []
false_negative_list = []
false_negative_truth_list = []
false_positive_truth_list = []
confidence_list = []
shap_list = []

for w in weight_list:
        
        X, y = make_data(weigths=[w, 1-w])
        acc, acc_truth,f1, f1_truth, false_positive, false_positive_truth, false_negative, false_negative_truth, shap_values_over_time, confidence = \
        feedback_loop("linear", X, y, iterations=50, ground_truth_prop_start_end=[1.0, 0.05])
        
        acc_list.append(acc)
        acc_truth_list.append(acc_truth)
        f1_list.append(f1)
        f1_truth_list.append(f1_truth)
        
        false_positive_list.append(false_positive)
        false_negative_list.append(false_negative)
        false_negative_truth_list.append(false_negative_truth)
        false_positive_truth_list.append(false_positive_truth)
        
        confidence_list.append(confidence)
        
        shap_list.append(shap_values_over_time)
        

# %%
# shap value
color1 = "#D4CC47"
color2 = "#7C4D8B"
col_list = plt_help.get_color_gradient(color1, color2, len(weight_list))
col_list = ["red", "blue", "black"]

# Plot the deviations from the ground truth over iterations
for a,at,c,w in zip(acc_list, acc_truth_list, col_list, weight_list):
    plt.plot(a, label=str(w), color=c)
    plt.plot(at, color=c, linestyle="dashed")
    
#plt.plot(acc_truth_list, color="red", linestyle="dashed", label="accuracy truth")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot the deviations from the ground truth over iterations
for a,at,c,w in zip(f1_list, f1_truth_list, col_list, weight_list):
    plt.plot(a, label=str(w), color=c)
    plt.plot(at, color=c, linestyle="dashed")
    
#plt.plot(acc_truth_list, color="red", linestyle="dashed", label="accuracy truth")
plt.xlabel("Iteration")
plt.ylabel("f1 score")
plt.legend()
plt.show()

# Plot the deviations from the ground truth over iterations
for a,at,c,w in zip(false_positive_list, false_positive_truth_list, col_list, weight_list):
    plt.plot(a, label=str(w), color=c)
    plt.plot(at, color=c, linestyle="dashed")
    
#plt.plot(acc_truth_list, color="red", linestyle="dashed", label="accuracy truth")
plt.xlabel("Iteration")
plt.ylabel("False positive count")
plt.legend()
plt.show()

# Plot the deviations from the ground truth over iterations
for a,at,c,w in zip(false_negative_list, false_negative_truth_list, col_list, weight_list):
    plt.plot(a, label=str(w), color=c)
    plt.plot(at, color=c, linestyle="dashed")
    
#plt.plot(acc_truth_list, color="red", linestyle="dashed", label="accuracy truth")
plt.xlabel("Iteration")
plt.ylabel("False negative count")
plt.legend()
plt.show()


# %%
# confidence
color1 = "#D4CC47"
color2 = "#7C4D8B"
col_list_tmp = plt_help.get_color_gradient(color1, color2, len(confidence_list[0]))

for conf,w in zip(confidence_list, weight_list):
    for i in range(len(conf)):
        plt.hist(conf[i], color=col_list_tmp[i], bins=10, alpha=0.4)
    plt.title(f"Class inbalance = {w}")
    plt.show()

# %%
# shap value analysis

# confidence
color1 = "#D4CC47"
color2 = "#7C4D8B"
col_list_tmp = plt_help.get_color_gradient(color1, color2, len(confidence_list[0]))

for sh,w in zip(shap_list, weight_list):
    for i in range(len(sh)):
        plt.plot(range(len(sh[i])), sh[i], color=col_list_tmp[i])
    plt.xlabel("features")
    plt.ylabel("shap value")
    plt.xticks(ticks=range(len(sh[-1])), labels=[f"feature {i}" for i in range(len(sh[-1]))], rotation=70)
    plt.title(f"Class inbalance = {w}")
    plt.show()


