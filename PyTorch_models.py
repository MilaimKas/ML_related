import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[64, 64],
        dropout_rate=0.0,
        output_type="logit",  # options: 'logit', 'beta'
    ):
        super(BaseClassifier, self).__init__()
        self.output_type = output_type
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim

        self.feature_extractor = nn.Sequential(*layers)

        if output_type == "logit":
            self.output_layer = nn.Linear(prev_dim, 1)
        elif output_type == "beta":
            self.output_layer = nn.Linear(prev_dim, 2)
        else:
            raise ValueError("Invalid output_type. Choose from 'logit', 'beta'.")

    def forward(self, x):
        x = self.feature_extractor(x)

        if self.output_type == "logit":
            logit = self.output_layer(x)
            prob = torch.sigmoid(logit)
            return prob, logit

        elif self.output_type == "beta":
            raw_out = self.output_layer(x)
            alpha = F.softplus(raw_out[:, 0]) + 1e-3  # to ensure >0
            beta = F.softplus(raw_out[:, 1]) + 1e-3
            return alpha, beta

class TorchTrainer:
    def __init__(self, model, lr=1e-3, epochs=20, batch_size=64, device=None):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history = {"train_loss": [], "val_loss": []}

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Determine loss function based on output type
        if self.model.output_type == "logit":
            self.criterion = nn.BCELoss()
        elif self.model.output_type == "beta":
            self.criterion = self.beta_bernoulli_nll
        else:
            raise ValueError("Unsupported output_type")

    def beta_bernoulli_nll(self, outputs, y_true):
        alpha, beta = outputs
        eps = 1e-8
        y_true = y_true.view(-1).to(self.device)
        term1 = y_true * torch.log((alpha + eps) / (alpha + beta + eps))
        term2 = (1 - y_true) * torch.log((beta + eps) / (alpha + beta + eps))
        return -torch.mean(term1 + term2)

    def train(self, train_loader, val_loader=None, verbose=True):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(xb)
                if self.model.output_type == "logit":
                    loss = self.criterion(outputs[0].view(-1), yb)
                else:  # beta
                    loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            self.history["train_loss"].append(avg_loss)

            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                if verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Train loss: {avg_loss:.4f} - Val loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Train loss: {avg_loss:.4f}")

    def evaluate(self, data_loader):
        self.model.eval()
        loss_total = 0.0
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(self.device), yb.to(self.device).float()
                outputs = self.model(xb)
                if self.model.output_type == "logit":
                    loss = self.criterion(outputs[0].view(-1), yb)
                else:
                    loss = self.criterion(outputs, yb)
                loss_total += loss.item()
        return loss_total / len(data_loader)

    def predict_proba(self, data_loader):
        """
        Returns: numpy array of predicted probabilities (mean of Beta or sigmoid(logit))
        """
        self.model.eval()
        preds = []

        with torch.no_grad():
            for xb, _ in data_loader:
                xb = xb.to(self.device)
                outputs = self.model(xb)

                if self.model.output_type == "logit":
                    prob = outputs[0].view(-1)
                elif self.model.output_type == "beta":
                    alpha, beta = outputs
                    prob = alpha / (alpha + beta)
                preds.append(prob.cpu())

        return torch.cat(preds).numpy()

    def predict_proba_with_uncertainty(self, data_loader):
        """
        Returns: tuple (mean, std) for each point
        """
        self.model.eval()
        mean_list, std_list = [], []

        with torch.no_grad():
            for xb, _ in data_loader:
                xb = xb.to(self.device)
                alpha, beta = self.model(xb)

                mean = alpha / (alpha + beta)
                var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
                std = torch.sqrt(var + 1e-8)

                mean_list.append(mean.cpu())
                std_list.append(std.cpu())

        return torch.cat(mean_list).numpy(), torch.cat(std_list).numpy()
