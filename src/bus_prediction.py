import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LSTMModule(nn.Module):
    def __init__(
        self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1, horizon=1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class DelayDataset(Dataset):
    def __init__(self, series, window_size, horizon):
        X, y = [], []
        for i in range(len(series) - window_size - horizon + 1):
            X.append(series[i : i + window_size])
            y.append(series[i + window_size : i + window_size + horizon])
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(
            -1
        )  # (samples, window, 1)
        self.y = torch.tensor(y, dtype=torch.float32)  # (samples, horizon)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_bus_data(data):
    # data is a pandas DataFrame
    boundary_time = pd.Timestamp("2025-01-01 00:00:00-08:00")
    # using 2024 data as training, 2025 as test
    train_data = data[data["time_bucket"] < boundary_time]
    test_data = data[data["time_bucket"] >= boundary_time]
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    return train_data, test_data


class BusDelayPredictor:
    def __init__(self, model):
        self.model = model


class LSTMPredictor(BusDelayPredictor):
    def __init__(self, model, window_size=24, horizon=1):
        super().__init__(model)
        self.model = model
        self.window_size = window_size
        self.horizon = horizon

    def train(self, X, epochs=10, batch_size=32, learning_rate=0.001):
        X["avg_delay"] = X["delay_total"] / X["count"]
        # ignoring time_bucket, using only avg_delay
        series = X["avg_delay"].values
        series = np.nan_to_num(series)

        train_cut = int(len(series) * 0.8)
        train_ds = DelayDataset(series[:train_cut], self.window_size, self.horizon)
        val_ds = DelayDataset(
            series[train_cut - self.window_size :], self.window_size, self.horizon
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for batch_X, batch_y in train_loader:
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    y_pred = self.model(batch_X)
                    loss = criterion(y_pred, batch_y)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "lstm_predictor.pt")

    def predict(self, X):
        self.modal.load_state_dict(torch.load("lstm_predictor.pt"))
        self.model.eval()
        X["avg_delay"] = X["delay_total"] / X["count"]
        series = X["avg_delay"].values
        series = np.nan_to_num(series)
        series = (
            torch.tensor(series[-self.window_size :], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        with torch.no_grad():
            y_pred = self.model(series)

        return y_pred.squeeze().numpy()
