import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class LSTMModule(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class DelayDataset(Dataset):
    def __init__(self, X, y, window_size):
        X_lagged = []
        y_lagged = []
        for i in range(window_size, len(X)):
            X_lagged.append(X[i - window_size : i])
            y_lagged.append(y[i])
        self.X = torch.tensor(X_lagged, dtype=torch.float32)
        self.y = torch.tensor(y_lagged, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_bus_data(data):
    # data is a pandas DataFrame
    boundary_time = pd.Timestamp("2025-01-01 00:00:00-08:00")

    # add avg_delay feature
    data["avg_delay"] = data["delay_total"] / data["count"]
    data["avg_delay"] = data["avg_delay"].fillna(0)

    # add time features
    data["month"] = data["time_bucket"].dt.month
    data["day"] = data["time_bucket"].dt.day
    data["hour"] = data["time_bucket"].dt.hour
    data["minute"] = data["time_bucket"].dt.minute

    # using 2024 data as training, 2025 as test
    train_data = data[data["time_bucket"] < boundary_time]
    test_data = data[data["time_bucket"] >= boundary_time]
    # print(f"Train data shape: {train_data.shape}")
    # print(f"Test data shape: {test_data.shape}")

    drop_columns = ["delay_total", "count", "time_bucket"]
    X_train = train_data.drop(columns=drop_columns)
    y_train = train_data["avg_delay"]
    X_test = test_data.drop(columns=drop_columns)
    y_test = test_data["avg_delay"]

    return X_train, y_train, X_test, y_test


class BusDelayPredictor:
    def __init__(self, model):
        self.model = model


class LSTMPredictor(BusDelayPredictor):
    def __init__(self, model, window_size=24):
        super().__init__(model)
        self.model = model
        self.window_size = window_size

    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.001):
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        train_ds = DelayDataset(X_train, y_train, self.window_size)
        val_ds = DelayDataset(X_test, y_test, self.window_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
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
