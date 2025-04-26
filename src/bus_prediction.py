import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting import Baseline
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_forecasting.data.encoders import NaNLabelEncoder

categorical_columns = [
    "holiday",
    "weekday",
    "month",
    "day",
    "hour",
    "minute",
]
numerical_columns = ["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "pres", "coco"]


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

    drop_columns = ["delay_total", "count", "late_5_min", "early_5_min"]
    X_train = train_data.drop(columns=drop_columns)
    y_train = train_data["avg_delay"]
    X_test = test_data.drop(columns=drop_columns)
    y_test = test_data["avg_delay"]

    return X_train, y_train, X_test, y_test


class BusDelayPredictor:
    def get_context_length(self):
        return 0

    def evaluate(self, test_X, test_y):
        self.load()
        y_pred = self.predict(test_X)
        test_y = test_y[self.get_context_length() :]
        print("Shape of y_pred:", y_pred.shape)
        print("Shape of test_y:", test_y.shape)
        mse = mean_squared_error(test_y, y_pred)
        return mse

    def load(self):
        pass

    def predict(self):
        raise NotImplementedError(
            "Predict function not implemented for this BusDelayPredictor"
        )


class LSTMPredictor(BusDelayPredictor):
    def __init__(self, model, window_size=24):
        self.model = model
        self.window_size = window_size

    def get_context_length(self):
        return self.window_size

    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.001):
        X = X.drop(columns=["time_bucket"])

        self.scaler = MinMaxScaler()
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])

        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y.values, test_size=0.2, shuffle=False
        )

        train_ds = DelayDataset(X_train, y_train, self.window_size)
        val_ds = DelayDataset(X_val, y_val, self.window_size)
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
                torch.save(self.model, "lstm_predictor.pt")

    def load(self):
        self.model = torch.load("lstm_predictor.pt", weights_only=False)
        self.model.eval()

    def predict(self, X):
        X = X.drop(columns=["time_bucket"])
        X[numerical_columns] = self.scaler.transform(X[numerical_columns])
        X = X.values

        # create lagged X, just like DelayDataset
        X_lagged = []
        for i in range(self.window_size, len(X)):
            X_lagged.append(X[i - self.window_size : i])
        X = torch.tensor(X_lagged, dtype=torch.float32)

        with torch.no_grad():
            y_pred = self.model(X)

        return y_pred.squeeze().detach().numpy()


class XGBoostPredictor(BusDelayPredictor):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        lagged_features=None,
        lag_offsets=None,
        avg_features=None,
        avg_ranges=None,
    ):
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
        )
        self.lagged_features = lagged_features
        self.lag_offsets = lag_offsets
        self.avg_features = avg_features
        self.avg_ranges = avg_ranges
        self.context_length = 0

    def get_context_length(self):
        return self.context_length

    def prepare_features(self, X):
        context_length = 0
        if self.avg_features is not None:
            for feature in self.avg_features:
                for avg_range in self.avg_ranges:
                    X[f"{feature}_avg_{avg_range}"] = (
                        X[feature].rolling(avg_range, closed="left").mean()
                    )
            context_length = max(self.context_length, max(self.avg_ranges))

        if self.lagged_features is not None:
            for feature in self.lagged_features:
                for lag_offset in self.lag_offsets:
                    X[f"{feature}_lag_{lag_offset}"] = X[feature].shift(lag_offset)
            context_length = max(self.context_length, max(self.lag_offsets))

        self.context_length = context_length
        return X

    def train(self, X, y):
        X = self.prepare_features(X.copy())

        if self.context_length > 0:
            # remove the first context_length rows, since they don't have enough data
            X = X[self.context_length :]
            y = y.copy()[self.context_length :]

        X = X.drop(columns=["time_bucket", "avg_delay"])

        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y.values, test_size=0.2, shuffle=False
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        print(f"Validation MSE: {mse:.4f}")
        self.model.save_model("xgboost_predictor.json")

    def load(self):
        self.model.load_model("xgboost_predictor.json")

    def predict(self, X):
        X = self.prepare_features(X.copy())

        if self.context_length > 0:
            # remove the first context_length rows, since they don't have enough data
            X = X[self.context_length :]

        X = X.drop(columns=["time_bucket", "avg_delay"])

        y_pred = self.model.predict(X.values)
        return y_pred


class TFTPredictor(BusDelayPredictor):
    def __init__(
        self,
        max_encoder_length=24,
        max_prediction_length=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        attention_head_size=4,
    ):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_head_size = attention_head_size

    def prepare_data(self, X, y):
        data = pd.concat([X, y], axis=1)

        # add constant bus_id as group id
        data["bus_id"] = 0

        # add time index using time_bucket
        data["time_idx"] = np.arange(len(data))
        data = data.drop(columns=["time_bucket"])

        data[categorical_columns] = data[categorical_columns].astype(str)

        return data

    def train(self, X, y, batch_size=64, epochs=10, learning_rate=0.001):
        self.scaler = StandardScaler()
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])

        data = self.prepare_data(X, y)

        data_train, data_val = train_test_split(data, test_size=0.2, shuffle=False)

        self.training_ds = TimeSeriesDataSet(
            data_train,
            time_idx="time_idx",
            target="avg_delay",
            group_ids=["bus_id"],
            min_encoder_length=1,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_categoricals=categorical_columns,
            time_varying_known_reals=numerical_columns + ["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=["avg_delay"],
            target_normalizer=None,
            categorical_encoders={
                col: NaNLabelEncoder(add_nan=True) for col in categorical_columns
            },
        )

        self.validation_ds = TimeSeriesDataSet.from_dataset(
            self.training_ds,
            data_val,
            predict=True,
            stop_randomization=True,
        )

        train_loader = self.training_ds.to_dataloader(train=True, batch_size=batch_size)
        val_loader = self.validation_ds.to_dataloader(
            train=False, batch_size=batch_size
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_ds,
            learning_rate=learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=8,
            output_size=1,
            loss=RMSE(),
            log_interval=10,
        )

        self.trainer = pl.Trainer(
            max_epochs=epochs,
            gradient_clip_val=0.1,
            enable_checkpointing=True,
            enable_model_summary=True,
            callbacks=[ModelCheckpoint(monitor="val_loss")],
        )

        self.trainer.fit(self.model, train_loader, val_loader)

    def evaluate(self, X, y):
        data = self.prepare_data(X, y)

        test_ds = TimeSeriesDataSet.from_dataset(
            self.training_ds,
            data,
            predict=True,
            stop_randomization=True,
        )

        test_loader = test_ds.to_dataloader(train=False, batch_size=64)

        # best_tft = TemporalFusionTransformer.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)
        self.trainer.test(self.model, test_loader, ckpt_path="best")


class NullModel(BusDelayPredictor):
    """Model that assumes the schedules are entirely correct."""

    def predict(self, X):
        return np.array([0] * len(X))


class BaselineModel(BusDelayPredictor):
    """Model that simply uses the average for a given day of week and time of day."""

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        df = X[["weekday", "minute"]].copy()
        df["delay"] = y.values
        self.parameters = df.groupby(["weekday", "minute"]).mean()
        return self.parameters

    def predict(self, X: pd.DataFrame):
        df = X[["weekday", "minute"]]
        df = df.join(self.parameters, ["weekday", "minute"])
        return df["delay"]
