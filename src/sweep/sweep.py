import argparse
import pandas as pd
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from mlflow.models import infer_signature


# Custom Dataset for DataLoader
class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# PyTorch Model Definition
class SalesPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SalesPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# PyTorch Model Wrapper for Scikit-Learn
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, epochs=100, batch_size=32, learning_rate=0.001, device="cpu"):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.model = SalesPredictor(input_dim)
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_losses = []
        self.val_losses = []
    
    def fit(self, X, y, X_val=None, y_val=None):
        train_dataset = SalesDataset(torch.tensor(X, dtype=torch.float32).to(self.device),
                                torch.tensor(y, dtype=torch.float32).reshape(-1,1).to(self.device))
        val_dataset = SalesDataset(torch.tensor(X_val, dtype=torch.float32).to(self.device), 
                                torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(self.device))
        
        if X_val is not None and y_val is not None:
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            batch_losses = []
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                batch_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            train_loss = np.mean(batch_losses)
            self.train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(self.val_loader)
                self.val_losses.append(val_loss)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                if X_val is not None and y_val is not None:
                    print(f'Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}')

        return self

    def evaluate(self, loader):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.flatten()


class TrainingPipeline(Pipeline):
    def fit(self, X, y, X_val=None, y_val=None, **fit_params):
        if 'pytorch_regressor__X_val' in fit_params:
            del fit_params['pytorch_regressor__X_val']
        if 'pytorch_regressor__y_val' in fit_params:
            del fit_params['pytorch_regressor__y_val']
        X_transformed = self.named_steps['preprocessor'].fit_transform(X)
        X_val_transformed = self.named_steps['preprocessor'].transform(X_val) if X_val is not None else None
        self.named_steps['pytorch_regressor'].fit(X_transformed, y, X_val_transformed, y_val, **fit_params)
        return self
    

def main(args, device):
    mlflow.sklearn.autolog(registered_model_name="SalesPredictor-PyTorch",
                           log_input_examples=True)
    
    features_df, stores_df, train_df = load_data(args.input_data)
    
    train_df = preprocess_data(features_df, stores_df, train_df)

    assert set(train_df.columns.tolist()) == set(['Store', 'IsHoliday', 'Dept', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Month', 'Year', 'Week', 'Type', 'Size']), "The CSV file doesn't contain the expected columns."
    
    X, y = train_df.drop(["Weekly_Sales"], axis=1), train_df["Weekly_Sales"].values

    ct = ColumnTransformer(
        [('num-preprocessor', MinMaxScaler(), X.columns)],
    )

    training_pipeline = TrainingPipeline([
        ('preprocessor', ct),
        ('pytorch_regressor', PyTorchRegressor(input_dim=X.shape[1], epochs=args.epochs, batch_size=32, learning_rate=args.lr, device=device))
    ])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

    training_pipeline.fit(X_train, y_train, X_val, y_val)

    plt.plot(training_pipeline.named_steps['pytorch_regressor'].train_losses, label='Train Loss')
    plt.plot(training_pipeline.named_steps['pytorch_regressor'].val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()
    mlflow.log_figure(plt.gcf(), "training_validation_loss.png")

    # Evaluate on test set
    predictions = training_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error on Test Set: {mse:.4f}')
    print(f'Root Mean Squared Error on Test Set: {rmse:.4f}')

    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)

    mlflow.sklearn.save_model(training_pipeline, args.model)
    mlflow.sklearn.log_model(training_pipeline,
                     registered_model_name="SalesPredictor-PyTorch",
                     artifact_path="model",
                     input_example=X_train.iloc[:2],
                     signature=infer_signature(X_test, y_test))


def load_data(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    
    return pd.read_csv(f"{path}/features.csv"), pd.read_csv(f"{path}/stores.csv"), pd.read_csv(f"{path}/train.csv")


def preprocess_data(features_data, stores_data, train_data):
    train_df = train_data.set_index(['Store', 'Date', 'IsHoliday'])
    features_data = features_data.set_index(['Store', 'Date', 'IsHoliday'])

    train_df = merge_df_on_date(train_df, features_data).reset_index()
    train_df = add_date_to_columns(train_df)

    train_df = train_df.fillna(0)

    train_df = merge_df_on_store(train_df, stores_data)
    
    type_mapping = {'A': 3, 'B': 2, 'C': 1}
    train_df['Type'] = train_df['Type'].map(type_mapping)

    return train_df


def add_date_to_columns(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Week'] = (df.index.isocalendar().week % 4) + 1
    return df


def merge_df_on_date(df1, df2):
    return df1.join(df2)


def merge_df_on_store(df1, df2):
    return df1.merge(df2, on='Store')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='input file path')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--model', type=str, help='output model path')
    return parser.parse_args()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    args = parse_args()
    main(args, device)