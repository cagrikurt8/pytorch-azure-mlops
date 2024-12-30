import argparse
import pandas as pd
import os
import glob
from pathlib import Path


def main(args):
    features_df, stores_df, train_df = load_data(args.input_data)
    
    final_train_df = preprocess_data(features_df, stores_df, train_df)

    assert set(final_train_df.columns.tolist()) == set(['Store', 'IsHoliday', 'Dept', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Month', 'Year', 'Week', 'Type', 'Size']), "The CSV file doesn't contain the expected columns."
    return final_train_df.to_csv((Path(args.output_data) / "sales-forecast-train.csv"), index=False)


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
    parser.add_argument('--output_data', type=str, help='output file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)