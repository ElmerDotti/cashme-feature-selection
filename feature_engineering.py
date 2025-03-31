
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler

def calculate_entropy(series):
    counts = series.value_counts(normalize=True)
    return entropy(counts)

def generate_lag_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["Target"])
    scaler = MinMaxScaler()
    for col in numeric_cols:
        df[f"{col}_lag1"] = df[col].shift(1).fillna(method="bfill")
        diff = df[col] - df[f"{col}_lag1"]
        df[f"{col}_var"] = (diff - diff.min()) / (diff.max() - diff.min())
        df[f"{col}_ent"] = calculate_entropy(df[col])
        df[f"{col}_score"] = scaler.fit_transform(df[[col]])
    return df
