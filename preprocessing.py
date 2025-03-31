
import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.difference(["Target"]).tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", axis=0, limit_direction="both")
    df[bool_cols] = df[bool_cols].fillna(False)
    df[object_cols] = df[object_cols].fillna("missing")
    return df
