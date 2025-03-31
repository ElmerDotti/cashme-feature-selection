
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def stratified_sampling(df, test_size=0.3, random_state=42):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, _ in splitter.split(df, df["Target"]):
        return df.iloc[train_idx]

def reduce_dimensionality(df, n_components=10):
    features = df.drop(columns=["Target"])
    numeric_cols = features.select_dtypes(include=["float64", "int64"]).columns
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features[numeric_cols])
    reduced_df = pd.DataFrame(reduced, columns=[f"PCA_{i+1}" for i in range(n_components)], index=df.index)
    reduced_df["Target"] = df["Target"].values
    return reduced_df

def cluster_data(df, n_clusters=3):
    features = df.drop(columns=["Target"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(features)
    return df
