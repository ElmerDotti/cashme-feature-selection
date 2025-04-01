import streamlit as st
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

# =============================
# Utilitários
# =============================

@st.cache_data
def load_large_csv(file, nrows=5000):
    return pd.read_csv(file, nrows=nrows, dtype='float32', low_memory=False)

def create_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col == "Target":
            continue
        std = df[col].std()
        mean = df[col].mean()
        if std > 0:
            score = (df[col] - mean) / std
        else:
            score = df[col] - mean
        df[f"{col}_score"] = np.abs(score.fillna(0))
    return df[[c for c in df.columns if c.endswith("_score")] + ["Target"]]

def optimize_lgbm(X, y, n_trials=30):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }
        model = lgb.LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# =============================
# Pipeline Principal
# =============================

def feature_selection_screen():
    st.subheader("📂 Upload dos Arquivos de Entrada")
    x_file = st.file_uploader("🔢 Variáveis (X.csv)", type=["csv"])
    y_file = st.file_uploader("🎯 Target (y.csv)", type=["csv"])

    if not x_file or not y_file:
        st.stop()

    with st.spinner("📥 Carregando arquivos..."):
        X = load_large_csv(x_file, nrows=5000)
        y = pd.read_csv(y_file, index_col=0)

    y = y.iloc[:len(X)]  # garante alinhamento
    df = X.copy()
    df["Target"] = y.values.ravel()

    # Limpeza de dados
    df.dropna(axis=1, inplace=True)  # remove colunas com NaN
    df = df.loc[:, df.nunique() > 1]  # remove colunas constantes

    # Criação de scores
    df_scores = create_score_features(df)

    # Amostragem estratificada
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=42)
    for sample_idx, _ in splitter.split(df_scores, df_scores["Target"]):
        df_sample = df_scores.iloc[sample_idx]

    X_sample = df_sample.drop(columns=["Target"])
    y_sample = df_sample["Target"]

    st.success("✅ Dados processados com sucesso!")

    # Otimização LGBM
    st.subheader("🚀 Otimizando LightGBM com Optuna...")
    with st.spinner("Executando busca de hiperparâmetros..."):
        best_params = optimize_lgbm(X_sample, y_sample)
    st.success("✅ Otimização concluída")
    st.json(best_params)

    # Treinamento LGBM
    model_lgb = LGBMClassifier(**best_params)
    model_lgb.fit(X_sample, y_sample)

    feature_importance = pd.Series(model_lgb.feature_importances_, index=X_sample.columns)
    top_lgb_features = feature_importance.sort_values(ascending=False).head(150)

    # Treina rede neural
    st.subheader("🧠 Treinando Rede Neural para rankeamento final...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample[top_lgb_features.index])
    nn = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    nn.fit(X_scaled, y_sample)

    nn_weights = pd.Series(np.abs(nn.coefs_[0]).sum(axis=1), index=top_lgb_features.index)
    top_final = nn_weights.sort_values(ascending=False
