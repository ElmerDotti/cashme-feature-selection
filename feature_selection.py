
import pandas as pd
import numpy as np
import streamlit as st
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def normalize_scores(df_scores):
    df_scores = df_scores.abs()
    return (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-6)

def load_data():
    st.subheader("📂 Upload dos Arquivos de Entrada")
    x_file = st.file_uploader("Selecione o arquivo de variáveis (X.csv)", type=["csv"])
    y_file = st.file_uploader("Selecione o arquivo de target (y.csv)", type=["csv"])
    if not x_file or not y_file:
        st.stop()
    X = pd.read_csv(x_file, index_col=0)
    y = pd.read_csv(y_file, index_col=0).values.ravel()
    return X, y

def feature_selection_screen():
    X, y = load_data()

    # Normalizar e remover colunas nulas ou zero
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    X = X.loc[:, (X != 0).any(axis=0)]
    X_scores = normalize_scores(X.add_suffix("_score"))

    # Amostragem estratificada com 100 registros
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=42)
    for train_idx, _ in splitter.split(X_scores, y):
        X_sampled = X_scores.iloc[train_idx]
        y_sampled = y[train_idx]

    st.subheader("🔍 Dataset de Entrada (Scores)")
    st.dataframe(X_sampled.head())

    # LightGBM + Optuna
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
        return model.fit(X_sampled, y_sampled).score(X_sampled, y_sampled)

    st.subheader("⚙️ Otimizando LightGBM com Optuna...")
    with st.spinner("Otimizando parâmetros..."):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

    best_params = study.best_params
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_sampled, y_sampled)
    importances = pd.Series(model.feature_importances_, index=X_sampled.columns)
    top_features = importances.sort_values(ascending=False).head(150)

    # Rede Neural
    clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
    clf.fit(X_sampled[top_features.index], y_sampled)
    nn_weights = pd.Series(np.abs(clf.coefs_[0]).sum(axis=1), index=top_features.index)

    # Seleção final
    top_100_features = nn_weights.sort_values(ascending=False).head(100)

    st.subheader("✅ Variáveis Selecionadas (Top 100)")
    st.write(list(top_100_features.index))
    st.download_button("📥 Baixar Lista de Variáveis", data="\n".join(top_100_features.index), file_name="variaveis_selecionadas.txt")

    # Histograma com pesos da rede neural
    st.subheader("📊 Importância das Variáveis via Rede Neural")
    fig, ax = plt.subplots(figsize=(12, 6))
    top_100_features.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Pesos das Variáveis Selecionadas")
    ax.set_xlabel("Peso da Rede Neural")
    st.pyplot(fig)
