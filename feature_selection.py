import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import streamlit as st

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        try:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
        except Exception:
            df[col] = 0
    return df


def entropy(col):
    p_data = col.value_counts() / len(col)
    return -sum(p_data * np.log2(p_data + 1e-9))


def create_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores = pd.DataFrame(index=df.index)

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "Target":
            continue
        try:
            for lag in range(1, 6):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
            df = df.fillna(method="bfill")

            # Razões positivas
            for lag in range(1, 6):
                ratio = df[col] / (df[f"{col}_lag{lag}"] + 1e-6)
                ratio = ratio.abs()  # garantir positividade
                df[f"{col}_ratio{lag}"] = ratio

            # Score baseado em entropia e razões
            components = [df[f"{col}_ratio{lag}"] for lag in range(1, 6)]
            components.append(entropy(df[col]))
            score = sum(components) / (len(components) + 1e-6)
            scores[f"{col}_score"] = score.abs()
        except Exception:
            continue

    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0)
    return scores


def optimize_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 25):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 80),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
        model = lgb.LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def feature_selection_screen():
    st.subheader("📂 Upload dos Arquivos de Entrada")
    x_file = st.file_uploader("Variáveis (X.csv)", type=["csv"])
    y_file = st.file_uploader("Target (y.csv)", type=["csv"])

    if not x_file or not y_file:
        st.stop()

    X_raw = pd.read_csv(x_file, index_col=0)
    y = pd.read_csv(y_file, index_col=0).values.ravel()

    df = encode_categoricals(X_raw)
    df["Target"] = y
    scores_df = create_score_features(df)
    scores_df["Target"] = y

    # Remover colunas vazias ou constantes
    X = scores_df.drop(columns=["Target"])
    X = X.loc[:, (X != 0).any(axis=0)]
    X = X.loc[:, X.nunique() > 1]
    y = scores_df["Target"]

    # Amostragem estratificada
    X_sample, _, y_sample, _ = train_test_split(X, y, stratify=y, test_size=0.95, random_state=42)

    st.subheader("⚙️ Otimizando parâmetros com LightGBM + Optuna")
    with st.spinner("Otimizando..."):
        best_params = optimize_lgbm(X_sample, y_sample, n_trials=20)
    st.success("✅ Otimização concluída!")
    st.json(best_params)

    model_lgbm = lgb.LGBMClassifier(**best_params)
    model_lgbm.fit(X_sample, y_sample)

    # Importância LGBM
    importance = pd.Series(model_lgbm.feature_importances_, index=X_sample.columns)
    top_lgbm_features = importance.sort_values(ascending=False).head(200).index

    # Rede Neural para calcular pesos
    model_nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    model_nn.fit(X_sample[top_lgbm_features], y_sample)

    nn_weights = np.abs(model_nn.coefs_[0]).sum(axis=1)
    weight_df = pd.Series(nn_weights, index=top_lgbm_features).sort_values(ascending=False)

    selected_features = weight_df.head(100).index.tolist()

    st.subheader("📌 Variáveis Selecionadas")
    st.write(f"Total: {len(selected_features)} variáveis")
    for feat in selected_features:
        st.markdown(f"- {feat}")

    # Salvar resultado
    Path(".outputs").mkdir(exist_ok=True)
    selected_path = Path(".outputs/selected_variables.csv")
    pd.Series(selected_features).to_csv(selected_path, index=False)

    st.download_button(
        label="📄 Baixar Nomes das Variáveis Selecionadas",
        data=selected_path.read_bytes(),
        file_name="selected_variables.csv"
    )

    # Histograma dos pesos da rede neural
    st.subheader("📈 Histograma dos Pesos da Rede Neural")
    fig, ax = plt.subplots(figsize=(12, 6))
    weight_df.head(100).plot(kind="bar", ax=ax)
    ax.set_title("Importância Relativa - Pesos Rede Neural (Top 100)")
    ax.set_ylabel("Peso Absoluto")
    ax.set_xlabel("Variável")
    plt.xticks(rotation=90)
    plt.tight_layout()
    hist_path = Path(".outputs/neural_network_weights_histogram.png")
    plt.savefig(hist_path)
    st.pyplot(fig)
