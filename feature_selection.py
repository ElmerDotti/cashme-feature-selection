import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import streamlit as st

from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        try:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
        except Exception as e:
            st.warning(f"Erro ao codificar '{col}': {e}")
            df[col] = 0
    return df


def create_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "Target":
            continue
        try:
            mean = df[col].mean()
            std = df[col].std() + 1e-6
            score = (df[col] - mean) / std
            score = np.abs(score)  # garantir valores positivos
            df[f"{col}_score"] = score
        except Exception as e:
            st.warning(f"Erro ao calcular score de '{col}': {e}")
    return df


def optimize_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 20):
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
        score = cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def load_data() -> pd.DataFrame:
    st.subheader("📂 Upload dos Arquivos de Entrada")

    x_file = st.file_uploader("Selecione o arquivo de variáveis (X.csv)", type=["csv"])
    y_file = st.file_uploader("Selecione o arquivo de target (y.csv)", type=["csv"])

    if not x_file or not y_file:
        st.stop()

    X = pd.read_csv(x_file, index_col=0)
    y = pd.read_csv(y_file, index_col=0)

    if X.shape[0] != y.shape[0]:
        st.error("Os arquivos X e y têm números de linhas diferentes!")
        st.stop()

    df = X.copy()
    df["Target"] = y.values.ravel()
    return df


def feature_selection_screen():
    df = load_data()
    if df.empty:
        st.warning("O dataset está vazio ou não foi carregado corretamente.")
        return

    df = encode_categoricals(df)
    df = create_score_features(df)

    # Apenas os scores e target
    score_cols = [col for col in df.columns if col.endswith("_score")]
    score_df = df[score_cols + ["Target"]].copy()

    # Remove colunas com valor zero ou NaN
    score_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    score_df.interpolate(axis=0, inplace=True)
    score_df.dropna(axis=1, inplace=True)
    score_df = score_df.loc[:, (score_df != 0).any(axis=0)]

    # Amostragem estratificada
    strat_sample = score_df.groupby("Target", group_keys=False).apply(lambda x: x.sample(min(len(x), 25), random_state=42))
    X_sampled = strat_sample.drop(columns=["Target"])
    y_sampled = strat_sample["Target"]

    st.subheader("🔍 Dataset de Entrada (Scores)")
    st.dataframe(X_sampled.head())

    with st.spinner("⚙️ Otimizando e treinando modelo..."):
        best_params = optimize_lgbm(X_sampled, y_sampled, n_trials=30)

    st.success("✅ Otimização concluída!")
    st.json(best_params)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_sampled, y_sampled)

    # Seleciona top 50 variáveis com maior importância
    feature_importance = pd.Series(model.feature_importances_, index=X_sampled.columns)
    top_50_features = feature_importance.nlargest(50).index.tolist()

    df_selected = X_sampled[top_50_features].copy()

    st.subheader("📌 Variáveis Selecionadas")
    st.write(f"Total de variáveis selecionadas: **{len(top_50_features)}**")
    st.dataframe(df_selected.head())

    # Gráfico de importância (histograma)
    st.subheader("📊 Importância Relativa das Variáveis Selecionadas")
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance[top_50_features].sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Importância das Variáveis Selecionadas (LightGBM)")
    st.pyplot(fig)

    # CSV para download
    st.subheader("📥 Baixar Variáveis Selecionadas")
    csv = pd.DataFrame(top_50_features, columns=["selected_features"])
    st.download_button("📄 Baixar Lista de Variáveis (CSV)", data=csv.to_csv(index=False).encode("utf-8"),
                       file_name="top_50_selected_features.csv")
