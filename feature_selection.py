import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from io import BytesIO


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        try:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
        except:
            df[col] = 0
    return df


def create_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score_df = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "Target":
            continue
        try:
            series = df[col].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
            if (series > 0).all():
                ratios = pd.concat(
                    [(series / series.shift(i)).rename(f"{col}_ratio_{i}") for i in range(1, 6)],
                    axis=1
                )
                score_df[f"{col}_entropy"] = -series.value_counts(normalize=True).mul(
                    np.log2(series.value_counts(normalize=True) + 1e-9)
                ).sum()
                score_df[f"{col}_score"] = (series - series.mean()) / (series.std() + 1e-6)
        except:
            continue
    score_df = score_df.filter(like="_score", axis=1)
    score_df = score_df.replace(0, np.nan).dropna(axis=1)
    return score_df


def optimize_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 20):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 80),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }
        model = lgb.LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def load_data() -> pd.DataFrame:
    x_file = st.file_uploader("📂 Upload X.csv (variáveis)", type=["csv"])
    y_file = st.file_uploader("📂 Upload y.csv (target)", type=["csv"])
    if not x_file or not y_file:
        st.stop()

    X = pd.read_csv(x_file, index_col=0)
    y = pd.read_csv(y_file, index_col=0)

    if X.shape[0] != y.shape[0]:
        st.error("X e y têm tamanhos diferentes.")
        st.stop()

    df = X.copy()
    df["Target"] = y.values.ravel()
    df = encode_categoricals(df)
    return df


def feature_selection_screen():
    df = load_data()
    df_scores = create_scores(df)
    df_scores["Target"] = df["Target"].values

    # 🔍 Preview
    st.subheader("🔍 Dataset de Entrada (Scores)")
    st.dataframe(df_scores.drop(columns="Target").head())

    # Stratified Sampling (pré-configurado: 100 amostras)
    df_sampled, _ = train_test_split(
        df_scores, stratify=df_scores["Target"], train_size=100, random_state=42
    )

    X = df_sampled.drop(columns=["Target"])
    y = df_sampled["Target"]

    # ⏱ Otimização com Optuna
    st.subheader("⚙️ Otimizando LightGBM com Optuna...")
    with st.spinner("Executando otimização..."):
        best_params = optimize_lgbm(X, y, n_trials=30)
    st.success("✅ Parâmetros Otimizados:")
    st.json(best_params)

    # 🔎 Seleção de Variáveis com LightGBM
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(50).index.tolist()
    X_selected = X[top_features]

    st.subheader("📈 Quantidade de Variáveis Selecionadas")
    st.metric("Variáveis Selecionadas", len(top_features))

    # 📊 SHAP Summary Plot
    st.subheader("🌟 SHAP - Interpretação do Modelo")
    try:
        explainer = shap.Explainer(model, X_selected)
        shap_values = explainer(X_selected)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_selected, plot_type="bar", show=False)
        st.pyplot()
    except Exception as e:
        st.warning(f"Erro ao gerar gráfico SHAP: {e}")

    # 📁 Download do Resultado
    st.subheader("📥 Baixar Variáveis Selecionadas")
    selected_df = pd.DataFrame(top_features, columns=["Variável"])
    csv_bytes = selected_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", csv_bytes, file_name="top_50_variaveis.csv")

    # Critério de Seleção
    with st.expander("🧠 Critério de Seleção das Variáveis"):
        st.markdown(
            """
            As variáveis foram selecionadas com base na **importância média** atribuída pelo modelo LightGBM,
            após otimização de hiperparâmetros com **Optuna**. Foram escolhidas as **50 features com maior capacidade de discriminação**
            do comportamento-alvo (`Target`).
            """
        )
