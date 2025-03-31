import pandas as pd
import numpy as np
import shap
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
import base64
import io


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(axis=1, thresh=len(df) * 0.5, inplace=True)  # Remove colunas com mais de 50% nulos
    df.fillna(-999, inplace=True)  # Fallback para demais nulos
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        try:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
        except Exception as e:
            st.warning(f"Erro ao codificar {col}: {e}")
            df[col] = 0
    return df


def show_eda_visuals(df: pd.DataFrame):
    st.subheader("🔍 Matriz de Correlação")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Distribuição da Variável Target")
    if "Target" in df.columns:
        st.bar_chart(df["Target"].value_counts())


def optimize_lgbm(X, y, n_trials=25):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
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
    st.title("📊 Seleção de Variáveis com ML + Otimização")

    uploaded_file = st.file_uploader("📁 Faça upload do CSV contendo a coluna 'Target'")
    if uploaded_file is None:
        st.info("Envie um arquivo CSV para continuar.")
        return

    df = pd.read_csv(uploaded_file)
    if "Target" not in df.columns:
        st.error("Coluna 'Target' obrigatória no dataset.")
        return

    show_eda_visuals(df)

    st.markdown("---")
    st.subheader("🔧 Pré-processamento e Codificação")

    df_clean = preprocess_data(df)
    df_encoded = encode_categoricals(df_clean)

    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    st.info(f"{X_train.shape[1]} variáveis consideradas para otimização...")

    with st.spinner("🧠 Otimizando hiperparâmetros..."):
        best_params = optimize_lgbm(X_train, y_train)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    selector = SelectFromModel(model, threshold="mean", prefit=True)
    selected_features = X.columns[selector.get_support()]
    df_selected = df_encoded[selected_features].copy()
    df_selected["Target"] = y.values

    st.success(f"{len(selected_features)} variáveis selecionadas!")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("📊 SHAP - Importância das Variáveis")
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("📉 PCA - Redução de Dimensionalidade")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    fig2, ax2 = plt.subplots()
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    ax2.set_title("PCA - 2 Componentes")
    st.pyplot(fig2)

    st.subheader("🧬 Segmentação via K-Means")
    clusters = KMeans(n_clusters=2, random_state=42).fit_predict(X)
    fig3, ax3 = plt.subplots()
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='cool')
    ax3.set_title("KMeans Clustering em 2D PCA")
    st.pyplot(fig3)

    csv = df_selected.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="selected_features.csv">📥 Baixar CSV com Features Selecionadas</a>'
    st.markdown(href, unsafe_allow_html=True)
