import pandas as pd
import numpy as np
import shap
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import streamlit as st
import gc

from io import BytesIO
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# ========== Utilitários ==========

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

    def entropy(col):
        p_data = col.value_counts() / len(col)
        return -sum(p_data * np.log2(p_data + 1e-9))

    score_df = pd.DataFrame(index=df.index)

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "Target":
            continue
        try:
            ratios = []
            for i in range(1, 6):
                ratio = df[col] / (df[col].shift(i).replace(0, np.nan))
                ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
                ratio.fillna(1, inplace=True)
                ratios.append(ratio)

            ent = entropy(df[col])
            mean = df[col].mean()
            std = df[col].std() + 1e-6

            score = (df[col] - mean) / std + ent + sum(ratios)
            score_df[f"{col}_score"] = score
        except Exception as e:
            st.warning(f"Erro ao criar score de '{col}': {e}")

    return score_df


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
        return cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()

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


# ========== Pipeline principal ==========

def feature_selection_screen():
    df = load_data()

    if df.empty:
        st.warning("O dataset está vazio ou não foi carregado corretamente.")
        return

    df = encode_categoricals(df)

    # Amostragem estratificada de 50 amostras
    st.subheader("🔎 Amostragem Estratificada")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=50, random_state=42)
    for train_idx, _ in splitter.split(df.drop(columns=["Target"]), df["Target"]):
        df = df.iloc[train_idx]
    st.write(f"Amostra com {df.shape[0]} registros.")

    score_df = create_score_features(df)
    score_df["Target"] = df["Target"].values
    st.subheader("🔍 Preview do Dataset (Scores)")
    st.dataframe(score_df.head())

    X = score_df.drop(columns=["Target"])
    y = score_df["Target"]

    # Remove colunas com variância nula
    X = X.loc[:, X.nunique() > 1]

    st.subheader("⚙️ Otimizando parâmetros do LightGBM com Optuna...")
    with st.spinner("Otimizando parâmetros..."):
        best_params = optimize_lgbm(X, y, n_trials=30)
    st.success("✅ Otimização concluída!")
    st.json(best_params)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)

    st.subheader("📊 Seleção de Variáveis com LightGBM")
    selector = SelectFromModel(model, threshold="mean", prefit=True)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    st.write("Variáveis selecionadas:", list(selected_features))

    df_selected = X[selected_features].copy()
    df_selected["Target"] = y.values

    # SHAP Plot
    st.subheader("🌟 SHAP - Interpretação do Modelo")
    try:
        explainer = shap.Explainer(model, df_selected.drop(columns=["Target"]))
        shap_values = explainer(df_selected.drop(columns=["Target"]))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, df_selected.drop(columns=["Target"]), plot_type="bar", show=False)
        st.pyplot()
    except Exception as e:
        st.warning(f"Erro ao gerar SHAP: {e}")

    # Download dos resultados
    st.subheader("📥 Baixar resultado")
    csv = df_selected.to_csv(index=False).encode('utf-8')
    st.download_button("📄 Baixar CSV com Features Selecionadas", data=csv, file_name="selected_features.csv")

    # Liberar memória
    del df, score_df, X, y, model, df_selected
    gc.collect()
