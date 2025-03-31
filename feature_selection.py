import pandas as pd
import numpy as np
import shap
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from pathlib import Path
from scipy.stats import entropy

# ======================================
# UTILS
# ======================================
def load_data():
    X = pd.read_csv("X.csv", index_col=0)
    y = pd.read_csv("y.csv", index_col=0)
    df = X.copy()
    df["Target"] = y["Target"]
    return df


def encode_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes(include=["object", "bool", "category"]).columns:
        try:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
        except:
            df[col] = 0
    return df


def generate_feature_engineering(df):
    df = df.copy()
    for col in df.columns:
        if col == "Target":
            continue

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            df[f"{col}_lag"] = series.shift(1).fillna(0)
            df[f"{col}_lag_diff"] = df[col] - df[f"{col}_lag"]
            df[f"{col}_normalized"] = (series - series.mean()) / (series.std() + 1e-6)

            # Entropia (para colunas discretas apenas)
            if series.nunique() < 100:
                counts = series.value_counts()
                df[f"{col}_entropy"] = entropy(counts, base=2)
            else:
                df[f"{col}_entropy"] = 0
    df = df.fillna(0)
    return df


def optimize_lgbm(X, y, n_trials=20):
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


# ======================================
# FEATURE SELECTION SCREEN
# ======================================
def feature_selection_screen():
    st.title("🏦 Desafio CashMe - Feature Selection")

    df = load_data()
    st.subheader("📊 Amostra dos Dados")
    st.dataframe(df.sample(5))

    if st.button("Selecionar Features"):
        with st.spinner("🔍 Executando pré-processamento..."):
            df = encode_categoricals(df)
            df = generate_feature_engineering(df)

        X = df.drop(columns=["Target"])
        y = df["Target"]

        with st.spinner("⚙️ Otimizando LightGBM..."):
            best_params = optimize_lgbm(X, y, n_trials=20)
            model = lgb.LGBMClassifier(**best_params)
            model.fit(X, y)

        with st.spinner("📈 Analisando SHAP Values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)[1]  # Para classe 1
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.tight_layout()
            Path("outputs").mkdir(exist_ok=True)
            plt.savefig("outputs/shap_summary.png")
            plt.close()

        st.image("outputs/shap_summary.png", caption="Importância das Features (SHAP)")

        with st.spinner("📦 Selecionando Features mais Relevantes..."):
            selector = SelectFromModel(model, threshold="mean", prefit=True)
            selected_features = X.columns[selector.get_support()]
            df_selected = df[selected_features.tolist() + ["Target"]]
            df_selected.to_csv("outputs/selected_features.csv", index=False)

            st.success(f"✅ {len(selected_features)} features selecionadas.")
            st.dataframe(selected_features)

        with st.spinner("📉 PCA - Redução de Dimensionalidade"):
            X_scaled = StandardScaler().fit_transform(X.select_dtypes(include=[np.number]))
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
            legend1 = ax.legend(*scatter.legend_elements(), title="Target")
            ax.add_artist(legend1)
            st.pyplot(fig)

        st.success("🎉 Processo concluído com sucesso!")
