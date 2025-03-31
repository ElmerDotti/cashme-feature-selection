import pandas as pd
import numpy as np
import shap
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# ========== Funções Auxiliares ==========

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

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def entropy(col):
        p_data = col.value_counts(normalize=True)
        return -np.sum(p_data * np.log2(p_data + 1e-9))

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "Target":
            continue

        try:
            for i in range(1, 6):
                df[f"{col}_r{i}"] = df[col] / (df[col].shift(i) + 1e-6)
            ratio_cols = [f"{col}_r{i}" for i in range(1, 6)]
            df[f"{col}_entropy"] = entropy(df[col])
            score = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
            df[f"{col}_score"] = score * df[f"{col}_entropy"]
        except Exception as e:
            st.warning(f"Erro ao derivar '{col}': {e}")
    
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

# ========== Pipeline Principal ==========

def feature_selection_screen():
    df = load_data()

    if df.empty:
        st.warning("O dataset está vazio ou não foi carregado corretamente.")
        return

    df = encode_categoricals(df)
    df = create_derived_features(df)

    st.subheader("🔍 Preview do Dataset")
    st.dataframe(df.head())

    # Somente scores como entrada
    score_cols = [col for col in df.columns if col.endswith("_score")]
    df_score_only = df[score_cols + ["Target"]]

    # Amostragem estratificada de 100 amostras
    df_sampled = df_score_only.groupby("Target", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 50), random_state=42)
    )

    X_sampled = df_sampled.drop(columns=["Target"])
    y_sampled = df_sampled["Target"]

    # Otimização e treino
    st.subheader("⚙️ Otimizando parâmetros do LightGBM com Optuna...")
    with st.spinner("Otimizando parâmetros..."):
        best_params = optimize_lgbm(X_sampled, y_sampled, n_trials=30)

    st.success("✅ Otimização concluída!")
    st.json(best_params)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_sampled, y_sampled)

    # Seleção de variáveis
    st.subheader("📊 Seleção de Variáveis com LightGBM")
    selector = SelectFromModel(model, threshold="mean", prefit=True)
    selected_mask = selector.get_support()
    selected_features = X_sampled.columns[selected_mask]
    st.write("Variáveis selecionadas:", list(selected_features))

    df_selected = X_sampled[selected_features].copy()
    df_selected["Target"] = y_sampled.values

    # SHAP
    st.subheader("🌟 SHAP - Interpretação do Modelo")
    try:
        explainer = shap.Explainer(model, df_selected.drop(columns=["Target"]))
        shap_values = explainer(df_selected.drop(columns=["Target"]))

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, df_selected.drop(columns=["Target"]), plot_type="bar", show=False)
        st.pyplot()
    except Exception as e:
        st.warning(f"Erro ao gerar SHAP: {e}")

    # Download do CSV
    st.subheader("📥 Baixar resultado")
    try:
        csv = df_selected.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Baixar CSV com Features Selecionadas", data=csv, file_name="selected_features.csv")
    except Exception as e:
        st.error(f"Erro ao gerar arquivo CSV: {e}")
