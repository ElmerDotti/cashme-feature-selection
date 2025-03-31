import pandas as pd
import numpy as np
import shap
import optuna
import lightgbm as lgb
import streamlit as st
import matplotlib.pyplot as plt

from io import BytesIO
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score

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

def entropy(col):
    p_data = col.value_counts(normalize=True)
    return -np.sum(p_data * np.log2(p_data + 1e-9))

def create_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score_features = pd.DataFrame(index=df.index)

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "Target":
            continue
        try:
            ratios = [df[col] / df[col].shift(i) for i in range(1, 6)]
            ratio_mean = pd.concat(ratios, axis=1).mean(axis=1).fillna(0)
            col_entropy = entropy(df[col])
            col_score = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
            score_features[f"{col}_score"] = col_score * col_entropy * ratio_mean
        except Exception as e:
            st.warning(f"Erro ao criar score para '{col}': {e}")
            score_features[f"{col}_score"] = 0

    return score_features

def stratified_sample(df: pd.DataFrame, target_col: str = "Target", sample_size: int = 50) -> pd.DataFrame:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)
    for _, test_idx in splitter.split(df, df[target_col]):
        return df.iloc[test_idx].reset_index(drop=True)
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

# ========== Pipeline principal ==========

def feature_selection_screen():
    df = load_data()

    if df.empty:
        st.warning("O dataset está vazio ou não foi carregado corretamente.")
        return

    df = encode_categoricals(df)

    # Engenharia de Atributos
    df_scores = create_score_features(df)
    df_scores["Target"] = df["Target"].values

    # Amostragem Estratificada
    df_sampled = stratified_sample(df_scores, target_col="Target", sample_size=50)

    st.subheader("🔍 Dataset de Entrada (Scores)")
    st.dataframe(df_sampled.drop(columns=["Target"]).head())

    X = df_sampled.drop(columns=["Target"])
    y = df_sampled["Target"]

    # Otimização
    st.subheader("⚙️ Otimizando modelo com Optuna...")
    with st.spinner("Otimizando..."):
        best_params = optimize_lgbm(X, y, n_trials=20)
    st.success("✅ Otimização concluída!")
    st.json(best_params)

    # Modelo final com parâmetros otimizados
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)

    # Seleção de Variáveis com Modelo Otimizado
    st.subheader("📊 Seleção de Variáveis com LightGBM")
    selector = SelectFromModel(model, threshold="mean", prefit=True)
    selected_features = X.columns[selector.get_support()]
    st.write(f"✅ {len(selected_features)} variáveis selecionadas:")
    st.code(list(selected_features))

    df_selected = X[selected_features].copy()
    df_selected["Target"] = y.values

    # SHAP Value
    st.subheader("🌟 SHAP - Interpretação do Modelo")
    try:
        explainer = shap.Explainer(model, X[selected_features])
        shap_values = explainer(X[selected_features])

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X[selected_features], plot_type="bar", show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Erro ao gerar gráfico SHAP: {e}")

    # CSV Download
    st.subheader("📥 Baixar Resultado")
    csv_buffer = BytesIO()
    df_selected.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📄 Baixar CSV das Variáveis Selecionadas",
        data=csv_buffer.getvalue(),
        file_name="selected_features.csv",
        mime="text/csv"
    )

    # Libera memória
    del df, df_scores, df_sampled, X, y, model, shap_values, explainer
    import gc
    gc.collect()
