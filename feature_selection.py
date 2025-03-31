import pandas as pd
import shap
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variáveis categóricas usando LabelEncoder.
    Conversão segura: transforma em string e trata falhas silenciosamente.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        try:
            df[col] = df[col].astype(str)  # Garante string
            df[col] = LabelEncoder().fit_transform(df[col])
        except Exception as e:
            print(f"Erro ao codificar '{col}': {e}")
            df[col] = 0  # fallback para evitar falha
    return df


def optimize_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 20):
    """
    Usa Optuna para buscar os melhores hiperparâmetros para o LightGBM.
    """

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


def select_features_with_lgbm(df: pd.DataFrame, output_dir: str = "outputs") -> tuple[pd.DataFrame, list[str]]:
    """
    Realiza a seleção de features:
    - Codifica categóricas
    - Otimiza e treina LightGBM
    - Seleciona features importantes
    - Gera gráfico de SHAP
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = encode_categoricals(df)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Otimização de hiperparâmetros
    best_params = optimize_lgbm(X, y, n_trials=30)
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)

    # Seleção com base na importância média
    selector = SelectFromModel(model, threshold="mean", prefit=True)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    df_selected = X[selected_features].copy()
    df_selected["Target"] = y.values

    # SHAP Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_values.png")
    plt.close()

    # Salva o novo dataset com features selecionadas
    df_selected.to_csv(f"{output_dir}/selected_features.csv", index=False)

    return df_selected, selected_features.tolist()

import streamlit as st
import base64
import io

def feature_selection_screen():
    st.title("📊 Seleção de Variáveis com LightGBM + Optuna")

    uploaded_file = st.file_uploader("📁 Faça upload de um arquivo CSV com a coluna 'Target'", type="csv")
    if uploaded_file is None:
        st.info("Por favor, envie um arquivo para continuar.")
        return

    df = pd.read_csv(uploaded_file)
    if "Target" not in df.columns:
        st.error("O arquivo precisa conter uma coluna chamada 'Target'.")
        return

    st.success("Arquivo carregado com sucesso! Iniciando seleção de features...")

    with st.spinner("Executando seleção... isso pode levar um tempo..."):
        selected_df, selected_cols = select_features_with_lgbm(df)

    st.subheader("✅ Features Selecionadas:")
    st.write(selected_cols)

    st.subheader("📈 Gráfico de Importância SHAP:")
    shap_path = Path("outputs/shap_values.png")
    if shap_path.exists():
        st.image(str(shap_path))
    else:
        st.warning("Gráfico SHAP não encontrado.")

    st.subheader("📥 Download dos Dados com Features Selecionadas:")
    csv = selected_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="selected_features.csv">Clique para baixar</a>'
    st.markdown(href, unsafe_allow_html=True)

