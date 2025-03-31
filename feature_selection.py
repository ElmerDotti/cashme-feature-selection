import pandas as pd
import numpy as np
import shap
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica colunas categóricas como números para uso em modelos."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def optimize_lgbm(X: pd.DataFrame, y: pd.Series, n_trials: int = 20):
    """Otimiza os hiperparâmetros de um modelo LightGBM usando Optuna."""
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }

        model = lgb.LGBMClassifier(**params)
        score = cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def select_features_with_lgbm(df: pd.DataFrame, output_dir: str = "outputs") -> pd.DataFrame:
    """Seleciona variáveis com LightGBM otimizado e gera SHAP values."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = encode_categoricals(df)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Otimização dos parâmetros
    best_params = optimize_lgbm(X, y, n_trials=30)
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)

    # Seleção de variáveis
    selector = SelectFromModel(model, threshold="mean", prefit=True)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    df_selected = X[selected_features].copy()
    df_selected["Target"] = y.values

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_values.png")
    plt.close()

    # Exportação
    df_selected.to_csv(f"{output_dir}/selected_features.csv")
    return df_selected, selected_features.tolist()
