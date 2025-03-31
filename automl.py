import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

def run_automl_comparison(df: pd.DataFrame):
    """
    Executa comparação entre múltiplos modelos usando LazyPredict.
    Retorna um DataFrame com as métricas de performance.
    """
    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, _ = clf.fit(X_train, X_test, y_train, y_test)

    models = models.sort_values(by="ROC AUC", ascending=False)
    models.to_csv("outputs/model_comparison.csv")

    return models
