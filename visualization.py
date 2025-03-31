
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_distributions(df, output_dir, num_features=10):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['Target'])
    top_corrs = df[numeric_cols].corrwith(df['Target']).abs().sort_values(ascending=False).head(num_features)
    for col in top_corrs.index:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribuição: {col}')
        plt.savefig(f"{output_dir}/dist_{col}.png")
        plt.close()

def plot_boxplots(df, output_dir, num_features=10):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['Target'])
    top_corrs = df[numeric_cols].corrwith(df['Target']).abs().sort_values(ascending=False).head(num_features)
    for col in top_corrs.index:
        plt.figure()
        sns.boxplot(x="Target", y=col, data=df)
        plt.title(f'Boxplot: {col} x Target')
        plt.savefig(f"{output_dir}/boxplot_{col}.png")
        plt.close()

def plot_correlation_matrix(df, output_dir, num_features=10):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['Target'])
    top_corrs = df[numeric_cols].corrwith(df['Target']).abs().sort_values(ascending=False).head(num_features).index
    corr_matrix = df[top_corrs].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlação - Top 10 Features")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
