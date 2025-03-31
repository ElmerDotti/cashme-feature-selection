
from data_loader import load_data
from preprocessing import preprocess_data
from feature_engineering import generate_lag_features
from visualization import plot_distributions, plot_boxplots, plot_correlation_matrix
from modeling import stratified_sampling, reduce_dimensionality, cluster_data
from feature_selection import select_features_with_lgbm
from utils import describe_features

def main():
    df = load_data("X.csv", "y.csv")
    df_clean = preprocess_data(df)
    df_features = generate_lag_features(df_clean)

    plot_distributions(df_features, output_dir="outputs")
    plot_boxplots(df_features, output_dir="outputs")
    plot_correlation_matrix(df_features, output_dir="outputs")

    df_sampled = stratified_sampling(df_features)
    df_reduced = reduce_dimensionality(df_sampled)
    df_segmented = cluster_data(df_reduced)

    df_final, selected_features = select_features_with_lgbm(df_segmented)

    descriptions = describe_features(df_final[selected_features])
    for name, desc in descriptions.items():
        print(f"{name}: {desc}")

if __name__ == "__main__":
    main()
