
import pandas as pd

def load_data(x_path, y_path):
    x_df = pd.read_csv(x_path, index_col=0)
    y_df = pd.read_csv(y_path, index_col=0)
    return x_df.join(y_df, how="inner")
