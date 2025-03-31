
def describe_features(df):
    descriptions = {}
    for col in df.columns:
        if col.endswith("_lag1"):
            descriptions[col] = "Lag de 1 período da variável original"
        elif col.endswith("_var"):
            descriptions[col] = "Razão de variação normalizada entre valor atual e o lag"
        elif col.endswith("_ent"):
            descriptions[col] = "Entropia da distribuição da variável original"
        elif col.endswith("_score"):
            descriptions[col] = "Score normalizado da variável original (entre 0 e 1)"
        elif col.startswith("PCA_"):
            descriptions[col] = "Componente principal extraída via PCA"
        elif col == "Cluster":
            descriptions[col] = "Segmento atribuído pelo modelo de clusterização"
        else:
            descriptions[col] = "Variável original ou derivada numérica"
    return descriptions
