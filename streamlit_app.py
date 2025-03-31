
import streamlit as st
import pandas as pd
from data_loader import load_data
from preprocessing import preprocess_data
from feature_engineering import generate_lag_features
from feature_selection import select_features_with_lgbm
from automl import run_automl_comparison

st.set_page_config(page_title="Feature Selection App", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if username == "Cashme123" and password == "Cashme123":
            st.session_state.authenticated = True
            st.success("Login bem-sucedido!")
        else:
            st.error("Credenciais inválidas.")
    st.stop()

st.title("🔍 CashMe - Feature Selection Interativo")

uploaded_x = st.file_uploader("Upload do X.csv", type="csv")
uploaded_y = st.file_uploader("Upload do y.csv", type="csv")

if uploaded_x and uploaded_y:
    progress = st.progress(0, text="Aguardando...")

    df = load_data(uploaded_x, uploaded_y)
    st.success("✅ Dados carregados")
    progress.progress(20, text="Dados carregados")

    if st.button("1️⃣ Processar dados"):
        df_clean = preprocess_data(df)
        df_features = generate_lag_features(df_clean)
        st.session_state["df_features"] = df_features
        st.success("✅ Dados processados!")
        progress.progress(50, text="Pré-processamento concluído")

    if st.button("2️⃣ Selecionar Features"):
        df_final, selected = select_features_with_lgbm(st.session_state["df_features"])
        st.session_state["df_final"] = df_final
        st.write("✅ Features selecionadas:", selected)
        st.download_button("📥 Baixar matriz final", df_final.to_csv().encode(), "selected_features.csv")
        progress.progress(80, text="Seleção de features finalizada")

    if st.button("3️⃣ Comparar Modelos (AutoML)"):
        models = run_automl_comparison(st.session_state["df_final"])
        st.dataframe(models)
        progress.progress(100, text="Comparação de modelos concluída")
