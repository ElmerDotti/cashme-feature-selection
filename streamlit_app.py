import streamlit as st
from feature_selection import feature_selection_screen

st.set_page_config(
    page_title="🏦 CashMe - Feature Selection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Desafio CashMe - Seleção de Variáveis com Machine Learning")

with st.sidebar:
    st.header("⚙️ Configurações")
    st.markdown("Este aplicativo realiza a **seleção automática das 50 variáveis mais relevantes** com base em técnicas de ML.")
    st.markdown("- Engenharia de atributos (Score)\n"
                "- Amostragem estratificada\n"
                "- Otimização com Optuna\n"
                "- Modelo LightGBM\n"
                "- Visualização de importância")

feature_selection_screen()
