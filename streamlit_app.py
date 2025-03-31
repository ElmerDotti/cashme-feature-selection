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
    st.markdown("Este aplicativo permite a **seleção automática de variáveis** com base em técnicas de ML/AI.")
    st.markdown("O pipeline utiliza engenharia de atributos, amostragem estratificada, LightGBM e SHAP.")

# Executa a tela principal
feature_selection_screen()
