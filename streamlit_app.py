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
    st.markdown("""
        Este aplicativo permite a **seleção automática de variáveis** com base em técnicas avançadas de ML.

        🔍 Pipeline:
        - Criação de variáveis derivadas (scores)
        - Amostragem estratificada
        - Otimização com Optuna
        - Seleção com LightGBM
        - Interpretação com SHAP
        - Exportação de variáveis selecionadas
    """)

# Executa o pipeline principal
feature_selection_screen()
