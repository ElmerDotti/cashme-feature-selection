import streamlit as st
from login import login_screen
from feature_selection import feature_selection_screen

st.set_page_config(
    page_title="🏦 CashMe - Feature Selection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializa autenticação
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Mostra tela de login se necessário
if not st.session_state.authenticated:
    login_screen()
else:
    st.title("🏦 Desafio CashMe - Seleção de Variáveis com Machine Learning")
    with st.sidebar:
        st.header("⚙️ Configurações")
        st.markdown("Este app realiza **seleção automática de variáveis** com ML, LGBM, rede neural e Optuna.")
    feature_selection_screen()
