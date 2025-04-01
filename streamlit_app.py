import streamlit as st
from feature_selection import feature_selection_screen
from login import login_screen

st.set_page_config(
    page_title="🏦 CashMe - Feature Selection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Executa a tela de login primeiro
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_screen()
else:
    st.title("🏦 Desafio CashMe - Seleção de Variáveis com Machine Learning")
    with st.sidebar:
        st.header("⚙️ Configurações")
        st.markdown("Este aplicativo permite a **seleção automática de variáveis** com base em técnicas de ML/AI.")
        st.markdown("O pipeline utiliza engenharia de atributos, amostragem estratificada, LightGBM e rede neural.")

    feature_selection_screen()
