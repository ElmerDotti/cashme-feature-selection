import streamlit as st
from login import login_screen
from feature_selection import feature_selection_screen

st.set_page_config(
    page_title="🏦 CashMe - Feature Selection",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Login Screen
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_screen()
    else:
        st.title("🏦 Desafio CashMe - Seleção de Variáveis com IA")
        with st.sidebar:
            st.header("⚙️ Configurações")
            st.markdown("Este app realiza **seleção automática de variáveis** via LightGBM + Rede Neural.")
        feature_selection_screen()

if __name__ == "__main__":
    main()
