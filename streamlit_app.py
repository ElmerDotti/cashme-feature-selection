import streamlit as st
from login import login_screen
from feature_selection import feature_selection_screen

def main():
    st.set_page_config(page_title="CashMe - Feature Selection", layout="wide")
    
    if login_screen():
        st.sidebar.title("📊 Navegação")
        page = st.sidebar.radio("Ir para:", ["🏠 Início", "⚙️ Seleção de Variáveis"])

        if page == "🏠 Início":
            show_home()
        elif page == "⚙️ Seleção de Variáveis":
            feature_selection_screen()

def show_home():
    st.title("🏠 Bem-vindo ao Sistema de Seleção de Variáveis - CashMe")
    st.markdown("""
        Esta aplicação permite realizar seleção de variáveis utilizando técnicas modernas como:
        - **LightGBM com Optuna**
        - **SHAP Values**
        - **LazyPredict para modelos base**
        
        Acesse o menu lateral para iniciar o processo de análise.
    """)

if __name__ == "__main__":
    main()
