import streamlit as st
import traceback

# Título fixo da página
st.set_page_config(page_title="CashMe Feature Selection", layout="wide")

try:
    # Importações dos módulos do projeto
    from login import login_screen
    from app_main import run_app

    # Executa a tela de login
    if login_screen():
        run_app()

except Exception as e:
    # Tratamento de erro com logging na interface
    st.error("Erro durante a execução da aplicação.")
    st.exception(e)
    st.text(traceback.format_exc())
