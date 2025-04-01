import streamlit as st

# Função principal da tela de login
def login_screen():
    st.title("🔐 Desafio CashMe - Autenticação")

    with st.form("login_form"):
        username = st.text_input("Usuário", key="username")
        password = st.text_input("Senha", type="password", key="password")

        login_button = st.form_submit_button("Entrar")

    if login_button:
        if username == "cashme123" and password == "cashme123":
            st.success("Login realizado com sucesso!")
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Credenciais inválidas. Tente novamente.")
