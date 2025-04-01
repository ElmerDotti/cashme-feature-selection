
import streamlit as st

def login():
    st.title("🏦 Desafio CashMe - Login")
    with st.form("login_form"):
        user = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submit = st.form_submit_button("Entrar")

    if submit:
        if user == "cashme123" and password == "cashme123":
            st.success("Login realizado com sucesso!")
            st.session_state.logged_in = True
        else:
            st.error("Usuário ou senha incorretos.")
