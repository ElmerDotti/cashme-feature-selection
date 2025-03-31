import streamlit as st

# Usuários e senhas armazenados em dicionário simples (apenas para fins de demonstração)
USERS = {
    "admin": "1234",
    "elmer": "cashme",
}

def login_screen():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return True

    st.title("🔐 Login")

    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        if USERS.get(username) == password:
            st.session_state.logged_in = True
            st.success("Login realizado com sucesso!")
            return True
        else:
            st.error("Usuário ou senha incorretos.")

    return False
