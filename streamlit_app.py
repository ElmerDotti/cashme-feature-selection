import streamlit as st
from feature_selection import feature_selection_screen

# Configuração da página
st.set_page_config(
    page_title="🏦 CashMe - Feature Selection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== Tela de Login ======
def login_screen():
    st.title("🏦 Desafio CashMe - Feature Selection")
    st.markdown("### 🔐 Login obrigatório para acesso ao aplicativo")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submit = st.form_submit_button("Entrar")

        if submit:
            if username == "cashme123" and password == "cashme123":
                st.session_state.logged_in = True
                st.success("✅ Login realizado com sucesso!")
            else:
                st.error("❌ Usuário ou senha incorretos.")

# ====== Inicialização de sessão ======
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ====== Aplicativo Principal ======
if not st.session_state.logged_in:
    login_screen()
else:
    with st.sidebar:
        st.header("⚙️ Configurações")
        st.markdown("Este aplicativo realiza **seleção automática de variáveis** com técnicas avançadas de ML.")
        st.markdown("Pipeline: engenharia de atributos, amostragem, LightGBM com Optuna, Rede Neural e download final.")
    
    feature_selection_screen()
