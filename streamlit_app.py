import streamlit as st
from feature_selection import feature_selection_screen

# ====== Configurações Gerais ======
st.set_page_config(
    page_title="🏦 CashMe - Feature Selection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== Tela de Login ======
def login():
    st.title("🏦 Desafio CashMe - Seleção de Variáveis")
    st.subheader("🔐 Acesso Restrito")

    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submit = st.form_submit_button("Entrar")

        if submit:
            if username == "cashme123" and password == "cashme123":
                st.session_state["authenticated"] = True
                st.success("Login realizado com sucesso!")
                st.experimental_rerun()
            else:
                st.error("Usuário ou senha inválidos!")

# ====== Execução Principal ======
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    with st.sidebar:
        st.header("⚙️ Configurações")
        st.markdown(
            "Este aplicativo realiza **seleção automática de variáveis** com base em ML/AI. "
            "O pipeline inclui engenharia de atributos, amostragem estratificada, LightGBM + Optuna, "
            "rede neural e histograma de importância relativa."
        )

    feature_selection_screen()
