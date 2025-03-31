import streamlit as st
from feature_selection import feature_selection_screen

st.set_page_config(page_title="🏦 CashMe Feature Selection", layout="wide")

st.title("🏦 Desafio CashMe - Feature Selection com LightGBM + SHAP")
st.markdown("""
Este aplicativo permite:

- 📂 Carregar arquivos de entrada `X.csv` e `y.csv`
- 🧠 Criar variáveis derivadas automaticamente
- ⚙️ Otimizar o modelo LightGBM via Optuna
- 📌 Selecionar as melhores features com base na importância média
- 🌟 Interpretar os resultados com SHAP
- 📥 Baixar os resultados como CSV e PNG
""")

# Executa a tela principal
feature_selection_screen()
