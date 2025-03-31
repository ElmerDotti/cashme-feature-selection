import streamlit as st
from feature_selection import feature_selection_screen

st.set_page_config(page_title="🏦 CashMe - Feature Selection", layout="wide")

st.title("🏦 Desafio CashMe - Feature Selection")
st.markdown("""
Este aplicativo realiza **seleção de variáveis** utilizando:
- Criação automática de features derivadas (razões temporais e score com entropia)
- Otimização de modelo **LightGBM** com **Optuna**
- **Seleção automática** das features mais relevantes
- **Redução de dimensionalidade** via PCA
- Interpretação via **SHAP**
""")

feature_selection_screen()
