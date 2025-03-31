import streamlit as st
from feature_selection import feature_selection_screen

# Título do App
st.set_page_config(page_title="🏦 Desafio CashMe - Feature Selection", layout="wide")

st.title("🏦 Desafio CashMe - Feature Selection")
st.write("Este aplicativo permite realizar seleção de variáveis com técnicas avançadas de machine learning.")

# Rodar a interface principal
feature_selection_screen()
