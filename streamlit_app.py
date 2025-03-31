import streamlit as st
from feature_selection import feature_selection_screen

st.set_page_config(page_title="Feature Selection CashMe", layout="wide")
feature_selection_screen()
