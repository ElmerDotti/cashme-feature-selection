# 🧠 CashMe Feature Selection App

Aplicativo interativo desenvolvido para análise, engenharia e **seleção automática de variáveis** em problemas de classificação com alta dimensionalidade, usando técnicas modernas de pré-processamento, SHAP, AutoML, PCA, e validação por PSI.

> Desenvolvido como parte de um desafio técnico da CashMe.

---

## 🚀 Acesse o App Online

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ElmerDotti/cashme-feature-selection/main/streamlit_app.py)

---

## 🧩 Funcionalidades

✅ Upload de arquivos `X.csv` e `y.csv`  
✅ Pipeline de tratamento e engenharia de features automatizada  
✅ Criação de variáveis: `lag1`, `variação`, `entropia`, `score`  
✅ Visualizações: histogramas, boxplots, matriz de correlação  
✅ Seleção automática de variáveis com LightGBM + SHAP  
✅ Comparação automática de modelos com AutoML  
✅ Download da matriz final  
✅ Login com autenticação  
✅ Barra de progresso dinâmica  
✅ Pronto para deploy no Streamlit Cloud

---

## 📁 Como usar localmente

```bash
# 1. Clone o repositório
git clone https://github.com/ElmerDotti/cashme-feature-selection.git
cd cashme-feature-selection

# 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação
streamlit run streamlit_app.py
