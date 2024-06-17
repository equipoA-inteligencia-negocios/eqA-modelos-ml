import streamlit as st
from multiapp import MultiApp
from despliegue import home, modelo_hibrido, modelo_rf, modelo_svm, modelo_lstm

app = MultiApp()
st.markdown("# Equipo A - Inteligencia de Negocios ")

#Agregar modelos aki
app.add_app("Home", home.app)
app.add_app("Modelo Híbrido", modelo_hibrido.app)
app.add_app("Modelo Random Forest", modelo_rf.app)
app.add_app("Modelo SVR", modelo_svm.app)
app.add_app("Modelo LSTM", modelo_lstm.app)

# The main app
app.run()
