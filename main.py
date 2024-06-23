import streamlit as st
from multiapp import MultiApp
from despliegue import home
from despliegue.semana11 import modelo_hibrido, modelo_rf, modelo_svm, modelo_lstm, modelo_knn
from despliegue.semana10.parte1 import lstm
from despliegue.semana10.parte1 import rnn
from despliegue.semana10.parte1 import mlp

app = MultiApp()
st.markdown("# Equipo A - Inteligencia de Negocios ")

#Agregar modelos aki
app.add_app("Home", home.app)
app.add_app("Semana 11 - Modelo Random Forest", modelo_rf.app)
app.add_app("Semana 11 - Modelo SVR", modelo_svm.app)
app.add_app("Semana 11 - Modelo LSTM", modelo_lstm.app)
app.add_app("Semana 11 - Modelo KNN", modelo_knn.app)
app.add_app("Semana 11 - Modelo Híbrido", modelo_hibrido.app)
app.add_app("Semana 10 - Modelo LSTM", lstm.app)
app.add_app("Semana 10 - Modelo RNN", rnn.app)
app.add_app("Semana 10 - Modelo MLP", mlp.app)


# The main app
app.run()
