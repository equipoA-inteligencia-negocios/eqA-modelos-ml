import streamlit as st
from multiapp import MultiApp
from despliegue import home, modelo_hibrido

app = MultiApp()
st.markdown("# Equipo A - Inteligencia de Negocios ")

#Agregar modelos aki
app.add_app("Home", home.app)
app.add_app("Modelo Híbrido", modelo_hibrido.app)

# The main app
app.run()