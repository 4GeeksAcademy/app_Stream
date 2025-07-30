import pandas as pd
import streamlit as st
from joblib import load

# Cargar el modelo desde el archivo
model = load("./models/random_forest_regressor_default_42.sav")
print("✅ Modelo cargado exitosamente!")

# Título de la aplicación
st.title("HIV - Predicción del Modelo")

# Ingreso de edad
val1 = st.slider("Sexo", min_value=18, max_value=60, step=1)

# Ingreso de género
val2 = st.slider("Año", min_value=2019, max_value=2023, step=1)


# Ingreso de examen de admisión
val3 = st.slider("Grupo Etario", min_value=1, max_value=9, step=1)


# Botón para realizar la predicción
if st.button("Predecir"):
    # Verificar que el nombre no esté vacío
    
        prediction = str(model.predict([[val1, val2, val3]])[0])
        st.write("Prediction:", prediction)
    
