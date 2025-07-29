import pandas as pd
import streamlit as st

# Cargar el modelo (asegúrate de que el archivo sea un modelo guardado correctamente)
model = pd.read_pickle("./models/random_forest_regressor_default_42.sav")  # Cambié a pd.read_pickle para cargar el modelo


# Título de la aplicación
st.title("HIV - Predicción del Modelo")

# Ingreso de edad
val1 = st.slider("Sexo", min_value=18, max_value=22, step=1)

# Ingreso de género
val2 = st.slider("Año", min_value=0, max_value=1, step=1)


# Ingreso de examen de admisión
val3 = st.slider("Grupo Etario", min_value=6, max_value=10, step=1)


# Botón para realizar la predicción
if st.button("Predecir"):
    # Verificar que el nombre no esté vacío
    
        prediction = str(model.predict([[val1, val2, val3]])[0])
        st.write("Prediction:", prediction)
    
