import pandas as pd
import streamlit as st

# Cargar el modelo (asegúrate de que el archivo sea un modelo guardado correctamente)
model = pd.read_pickle("./models/random_forest_classifier_default_42.sav")  # Cambié a pd.read_pickle para cargar el modelo

# Diccionario de clases
class_dict = {
    "0": "Alumno no admitido",
    "1": "Alumno admitido",
}

# Título de la aplicación
st.title("Admisión de Alumnos - Predicción del Modelo")

# Ingreso de nombre
name = st.text_input("Ingrese su Nombre:")

# Ingreso de edad
val2 = st.slider("Edad", min_value=18, max_value=22, step=1)

# Ingreso de género
val3 = st.slider("Género", min_value=0, max_value=1, step=1)

# Selección de ciudad
ciudades = ["Nueva York", "Los Ángeles", "Londres", "Madrid", "Tokio", "Buenos Aires", "Ciudad de México", "Sídney", "Berlín", "París"]
val4 = st.selectbox("Ciudad", ciudades)

# Ingreso de examen de admisión
val5 = st.slider("Examen de admisión", min_value=6, max_value=10, step=1)

# Ingreso de porcentaje de escuela secundaria
val6 = st.slider("Porcentaje Escuela Secundaria", min_value=60, max_value=100, step=10)

# Botón para realizar la predicción
if st.button("Predecir"):
    # Convertir la ciudad seleccionada a un índice o valor numérico si es necesario
    ciudad_index = ciudades.index(val4)  # Obtener el índice de la ciudad seleccionada
    prediction = str(model.predict([[name, val2, val3, ciudad_index, val5, val6]])[0])
    pred_class = class_dict[prediction]
    st.write("Predicción:", pred_class)
