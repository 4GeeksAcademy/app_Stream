import pandas as pd
import streamlit as st

# Cargar el modelo (asegúrate de que el archivo sea un modelo guardado correctamente)
model = pd.read_pickle("./models/decision_tree_classifier_default_42.sav")  # Cambié a pd.read_pickle para cargar el modelo

# Diccionario de clases
class_dict = {
    "0": "Alumno no admitido",
    "1": "Alumno admitido",
    "-1": "Clase desconocida",
}

# Título de la aplicación
st.title("Admisión de Alumnos - Predicción del Modelo")

# Ingreso de edad
val1 = st.slider("Edad", min_value=18, max_value=22, step=1)

# Ingreso de género
val2 = st.slider("Género", min_value=0, max_value=1, step=1)


# Ingreso de examen de admisión
val3 = st.slider("Examen de admisión", min_value=6, max_value=10, step=1)

# Ingreso de porcentaje de escuela secundaria
val4 = st.slider("Porcentaje Escuela Secundaria", min_value=60, max_value=100, step=10)

# Botón para realizar la predicción
if st.button("Predecir"):
    # Verificar que el nombre no esté vacío
    
        prediction = str(model.predict([[val1, val2, val3, val4]])[0])
        pred_class = class_dict[prediction]
        st.write("Prediction:", pred_class)
    
