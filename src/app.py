from pickle import load
import streamlit as st

model = load(open("./models/decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Extrovertido",
    "1": "Introvertido",
    
}

st.title("Personalidad - Model prediction")

val1 = st.slider("Apertura a nuevas experiencias", min_value = 1, max_value = 5, step = 1)
val2 = st.slider("Responsabilidad", min_value = 1, max_value = 5, step = 1)
val3 = st.slider("Extroversión", min_value = 1, max_value = 5, step = 1)
val4 = st.slider("Amabilidad", min_value = 1, max_value = 5, step = 1)
val5 = st.slider("Estabilidad emocional", min_value = 1, max_value = 1, step = 1)
val6 = st.slider("Habilidades sociales", min_value = 1, max_value = 5, step = 1)
val7 = st.slider("Energía y actividad", min_value = 1, max_value = 5, step = 1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5, val6, val7]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
