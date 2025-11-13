import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# Load model + symptom columns
# --------------------------
model = joblib.load("disease_prediction_best_model.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

# --------------------------
# Streamlit configuration
# --------------------------
st.set_page_config(
    page_title="AI Health Disease Detector",
    page_icon="🧬",
    layout="centered",
)

# Custom CSS for card style + dark mode friendliness
st.markdown("""
<style>
body { background-color: #0E1117; color: #FAFAFA; }
.card {
    background-color: #1E1E1E;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 0 15px rgba(255,255,255,0.05);
    margin-bottom: 1rem;
}
h1, h2, h3, h4 { color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Header
# --------------------------
st.title("🩺 AI Health Disease Predictor")
st.caption("Select your symptoms below and get a ranked prediction of possible diseases.")

# --------------------------
# Symptom selection
# --------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 1: Select Symptoms")
    selected_symptoms = st.multiselect(
        "Choose all that apply:",
        symptom_columns,
        help="You can select multiple symptoms."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Prediction
# --------------------------
if st.button("🔍 Predict Disease", use_container_width=True):
    if not selected_symptoms:
        st.warning("Please select at least one symptom before predicting.")
    else:
        input_data = [1 if s in selected_symptoms else 0 for s in symptom_columns]
        input_df = pd.DataFrame([input_data], columns=symptom_columns)

        prediction = model.predict(input_df)[0]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧬 Predicted Result")
        st.success(f"**{prediction}**")

        # Probability output
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            top_idx = np.argsort(proba)[::-1]
            top3_idx = top_idx[:3]

            st.write("### Confidence Overview:")
            for i in top3_idx:
                conf = proba[i] * 100
                st.write(f"**{model.classes_[i]}**")
                st.progress(min(int(conf), 100))
                st.caption(f"{conf:.2f}% confidence")
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("⚕️ Built with Streamlit · Model trained using scikit-learn")
