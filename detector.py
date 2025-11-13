import joblib
import pandas as pd
import numpy as np

# Load model and symptoms
model = joblib.load("disease_prediction_best_model.pkl")
symptoms = joblib.load("symptom_columns.pkl")

print("Enter symptoms (yes/no):")

# Gather user inputs
user_input = []
for s in symptoms:
    val = input(f"{s}: ").strip().lower()
    user_input.append(1 if val in ["yes", "y", "1"] else 0)

# Convert to DataFrame
user_df = pd.DataFrame([user_input], columns=symptoms)

# Get prediction probabilities for all diseases
proba = model.predict_proba(user_df)[0]

# Sort probabilities (highest first)
top_indices = np.argsort(proba)[::-1][:3]  # top 3
disease_names = model.classes_[top_indices]
probabilities = proba[top_indices] * 100

# Display top 3 results
print("\n🩺 Top 3 Most Likely Diseases:\n")
for i in range(len(disease_names)):
    print(f"{i+1}. {disease_names[i]} — {probabilities[i]:.2f}% confidence")

# Display the final prediction (best match)
print(f"\n✅ Most likely disease: {disease_names[0]}")
