import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Connect to the SQLite DB
conn = sqlite3.connect("mediscan.db")

# Load tables into dataframes
df = pd.read_sql("SELECT * FROM diseases", conn)
description_data = pd.read_sql("SELECT * FROM descriptions", conn)
precaution_data = pd.read_sql("SELECT * FROM precautions", conn)
severity_data = pd.read_sql("SELECT * FROM symptoms", conn)

# (your prediction logic remains unchanged after this point)




# ---------------------------
# Step 1: Load Data
# ---------------------------
df = pd.read_csv("dataset.csv")
description_data = pd.read_csv("symptom_Description.csv")
precaution_data = pd.read_csv("symptom_precaution.csv")
severity_data = pd.read_csv("Symptom-severity.csv")

# Normalize text data
df["Disease"] = df["Disease"].str.lower()
description_data["Disease"] = description_data["Disease"].str.lower()
precaution_data["Disease"] = precaution_data["Disease"].str.lower()
severity_data["Symptom"] = severity_data["Symptom"].str.lower()

# Detect symptom columns
symptom_cols = [col for col in df.columns if col.lower().startswith("symptom")]
pre_col = [col for col in precaution_data.columns if col.lower().startswith("precaution")]

# Combine all symptoms into a single string
df["all_symptoms"] = df[symptom_cols].fillna("").apply(lambda x: " ".join(x), axis=1).str.lower().str.strip()
precaution_data["all_Precaution"] = precaution_data[pre_col].fillna("").apply(lambda x: " ".join(x), axis=1).str.lower().str.strip()

# Vectorize
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["all_symptoms"])

# ---------------------------
# Prediction Function
# ---------------------------
def predict_disease(user_input, top_n=4):
    # Normalize input: lowercase + replace spaces with underscores
    user_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input.split(",") if s.strip()]
    user_text = " ".join(user_symptoms)

    user_vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()

    disease_scores = {}

    for idx, score in enumerate(similarity):
        if score > 0:
            row = df.iloc[idx]
            disease = row["Disease"]

            if disease not in disease_scores:
                disease_scores[disease] = {
                    "score": 0,
                    "matching": set(),
                    "non_matching": set(user_symptoms),
                    "rows": []
                }

            known = [str(row[col]).strip().lower() for col in symptom_cols if pd.notna(row[col])]
            matching = [sym for sym in user_symptoms if sym in known]

            disease_scores[disease]["score"] += len(matching)
            disease_scores[disease]["matching"].update(matching)
            disease_scores[disease]["non_matching"] = disease_scores[disease]["non_matching"].difference(matching)
            disease_scores[disease]["rows"].append(idx)

    results = []

    for disease, info in disease_scores.items():
        desc_row = description_data[description_data["Disease"] == disease]
        description = desc_row["Description"].values[0] if not desc_row.empty else "No description available."

        precaution_row = precaution_data[precaution_data["Disease"] == disease]
        precaution = precaution_row["all_Precaution"].values[0] if not precaution_row.empty else "No precaution available."

        # ✅ Fix: Calculate severity from matching symptoms (not from description)
        matching_severities = severity_data[severity_data["Symptom"].isin(info["matching"])]
        if not matching_severities.empty:
            avg_severity_value = matching_severities["Severity"].astype(int).mean()
            if avg_severity_value >= 6:
                severity = "Severe"
            elif avg_severity_value >= 3:
                severity = "Moderate"
            else:
                severity = "Mild"
        else:
            severity = "Unknown"

        results.append({
            "disease": disease.title(),
            "description": description,
            "precaution": precaution,
            "severity": severity,
            "matching": list(info["matching"]),
            "non_matching": list(info["non_matching"]),
            "score": info["score"]
        })

    # Sort by matching score and prefer mild severity
    results = sorted(results, key=lambda x: (-x["score"], x["severity"] if x["severity"] == "Mild" else "zz"))

    total_user_symptoms = len(user_symptoms)
    for r in results:
        r["probability"] = round((len(r["matching"]) / total_user_symptoms) * 100, 2) if total_user_symptoms > 0 else 0.0

    # 🔁 Fallback: if nothing matched, return most similar disease
    if not results:
        best_idx = similarity.argmax()
        fallback_disease = df.iloc[best_idx]["Disease"]

        desc_row = description_data[description_data["Disease"] == fallback_disease]
        description = desc_row["Description"].values[0] if not desc_row.empty else "No description available."

        precaution_row = precaution_data[precaution_data["Disease"] == fallback_disease]
        precaution = precaution_row["all_Precaution"].values[0] if not precaution_row.empty else "No precaution available."

        results.append({
            "disease": fallback_disease.title(),
            "description": description,
            "precaution": precaution,
            "severity": "Unknown",
            "matching": [],
            "non_matching": user_symptoms,
            "score": 0,
            "probability": 0.0
        })

    return results[:top_n]
