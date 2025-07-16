import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Step 1: Load Data
# ---------------------------
df = pd.read_csv("C:\\Users\\HP\\Downloads\\archive\\dataset.csv")
desc_df = pd.read_csv("C:\\Users\\HP\\Downloads\\archive\\symptom_Description.csv")
severity_df = pd.read_csv("C:\\Users\\HP\\Downloads\\archive\\Symptom-severity.csv")
precaution_df = pd.read_csv("C:\\Users\\HP\\Downloads\\archive\\symptom_precaution.csv")

# Normalize disease names
df["Disease"] = df["Disease"].str.lower()
desc_df["Disease"] = desc_df["Disease"].str.lower()
precaution_df["Disease"] = precaution_df["Disease"].str.lower()
severity_df["Symptom"] = severity_df["Symptom"].str.lower()

# Detect symptom columns
symptom_cols = [col for col in df.columns if col.lower().startswith("symptom")]
pre_col = [col for col in precaution_df.columns if col.lower().startswith("precaution")]

# Combine symptoms into a single text
df["all_symptoms"] = df[symptom_cols].fillna("").apply(lambda x: " ".join(x), axis=1).str.lower().str.strip()
precaution_df["all_Precaution"] = precaution_df[pre_col].fillna("").apply(lambda row1: " ".join(row1), axis=1).str.lower().str.strip()
# ---------------------------
# Step 2: Convert to Vectors
# ---------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["all_symptoms"])

# ---------------------------
# Step 3: Prediction Function
# ---------------------------
def predict_disease(user_input, top_n=4):
    user_symptoms = [s.strip().lower() for s in user_input.split(",")]
    user_text = " ".join(user_symptoms)

    user_vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()


    disease_scores = {}

    for idx, score in enumerate(similarity):
        if score > 0:
            row = df.iloc[idx]
            disease = row["Disease"]

            # Get or initialize entry
            if disease not in disease_scores:
                disease_scores[disease] = {
                    "score": 0,
                    "matching": set(),
                    "non_matching": set(user_symptoms),
                    "rows": []
                }

            known = [str(row[col]).strip().lower() for col in symptom_cols if pd.notna(row[col])]
            matching = [sym for sym in user_symptoms if sym in known]

            # Update score and matching symptoms
            disease_scores[disease]["score"] += len(matching)
            disease_scores[disease]["matching"].update(matching)
            disease_scores[disease]["non_matching"] = disease_scores[disease]["non_matching"].difference(matching)
            disease_scores[disease]["rows"].append(idx)

    results = []
    for disease, info in disease_scores.items():
        desc_row = desc_df[desc_df["Disease"] == disease]
        description = desc_row["Description"].values[0] if not desc_row.empty else "No description"
        precaution_row = precaution_df[precaution_df["Disease"] == disease]
        precaution = precaution_row["all_Precaution"].values[0] if not precaution_row.empty else "No precaution"
        severity = desc_row["Severity"].values[0] if "Severity" in desc_row.columns and not desc_row.empty else "Unknown"

        results.append({
            "disease": disease.title(),
            "description": description,
            "precaution" : precaution,
            "severity": severity,
            "matching": list(info["matching"]),
            "non_matching": list(info["non_matching"]),
            "score": info["score"]
        })

    # Sort by matching symptom count and prefer mild severity
    results = sorted(results, key=lambda x: (-x["score"], x["severity"] if x["severity"].lower() == "mild" else "zz"))
    
    # Total score of all diseases for probability calculation
   # Total number of symptoms entered by the user
    total_user_symptoms = len(user_symptoms)

# Calculate probability based on match ratio with user symptoms
    for r in results:
       if total_user_symptoms > 0:
          r["probability"] = round((len(r["matching"]) / total_user_symptoms) * 100, 2)
       else:
          r["probability"] = 0.0

    return results[:top_n]

# ---------------------------
# Step 4: Run the Prediction
# ---------------------------
user_input =  "vomiting, headache, weakness_of_one_body_side, altered_sensorium"

predictions = predict_disease(user_input)

for i, res in enumerate(predictions, 1):
    print(f"\n{i}. Disease: {res['disease']} ({res['probability']}% chance)")
    print(f"Description: {res['description']}")
    print(f"Precaution: {res['precaution']}")
    print(f"Severity: {res['severity']}")
    print(f"✅ Matching symptoms: {', '.join(res['matching'])}")
    print(f"❌ Non-matching symptoms: {', '.join(res['non_matching'])}")
