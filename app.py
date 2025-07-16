from flask import Flask, request, jsonify
from flask_cors import CORS
from symptoms import predict_disease
import os

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", "")
        print("🧪 Received:", symptoms)
        result = predict_disease(symptoms)
        return jsonify(result), 200
    except Exception as e:
        import traceback
        print("🔥 Full error:")
        traceback.print_exc()  # This gives detailed info in Render logs
        return jsonify({"error": str(e)}), 500

