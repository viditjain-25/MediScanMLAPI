from flask import Flask, request, jsonify
from flask_cors import CORS
from symptoms import predict_disease
import os

# ✅ Create Flask app instance
app = Flask(__name__)
CORS(app)

# ✅ Home route
@app.route("/", methods=["GET"])
def home():
    return "✅ MediScanMLAPI is running. Use POST /predict to send symptoms."

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", "")
        print("🧪 Received symptoms:", symptoms)
        result = predict_disease(symptoms)
        return jsonify(result), 200
    except Exception as e:
        print("🔥 Exception in /predict:", e)
        return jsonify({"error": str(e)}), 500

# ✅ Allow Render to set dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
