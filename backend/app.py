from pathlib import Path
import io
import pandas as pd
from flask import Flask, jsonify, request, send_file
from model_service import ModelService, ModelNotReadyError, ExplainabilityUnavailableError

app = Flask(__name__)
service = ModelService(Path(__file__).parent.parent / "models")

@app.get("/api/health")
def health():
    return jsonify(service.health())

@app.post("/api/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Attach a CSV file under the 'file' field."}), 400
    upload = request.files["file"]
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400
    try:
        frame = pd.read_csv(upload.stream)
        result = service.predict(frame)
        return jsonify(result)
    except ModelNotReadyError as exc:
        return jsonify({"status": "model_not_ready", "error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("[v0] prediction failed")
        return jsonify({"error": "Unable to process this CSV."}), 500

@app.post("/api/predict/download")
def predict_download():
    if "file" not in request.files:
        return jsonify({"error": "Attach a CSV file."}), 400
    try:
        frame = pd.read_csv(request.files["file"].stream)
        output = service.predict_frame(frame)
        buffer = io.BytesIO()
        output.to_csv(buffer, index=False)
        buffer.seek(0)
        return send_file(buffer, mimetype="text/csv", as_attachment=True, download_name="ids_predictions.csv")
    except ModelNotReadyError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

@app.post("/api/explain")
def explain():
    if "file" not in request.files:
        return jsonify({"error": "Attach a CSV file under the 'file' field."}), 400
    upload = request.files["file"]
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400
    try:
        frame = pd.read_csv(upload.stream)
        result = service.explain(frame)
        return jsonify(result)
    except ModelNotReadyError as exc:
        return jsonify({"status": "model_not_ready", "error": str(exc)}), 503
    except ExplainabilityUnavailableError as exc:
        return jsonify({"status": "explain_unavailable", "error": str(exc)}), 422
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("[v0] explanation failed")
        return jsonify({"error": "Unable to generate SHAP explanations for this CSV."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
