from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
# import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
# Using global variables to load them once at startup
MODELS = {}
MODEL_ERRORS = {}

def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Linear Regression
    try:
        MODELS['linear_regression'] = joblib.load(os.path.join(base_path, 'linear_regression_model.joblib'))
        print("Linear Regression loaded.")
    except Exception as e:
        MODEL_ERRORS['linear_regression'] = str(e)
        print(f"Error loading Linear Regression: {e}")

    # Random Forest
    try:
        MODELS['random_forest'] = joblib.load(os.path.join(base_path, 'random_forest_model.joblib'))
        print("Random Forest loaded.")
    except Exception as e:
        MODEL_ERRORS['random_forest'] = str(e)
        print(f"Error loading Random Forest: {e}")

    # ANN
    # try:
    #     import tensorflow as tf
    #     MODELS['ann'] = tf.keras.models.load_model(os.path.join(base_path, 'ann_model.keras'))
    #     print("ANN loaded.")
    # except ImportError:
    #     MODEL_ERRORS['ann'] = "TensorFlow is not installed. Python 3.14 is currently not supported by TensorFlow."
    #     print("ANN skipped: TensorFlow not installed.")
    # except Exception as e:
    #     MODEL_ERRORS['ann'] = str(e)
    #     print(f"Error loading ANN: {e}")

load_models()

# Helper Functions for Prediction
def predict_linear(data):
    if 'linear_regression' not in MODELS:
        return {"error": MODEL_ERRORS.get('linear_regression', "Model not loaded")}
    
    input_data = np.array(data).reshape(1, -1)
    prediction = MODELS['linear_regression'].predict(input_data)
    return float(prediction[0])

def predict_rf(data):
    if 'random_forest' not in MODELS:
        return {"error": MODEL_ERRORS.get('random_forest', "Model not loaded")}
    
    input_data = np.array(data).reshape(1, -1)
    prediction = MODELS['random_forest'].predict(input_data)
    return float(prediction[0])

def predict_ann(data):
    if 'ann' not in MODELS:
        return {"error": MODEL_ERRORS.get('ann', "Model not loaded. Note: TensorFlow is required for ANN.")}
    
    import tensorflow as tf
    input_data = np.array(data).reshape(1, -1)
    prediction = MODELS['ann'].predict(input_data)
    return float(prediction[0][0])

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "server": "weather-predict-api"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Payload:
    {
        "models": ["linear_regression", "random_forest", "ann"],
        "data": [20.5, 21.0, ...]  # 14 days of temperatures
    }
    """
    try:
        req_data = request.get_json()
        selected_models = req_data.get('models', [])
        input_temps = req_data.get('data', [])

        if not input_temps or len(input_temps) != 14:
            return jsonify({"error": "Invalid data. Expected 14 temperature values."}), 400

        results = {}

        for model_key in selected_models:
            if model_key == 'linear_regression':
                results['linear_regression'] = predict_linear(input_temps)
            elif model_key == 'random_forest':
                results['random_forest'] = predict_rf(input_temps)
            elif model_key == 'ann':
                results['ann'] = predict_ann(input_temps)
            else:
                results[model_key] = "Unknown model selected"

        return jsonify({
            "input_data": input_temps,
            "predictions": results
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
