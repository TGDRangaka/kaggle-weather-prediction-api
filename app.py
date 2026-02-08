
import os
os.environ['KERAS_BACKEND'] = 'torch' # Use Torch backend for Keras 3

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration ---
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
SCALERS_DIR = MODELS_DIR # Assuming scalers are in the same directory

# Model configurations
# Key: Model ID used in API
# File: Filename in models directory
# Type: 'sklearn' (joblib) or 'keras' (tensorflow)
MODEL_CONFIG = {
    'lr': {'file': 'linear_regression_model.joblib', 'type': 'sklearn'},
    'rf': {'file': 'random_forest_model.joblib', 'type': 'sklearn'},
    'svr': {'file': 'svr_model.joblib', 'type': 'sklearn'},
    'svr_tuned': {'file': 'tuned_svr_model.joblib', 'type': 'sklearn'},
    'gbr': {'file': 'gbr_model.joblib', 'type': 'sklearn'},
    'gbr_tuned': {'file': 'tuned_gbr_model.joblib', 'type': 'sklearn'},
    'ann': {'file': 'ann_model.joblib', 'type': 'keras'},
    'ann_tuned': {'file': 'tuned_ann_model.joblib', 'type': 'keras'},
    'lstm': {'file': 'lstm_model.joblib', 'type': 'keras'}
}

# --- Global State ---
LOADED_MODELS = {}
LOADED_SCALERS = {}
MODEL_ERRORS = {}

# --- Loading Functions ---

def load_scalers():
    """Attempts to load scalers if they exist."""
    scaler_path = os.path.join(SCALERS_DIR, 'scaler_X.pkl') # Try .pkl first as per prompt
    if not os.path.exists(scaler_path):
         scaler_path = os.path.join(SCALERS_DIR, 'scaler.joblib') # Fallback

    if os.path.exists(scaler_path):
        try:
            LOADED_SCALERS['X'] = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {e}")
    else:
        print("No scaler found (scaler_X.pkl or scaler.joblib). Inference will run without scaling.")

def load_models():
    """Loads all configured models into memory."""
    global LOADED_MODELS, MODEL_ERRORS
    LOADED_MODELS = {}
    MODEL_ERRORS = {}

    for model_key, config in MODEL_CONFIG.items():
        file_path = os.path.join(MODELS_DIR, config['file'])
        
        try:
            if config['type'] == 'sklearn':
                if os.path.exists(file_path):
                    LOADED_MODELS[model_key] = joblib.load(file_path)
                    print(f"Loaded {model_key}")
                else:
                    MODEL_ERRORS[model_key] = f"File not found: {config['file']}"
            
            elif config['type'] == 'keras':
                # Use Keras 3 which can run on Torch backend
                try:
                    import keras
                    if os.path.exists(file_path):
                        if file_path.endswith('.joblib'):
                            LOADED_MODELS[model_key] = joblib.load(file_path)
                        else:
                            LOADED_MODELS[model_key] = keras.models.load_model(file_path)
                        print(f"Loaded {model_key} (Keras)")
                    else:
                        MODEL_ERRORS[model_key] = f"File not found: {config['file']}"
                except ImportError:
                    MODEL_ERRORS[model_key] = "Keras not installed."
                except Exception as e:
                    MODEL_ERRORS[model_key] = f"Error loading Keras model: {str(e)}"
                    
        except Exception as e:
            MODEL_ERRORS[model_key] = f"Error loading: {str(e)}"
            print(f"Failed to load {model_key}: {e}")

# Initial load
load_scalers()
load_models()


# --- Helper Functions ---

def validate_input(window):
    """Validates the 14x5 input window."""
    if not isinstance(window, list):
        return False, "Input must be a list."
    if len(window) != 14:
        return False, f"Expected 14 rows, got {len(window)}."
    
    for i, row in enumerate(window):
        if not isinstance(row, list):
            return False, f"Row {i} must be a list."
        if len(row) != 5:
            return False, f"Row {i} must have 5 values (Max, Min, Precip, Snow, SnowDepth)."
        # Check for numbers
        try:
            row_floats = [float(x) for x in row]
            if any(np.isnan(row_floats)):
                return False, f"Row {i} contains NaN values."
        except (ValueError, TypeError):
             return False, f"Row {i} must contain valid numbers."
             
    return True, ""

def preprocess_input(window, model_type):
    """
    Prepares input for prediction.
    - Scales if scaler is available.
    - Flattens for sklearn (1, 70).
    - Keeps as (1, 14, 5) for LSTM/RNN.
    """
    data = np.array(window, dtype=np.float32) # (14, 5)
    
    # Scale if scaler exists
    # Note: If scaler was fitted on (N, 5) or (N, 70), we need to match.
    # Usually standard scalers scale features (columns). So (14, 5).
    if 'X' in LOADED_SCALERS:
        try:
            # Assume scaler expects (N, 5) where N is samples * timesteps, or just features?
            # Standard way for sliding window: Scale the original features properly.
            # If the scaler was trained on the 5 features, we can scale the (14, 5) array.
            # However, `scaler.transform` expects 2D array.
            original_shape = data.shape
            data_reshaped = data.reshape(-1, 5) # (14, 5)
            data_scaled = LOADED_SCALERS['X'].transform(data_reshaped)
            data = data_scaled.reshape(original_shape)
        except Exception as e:
            print(f"Scaling failed: {e}. Using raw data.")
            
    # Reshape based on model type
    if model_type == 'sklearn':
        # Flatten: (1, 14*5) = (1, 70)
        return data.flatten().reshape(1, -1)
    elif model_type == 'keras':
        # LSTM/ANN: (1, 14, 5)
        # However, some ANNs might expect flattened input too. 
        # The prompt says: "For LSTM: shape (1, 14, 5)". 
        # It doesn't explicitly say for ANN. Usually ANN (MLP) takes flattened.
        # But if ANN was trained on the same data, maybe it expects flattened?
        # User says: "For sklearn models: flatten... For LSTM: shape (1, 14, 5)".
        # Returns for "ann" -> "For sklearn models: flatten... For LSTM".
        # I'll Assume ANN is like sklearn (flattened) unless it's a specific sequence model.
        # But since 'ann' is listed under Keras, I should check the prompt carefully.
        # "For sklearn models: flatten to shape (1, 70). For LSTM: shape (1, 14, 5)".
        # It implies ONLY LSTM gets (1, 14, 5). 
        # What about 'ann' (Keras)? If it's a standard Dense network, it needs flattened (1, 70).
        # I will treat 'ann' and 'ann_tuned' as needing flattened input, and ONLY 'lstm' getting 3D.
        
        # Checking prompt again: "For LSTM: shape (1, 14, 5)".
        # If I have a Keras ANN, it likely needs (1, 70).
         
        return data.reshape(1, 14, 5)

def preprocess_for_model(window, model_key):
    # Logic to decide shape based on model KEY, not just type, because Keras ANN != Keras LSTM
    data = np.array(window, dtype=np.float32) #(14, 5)
    
    # Scaling
    if 'X' in LOADED_SCALERS:
        try:
            data = LOADED_SCALERS['X'].transform(data) # Scale the (14, 5) directly
        except Exception as e:
            print(f"Scaling failed: {e}")

    if model_key == 'lstm':
        # Based on model inspection, LSTM expects (None, 70, 1)
        return data.flatten().reshape(1, 70, 1)
    else:
        # Default for sklearn and standard ANN: (None, 70)
        return data.flatten().reshape(1, -1)


# --- Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "loaded_models": list(LOADED_MODELS.keys()),
        "model_errors": MODEL_ERRORS
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        selected_models = req_data.get('models', [])
        window = req_data.get('window', [])
        
        # Validation
        is_valid, error_msg = validate_input(window)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
            
        results = {}
        errors = {}
        
        # Run predictions
        for model_key in selected_models:
            # Check availability
            if model_key not in LOADED_MODELS:
                errors[model_key] = MODEL_ERRORS.get(model_key, "Model not found or failed to load")
                continue
            
            model = LOADED_MODELS[model_key]
            
            try:
                # Prepare input
                input_data = preprocess_for_model(window, model_key)
                
                # Predict
                # Only Keras models support the 'verbose' parameter in predict()
                if MODEL_CONFIG.get(model_key, {}).get('type') == 'keras':
                    prediction = model.predict(input_data, verbose=0)
                else:
                    prediction = model.predict(input_data)
                
                # Extract value
                val = 0.0
                if isinstance(prediction, (np.ndarray, list)):
                    # Handle different output shapes (e.g. [[val]], [val])
                    flat_pred = np.array(prediction).flatten()
                    if len(flat_pred) > 0:
                        val = float(flat_pred[0])
                elif isinstance(prediction, (float, int)):
                    val = float(prediction)
                    
                results[model_key] = {"value": round(val, 2), "unit": "°F"}
                
            except Exception as e:
                errors[model_key] = f"Prediction failed: {str(e)}"
        
        response = {
            "predictions": results,
            "errors": errors,  # Include errors field as requested
            "meta": {
                "input_shape": {"rows": 14, "cols": 5},
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
