from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
model_heart_failure = pickle.load(open('G:\\all models\\models\\heart_disease_xgb_model (1).pkl', 'rb'))
model_pad = pickle.load(open('G:\\all models\\models\\pad_xgb_model.pkl', 'rb'))
model_chd = pickle.load(open('G:\\all models\\models\\CHD_model.pkl', 'rb'))
model_htp = pickle.load(open('G:\\all models\\models\\heart_attack_xgb_model.pkl', 'rb'))

# Define the features required for each model
features_heart_failure = [
    'age', 'sex', 'chest_pain', 'resting_bp', 'cholesterol', 
    'fasting_blood_sugar', 'resting_ecg', 'maximum_heart_rate', 
    'exercise_angina', 'old_peak', 'st_slope'
]

features_pad = [
    'age', 'gender', 'smoking_status', 'diabetes', 'hypertension', 
    'cholesterol', 'intermittent_claudication', 'rest_pain', 
    'cold_feet', 'skin_change'
]

features_chd = [
    'sbp', 'tobacco', 'ldl', 'adiposity', 'family_history', 
    'type_a', 'obesity', 'alcohol', 'age'
]

features_htp = [
    'age', 'sex', 'cholesterol', 'blood_pressure', 'heart_rate', 
    'diabetes', 'family_history', 'smoking', 'obesity', 'alcohol', 
    'exercise_hours_per_week', 'diet', 'previous_heart_problem', 
    'medication_use', 'stress_level', 'sedentary_hours_per_day', 
    'income', 'bmi', 'triglyceride', 'physical_activity_days', 
    'sleep_hours_per_day', 'country', 'continent', 'hemisphere'
]

# Default value to fill missing features
DEFAULT_VALUE = 0.0  # Use 0, mean, or median based on training preprocessing

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json

        # Function to extract features for a specific model
        def extract_features(model_features):
            # For each required feature, either use the provided value or fill with default
            extracted = [
                float(data[feature]) if feature in data and data[feature] != "" else DEFAULT_VALUE
                for feature in model_features
            ]
            return extracted

        # Extract features for each model
        hf_features = extract_features(features_heart_failure)
        pad_features = extract_features(features_pad)
        chd_features = extract_features(features_chd)
        htp_features = extract_features(features_htp)

        predictions = {}

        # Predict for each disease and skip if there's an error
        try:
            prediction_hf = model_heart_failure.predict_proba([hf_features])[0][1]
            predictions["Heart Failure"] = float(prediction_hf)  # Convert to standard float
        except Exception as e:
            predictions["Heart Failure"] = f"Error: {str(e)}"

        try:
            prediction_pad = model_pad.predict_proba([pad_features])[0][1]
            predictions["Peripheral Artery Disease (PAD)"] = float(prediction_pad)  # Convert to standard float
        except Exception as e:
            predictions["Peripheral Artery Disease (PAD)"] = f"Error: {str(e)}"

        try:
            prediction_chd = model_chd.predict_proba([chd_features])[0][1]
            predictions["Coronary Heart Disease (CHD)"] = float(prediction_chd)  # Convert to standard float
        except Exception as e:
            predictions["Coronary Heart Disease (CHD)"] = f"Error: {str(e)}"

        try:
            prediction_htp = model_htp.predict_proba([htp_features])[0][1]
            predictions["Hypertension (HTP)"] = float(prediction_htp)  # Convert to standard float
        except Exception as e:
            predictions["Hypertension (HTP)"] = f"Error: {str(e)}"

        # Determine the highest risk disease
        valid_predictions = {k: v for k, v in predictions.items() if isinstance(v, float)}
        highest_risk = max(valid_predictions, key=valid_predictions.get) if valid_predictions else "No valid predictions"

        return jsonify({
            "predictions": predictions,
            "highest_risk": highest_risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
