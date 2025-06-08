# app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, jsonify
from backend.app.config import DATASETS, ML_MODELS, CF_METHODS
from backend.app.chains import create_data_collection_chain, create_explanation_chain
from backend.app.utils import load_dataset, format_counterfactual
from backend.app.model_manager import ModelManager
import uuid

app = Flask(__name__)
model_manager = ModelManager(datasets_config=DATASETS)
sessions = {}
collection_chain = create_data_collection_chain()
explanation_chain = create_explanation_chain()

@app.route('/')
def index():
    dataset_options = list(DATASETS.keys())
    model_options = list(ML_MODELS.keys())
    cf_method_options = list(CF_METHODS.keys())
    
    return render_template('index.html',
                           datasets=dataset_options,
                           models=model_options,
                           cf_methods=cf_method_options)

@app.route('/start_session', methods=['POST'])
def start_session():
    # Extract all parameters from request FIRST
    dataset_choice = request.json.get('dataset')
    model_choice = request.json.get('model')
    cf_method_choice = request.json.get('cf_method')
    
    # Validate all inputs
    if not dataset_choice or dataset_choice not in DATASETS:
        return jsonify({"error": "Invalid dataset selection"}), 400
    if not model_choice or model_choice not in ML_MODELS:
        return jsonify({"error": "Invalid model selection"}), 400
    if not cf_method_choice or cf_method_choice not in CF_METHODS:
        return jsonify({"error": "Invalid counterfactual method selection"}), 400

    session_id = str(uuid.uuid4())
    dataset_config = DATASETS[dataset_choice].copy()

    try:
        dataset_config["df"] = load_dataset(dataset_config)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # Create session with all validated parameters
    sessions[session_id] = {
        "dataset_config": dataset_config,
        "collected_data": [],
        'model_choice': model_choice,  # Changed key to match usage in submit_answer
        'cf_method_choice': cf_method_choice,  # Changed key to match usage in submit_answer
        "current_index": 0,
        "dataset_choice": dataset_choice  # Added this for submit_answer
    }

    return get_next_question(session_id)

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    session_id = request.json['session_id']
    answer = request.json['answer']

    if session_id not in sessions:
        return jsonify({"error": "Invalid session ID"}), 400
        
    session = sessions[session_id]
    dataset_config = session["dataset_config"]
    features = list(dataset_config["feature_types"].keys())
    current_index = session["current_index"]

    # Convert and store answer
    feature = features[current_index]
    feature_type = dataset_config["feature_types"][feature]

    try:
        if feature_type == "numeric":
            value = float(answer)
        else:
            value = answer
        session["collected_data"].append(value)
    except ValueError:
        return jsonify({"error": f"Invalid value for {feature}. Please enter a number."}), 400

    # Move to next feature
    session["current_index"] += 1

    if session["current_index"] < len(features):
        return get_next_question(session_id)
    else:
        # Data collection complete
        user_data = session["collected_data"]
        features = list(dataset_config["feature_types"].keys())
        
        # Convert to dictionary format for prediction
        user_data_dict = {feature: value for feature, value in zip(features, user_data)}
        
        try:
            prediction = model_manager.predict(
                dataset_name=session["dataset_choice"],
                model_type=session["model_choice"],  # Fixed key name
                input_data=user_data_dict
            )[0]
            
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        # Simulate counterfactual processing
        counterfactual_result = {
            "user_data": user_data,
            "original_prediction": 1,
            "required_changes": [
                {"feature": "chol", "current": 280, "new": 200},
                {"feature": "trestbps", "current": 150, "new": 120}
            ],
            "new_prediction": 0,
            "confidence": "High"
        }
        
        # Generate explanation
        formatted_cf = format_counterfactual(counterfactual_result, dataset_config)
        explanation = explanation_chain.invoke(formatted_cf)

        # Clean up session
        del sessions[session_id]
        
        return jsonify({
            "status": "complete",
            "user_data": user_data,
            "explanation": explanation.strip()
        })

def get_next_question(session_id):
    session = sessions[session_id]
    dataset_config = session["dataset_config"]
    features = list(dataset_config["feature_types"].keys())
    current_index = session["current_index"]

    # Generate question prompt
    response = collection_chain.invoke({
        "dataset_name": dataset_config["name"],
        "features": ", ".join(features),
        "collected_count": len(session["collected_data"]),
        "total_features": len(features),
        "next_feature": features[current_index]
    })

    return jsonify({
        "session_id": session_id,
        "question": response.strip(),
        "feature": features[current_index],
        "feature_type": dataset_config["feature_types"][features[current_index]],
        "progress": f"{current_index + 1}/{len(features)}"
    })

if __name__ == '__main__':
    app.run(debug=True)
