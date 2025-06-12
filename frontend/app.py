import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, jsonify
from backend.app.config import DATASETS, ML_MODELS, CF_METHODS
from backend.app.chains import create_data_collection_chain, create_explanation_chain
from backend.app.utils import load_dataset, format_counterfactual
from backend.app.model_manager import ModelManager
import uuid
import numpy as np
import re

app = Flask(__name__)
model_manager = ModelManager(datasets_config=DATASETS)
sessions = {}
collection_chain = create_data_collection_chain()
explanation_chain = create_explanation_chain()

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
    try:
        # Input validation
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.json
        app.logger.debug(f"Received start_session request: {data}")

        # Check parameter presence
        required = ['dataset', 'model', 'cf_method']
        missing = [p for p in required if p not in data]
        if missing:
            return jsonify({"error": f"Missing parameters: {', '.join(missing)}"}), 400

        # Validate parameter values
        dataset_choice = data['dataset']
        model_choice = data['model']
        cf_method_choice = data['cf_method']

        validation_errors = []
        if dataset_choice not in DATASETS:
            validation_errors.append(f"Invalid dataset: {dataset_choice}")
        if model_choice not in ML_MODELS:
            validation_errors.append(f"Invalid model: {model_choice}")
        if cf_method_choice not in CF_METHODS:
            validation_errors.append(f"Invalid CF method: {cf_method_choice}")
        
        if validation_errors:
            return jsonify({"error": "; ".join(validation_errors)}), 400

        # Log successful parameters
        app.logger.info(
            f"Starting session with: "
            f"dataset={dataset_choice}, "
            f"model={model_choice}, "
            f"cf_method={cf_method_choice}"
        )
        
        session_id = str(uuid.uuid4())
        dataset_config = DATASETS[dataset_choice].copy()

        # Create session with all validated parameters
        sessions[session_id] = {
            "dataset_config": dataset_config,
            "collected_data": [],
            'model_choice': model_choice,
            'cf_method_choice': cf_method_choice,
            "current_index": 0,
            "dataset_choice": dataset_choice
        }

        return get_next_question(session_id)
    
    except Exception as e:
        app.logger.error(f"Error in start_session: {str(e)}")
        return jsonify({"error": f"Failed to start session: {str(e)}"}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        if not data or 'session_id' not in data or 'answer' not in data:
            return jsonify({"error": "Missing session_id or answer"}), 400
            
        session_id = data['session_id']
        answer = data['answer']

        if session_id not in sessions:
            return jsonify({"error": "Invalid session ID"}), 400
            
        session = sessions[session_id]
        dataset_config = session["dataset_config"]
        features = list(dataset_config["feature_types"].keys())
        current_index = session["current_index"]

        if current_index >= len(features):
            return jsonify({"error": "All questions already answered"}), 400

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
            # Data collection complete - generate results
            return generate_final_results(session_id)
            
    except Exception as e:
        app.logger.error(f"Error in submit_answer: {str(e)}")
        return jsonify({"error": f"Failed to submit answer: {str(e)}"}), 500
    
def generate_final_results(session_id):
    try:
        session = sessions[session_id]
        collected_data = session["collected_data"]
        dataset_name = session["dataset_choice"]
        model_choice = session["model_choice"]
        cf_method_choice = session["cf_method_choice"]
        
        if not collected_data:
            return jsonify({"error": "No data collected"}), 400

        # Convert to dictionary format for prediction
        features = list(session["dataset_config"]["feature_types"].keys())
        user_data_dict = {feature: value for feature, value in zip(features, collected_data)}

        # Get original prediction
        try:
            prediction_result = model_manager.predict(
                dataset_name=dataset_name,
                model_type=model_choice,
                input_data=user_data_dict
            )
            
            # Extract class prediction from probabilities
            if isinstance(prediction_result, (list, np.ndarray)) and len(prediction_result) > 0:
                original_prediction = int(prediction_result[0]) if hasattr(prediction_result[0], 'item') else int(prediction_result[0])
            else:
                original_prediction = int(prediction_result) if hasattr(prediction_result, 'item') else int(prediction_result)
        except Exception as e:
            app.logger.error(f"Prediction failed: {str(e)}")
            original_prediction = 0  # Default fallback
            
        # Generate actual counterfactual explanation
        
        try:
            # Use the actual counterfactual generation method
            cf_explanation = model_manager.generate_counterfactual(
                model=model_choice,
                dataset=dataset_name,
                instance=user_data_dict,
                method=cf_method_choice
            )

            new_prediction = 1 - original_prediction
                
                
        except Exception as e:
            app.logger.error(f"Counterfactual generation failed: {str(e)}")
            # No fallback - leave required_changes as empty list
        
        # Create counterfactual result
        counterfactual_result = {
            "user_data": collected_data,
            "original_prediction": original_prediction,
            "required_changes": cf_explanation,
            "new_prediction": new_prediction,
            "confidence": "High"
        }

        # Convert enum values to their string representations
        serializable_changes = []
        for change in cf_explanation:
            serializable_change = change.copy()
            serializable_change["operator"] = str(change["operator"]).split('.')[-1]  # Convert enum to string
            serializable_changes.append(serializable_change)

        # Now use serializable_changes in your response

        dataset_config = session["dataset_config"]

        original_label = dataset_config['class_labels'].get(original_prediction, f"Class {original_prediction}")
        new_label = dataset_config['class_labels'].get(new_prediction, f"Class {new_prediction}")
        explanation = explanation_chain.invoke({    
            "dataset_name": dataset_config.get("name", dataset_name),
            "original_prediction": original_prediction,
            "original_class": original_label,
            "new_prediction": new_prediction,
            "new_class": new_label,
            "confidence": "High", 
            "changes": cf_explanation,
            "user_data_str": str(user_data_dict)
        })
        # Convert all numpy types to native Python types
        response_data = {
            "status": "complete",
            "user_data": convert_numpy_types(collected_data),
            "feature_names": features,
            "original_prediction": convert_numpy_types(original_prediction),
            "original_prediction_label": original_label,
            "new_prediction_label": new_label,
            "required_changes": serializable_changes,
            "explanation": explanation.strip()
        }

        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Error generating final results: {str(e)}")
        return jsonify({"error": f"Failed to generate results: {str(e)}"}), 500

def get_next_question(session_id):
    try:
        session = sessions[session_id]
        dataset_config = session["dataset_config"]
        features = list(dataset_config["feature_types"].keys())
        
        if not features:
            return jsonify({"error": "No features available for this dataset"}), 400

        current_index = session["current_index"]
        
        if current_index >= len(features):
            return jsonify({"error": "All questions already answered"}), 400

        # Generate question prompt
        try:
            response = collection_chain.invoke({
                "dataset_name": dataset_config.get("name", session["dataset_choice"]),
                "features": ", ".join(features),
                "collected_count": len(session["collected_data"]),
                "total_features": len(features),
                "next_feature": features[current_index]
            })
            question = response if isinstance(response, str) else str(response)
        except Exception as e:
            app.logger.error(f"Question generation failed: {e}")
            # Fallback to simple question
            question = f"Please enter a value for {features[current_index]}:"

        return jsonify({
            "session_id": session_id,
            "question": question.strip(),
            "feature": features[current_index],
            "feature_type": dataset_config["feature_types"][features[current_index]],
            "progress": f"{current_index + 1}/{len(features)}"
        })
        
    except Exception as e:
        app.logger.error(f"Error in get_next_question: {str(e)}")
        return jsonify({"error": f"Failed to get next question: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)