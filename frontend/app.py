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
    """Generate final results when all questions are answered"""
    try:
        session = sessions[session_id]
        user_data = session["collected_data"]
        dataset_config = session["dataset_config"]
        features = list(dataset_config["feature_types"].keys())
        
        # Convert to dictionary format for prediction
        user_data_dict = {feature: value for feature, value in zip(features, user_data)}

        # Get original prediction
        try:
            original_prediction = model_manager.predict(
                dataset_name=session["dataset_choice"],
                model_type=session["model_choice"],
                input_data=user_data_dict
            )
        except Exception as e:
            app.logger.error(f"Prediction failed: {str(e)}")
            # Use mock prediction for demo
            original_prediction = 0
            
        # Generate counterfactual (mock implementation)
        counterfactual_result = {
            "user_data": user_data,
            "original_prediction": original_prediction,
            "required_changes": generate_mock_changes(user_data_dict, session["dataset_choice"]),
            "new_prediction": 1 - original_prediction,  # Flip for demo
            "confidence": "High"
        }
        
        # Generate explanation
        try:
            formatted_cf = format_counterfactual(counterfactual_result, dataset_config)
            explanation_result = explanation_chain.invoke(formatted_cf)
            explanation = explanation_result if isinstance(explanation_result, str) else str(explanation_result)
            if not explanation or not explanation.strip():
                explanation = "Unable to generate explanation at this time."
        except Exception as e:
            app.logger.error(f"Explanation generation failed: {e}")
            explanation = generate_simple_explanation(counterfactual_result, session["dataset_choice"])

        # Map predictions to labels
        original_label = get_prediction_label(counterfactual_result["original_prediction"], session["dataset_choice"])
        new_label = get_prediction_label(counterfactual_result["new_prediction"], session["dataset_choice"])

        # Clean up session
        del sessions[session_id]
        
        # Return response matching frontend expectations
        return jsonify({
            "status": "complete",
            "user_data": user_data,
            "original_prediction": counterfactual_result["original_prediction"],
            "original_prediction_label": original_label,
            "new_prediction_label": new_label,
            "required_changes": counterfactual_result["required_changes"],
            "explanation": explanation.strip()
        })
        
    except Exception as e:
        app.logger.error(f"Error generating final results: {str(e)}")
        return jsonify({"error": f"Failed to generate results: {str(e)}"}), 500

def generate_mock_changes(user_data_dict, dataset_name):
    """Generate mock counterfactual changes based on dataset"""
    if dataset_name == "heart_disease":
        changes = []
        if "chol" in user_data_dict and user_data_dict["chol"] > 200:
            changes.append({
                "feature": "chol",
                "current": user_data_dict["chol"],
                "new": 200
            })
        if "trestbps" in user_data_dict and user_data_dict["trestbps"] > 120:
            changes.append({
                "feature": "trestbps",
                "current": user_data_dict["trestbps"],
                "new": 120
            })
        return changes if changes else [{"feature": "age", "current": user_data_dict.get("age", 50), "new": 45}]
    else:
        # Generic mock change
        first_key = list(user_data_dict.keys())[0]
        return [{"feature": first_key, "current": user_data_dict[first_key], "new": "improved_value"}]

def generate_simple_explanation(counterfactual_result, dataset_name):
    """Generate a simple explanation when LLM fails"""
    changes = counterfactual_result["required_changes"]
    if not changes:
        return "No significant changes needed to alter the prediction."
    
    change_descriptions = []
    for change in changes:
        change_descriptions.append(f"{change['feature']} from {change['current']} to {change['new']}")
    
    return f"To change the prediction, consider adjusting: {', '.join(change_descriptions)}."

def get_prediction_label(pred_value, dataset_name):
    """Map prediction values to human-readable labels"""
    if dataset_name == "heart_disease":
        return "Heart Disease Risk" if pred_value == 1 else "No Heart Disease Risk"
    elif dataset_name == "diabetes":
        return "Diabetes Risk" if pred_value == 1 else "No Diabetes Risk"
    else:
        return f"Class {pred_value}"

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