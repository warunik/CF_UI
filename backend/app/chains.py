# chains.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

def create_data_collection_chain():
    template = """
    You are an AI assistant collecting user data for the {dataset_name} dataset.
    Your task is to collect values for all features needed for diagnosis.

    **Features to collect (in this order):**
    {features}

    **Instructions:**
    1. Ask ONE question at a time for each feature in the exact order shown above.
    2. After receiving the user's answer:
       - If the feature is numeric, convert to number
       - If categorical, keep as string
    3. After collecting ALL features, output EXACTLY:
       ```json
       {{"status": "complete", "user_data": [value1, value2, ...]}}
       ```
    Current Progress:

    Features collected: {collected_count}/{total_features}

    Next feature: {next_feature}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

def create_explanation_chain():
    template = """
    You are an AI assistant providing counterfactual explanations for {dataset_name}.
    Below are the original user data and required changes:

    Original Prediction: {original_prediction} ({original_class})
    New Prediction: {new_prediction} ({new_class})
    Confidence: {confidence}

    Required Changes:
    {changes}

    user Data:
    {user_data_str}

    Your task is to explain the changes needed to improve the user's condition based on the counterfactual analysis.
    Provide a very simple, clear explanation in plain English, focusing on the changes made and their impact on the prediction.
    Do not include any technical jargon or complex terms. The explanation should be suitable for a non-technical audience.
    Do not generate new data or predictions, just explain the changes made.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model