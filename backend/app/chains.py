# chains.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

def create_data_collection_chain():
    template = """
         You are an AI assistant collecting user data for the {dataset_name} dataset.
         Your task is to collect values for all features required for diagnosis.

         **Feature list (in this order):**
         {features}

         **Instructions:**
         1. Ask a *very short* question for each feature, in the *exact* order given above (e.g., "What is your age?").
         2. Ask *only one* question at a time.
         3. After the user replies:
            - If the feature is numeric, convert the input to a valid number.
            - If the input is of the wrong type (e.g., "female" instead of 0), intelligently map it to the correct format without asking again.
            - If the feature is categorical, store the value as-is, or standardize if needed.
         4. Do not repeat or skip features.
         5. After collecting all features, output ONLY the following:
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

    Your task is to explain the changes needed to achieve the desired outcome ({new_class}) based on the counterfactual analysis.
    Provide a very short, simple, clear explanation in plain English, focusing on:
    1. What the original prediction was
    2. What changes would lead to a different outcome
    3. Why these changes are significant
    
    Make the explanation empathetic and actionable. The explanation should be suitable for a non-technical audience.
    Do not include any technical jargon or complex terms. Do not imagine any additional context or data.
    
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model