from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

def create_data_collection_chain():
    template = """
    You are helping collect data for a {dataset_name} prediction system.
    
    Current progress: {collected_count} out of {total_features} features collected.
    
    Next feature to collect: {next_feature}
    
    Create a simple, clear question to ask the user for the value of "{next_feature}".
    
    Instructions:
    - Ask only for the specific feature mentioned
    - Keep the question short and easy to understand
    - Do not ask for multiple values at once
    - Make the question conversational and friendly
    
    Question:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

def create_explanation_chain():
    template = """
    You are providing a medical explanation based on the following information:

    Dataset: {dataset_name}
    Original Prediction: {original_prediction} ({original_class})
    Alternative Prediction: {new_prediction} ({new_class})
    Confidence Level: {confidence}

    Changes needed to get different outcome:
    {changes}

    User's current data:
    {user_data_str}

    Please provide a clear, simple explanation that:
    1. Explains what the current prediction means
    2. Describes what changes would lead to a different outcome
    3. Explains why these changes matter
    4. Uses simple, non-technical language
    5. Is supportive and informative

    Keep the explanation concise but helpful.
    
    Explanation:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model