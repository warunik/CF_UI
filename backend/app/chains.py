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


#####################################################################

# from langchain_openai import ChatOpenAI  # Replace Ollama
# from langchain_core.prompts import ChatPromptTemplate
# import os

# # Set your API key (get free credits: https://platform.openai.com/)
# os.environ["OPENAI_API_KEY"] = "sk-..."  

# def create_data_collection_chain():
#     template = """[Your template unchanged]"""  
#     prompt = ChatPromptTemplate.from_template(template)
#     return prompt | ChatOpenAI(model="gpt-3.5-turbo")  # Free tier eligible

# def create_explanation_chain():
#     template = """[Your template unchanged]"""
#     prompt = ChatPromptTemplate.from_template(template)
#     return prompt | ChatOpenAI(model="gpt-3.5-turbo")

#####################################################################

# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# # Load GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token  # Set pad token using string
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.config.pad_token_id = tokenizer.pad_token_id
# model.eval()

# # Define context and question
# context = """The Indian Premier League (IPL) is a professional Twenty20 cricket league in India organised by the Board of Control for Cricket in India (BCCI). Founded in 2007, the league features ten state-based or city-based franchise teams. The IPL is the most popular and richest cricket league in the world and is held between March and May. It has an exclusive window in the Future Tours Programme of the International Cricket Council (ICC), resulting in fewer international cricket tours occurring during the IPL seasons. It is also the most viewed sports competition in India, as per the Broadcast Audience Research Council.

# In 2014, it ranked sixth in attendance among all sports leagues. In 2010, the IPL became the first sporting event to be broadcast live on YouTube. Inspired by the success of the IPL, other Indian sports leagues have been established. IPL is the second-richest sports league in the world by per-match value, after the NFL. In 2023, the league sold its media rights for the next 4 seasons for US$6.4 billion to Viacom18 and Star Sports, which meant that each IPL match was valued at $13.4 million. As of 2024, there have been seventeen seasons of the tournament. The current champions are the Kolkata Knight Riders, who won the 2024 season after defeating the Sunrisers Hyderabad in the final. In just six years, the IPL's value has more than doubled, reaching $12 billion in 2024."""

# question = "When did IPL start?"
# prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

# # Tokenize input with attention mask
# inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]

# # Generate output
# with torch.no_grad():
#     outputs = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         max_new_tokens=50,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id
#     )

# # Decode and extract generated answer
# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
# generated_answer = decoded[len(prompt):].strip()

# print("Generated Answer:\n", generated_answer)

#####################################################################################
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline, pipeline
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.llms import HuggingFacePipeline

# # ─── 1) Initialize the GPT‑2 tokenizer & model ─────────────────────────────────

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.config.pad_token_id = tokenizer.pad_token_id

# # If you want a stronger free model, you can swap "gpt2" for
# # "gpt2-medium", "gpt2-large", or "distilgpt2" here.

# # ─── 2) Wrap into a HuggingFace Text‑Generation pipeline ────────────────────

# # We use a transformers pipeline, then wrap it for LangChain
# text_gen = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     # any default gen args you'd like:
#     max_length=256,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9,
# )

# hf_llm = HuggingFacePipeline(
#     pipeline=text_gen,
#     # you can also pass additional model_kwargs if needed
# )

# # ─── 3) Re‑define your two chains with the identical prompt templates ───────

# def create_data_collection_chain():
#     template = """
#     You are helping collect data for a {dataset_name} prediction system.
    
#     Current progress: {collected_count} out of {total_features} features collected.
    
#     Next feature to collect: {next_feature}
    
#     Create a simple, clear question to ask the user for the value of "{next_feature}".
    
#     Instructions:
#     - Ask only for the specific feature mentioned
#     - Keep the question short and easy to understand
#     - Do not ask for multiple values at once
#     - Make the question conversational and friendly
    
#     Question:
#     """
    
#     prompt = ChatPromptTemplate.from_template(template)
#     return prompt | hf_llm


# def create_explanation_chain():
#     template = """
#     You are providing a medical explanation based on the following information:

#     Dataset: {dataset_name}
#     Original Prediction: {original_prediction} ({original_class})
#     Alternative Prediction: {new_prediction} ({new_class})
#     Confidence Level: {confidence}

#     Changes needed to get different outcome:
#     {changes}

#     User's current data:
#     {user_data_str}

#     Please provide a clear, simple explanation that:
#     1. Explains what the current prediction means
#     2. Describes what changes would lead to a different outcome
#     3. Explains why these changes matter
#     4. Uses simple, non-technical language
#     5. Is supportive and informative

#     Keep the explanation concise but helpful.
    
#     Explanation:
#     """
    
#     prompt = ChatPromptTemplate.from_template(template)
#     return prompt | hf_llm


#####################################################################################
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# # Global model and tokenizer setup
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.config.pad_token_id = tokenizer.pad_token_id
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# def generate_from_prompt(prompt: str, max_new_tokens: int = 100) -> str:
#     """Generate text from prompt using GPT-2"""
#     inputs = tokenizer(
#         prompt, 
#         return_tensors="pt", 
#         padding=True, 
#         truncation=True, 
#         max_length=512
#     ).to(device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     # Extract only the generated text (excluding input)
#     generated = outputs[0][inputs["input_ids"].shape[1]:]
#     return tokenizer.decode(generated, skip_special_tokens=True).strip()

# def create_data_collection_chain():
#     template = """
#     You are helping collect data for a {dataset_name} prediction system.
    
#     Current progress: {collected_count} out of {total_features} features collected.
    
#     Next feature to collect: {next_feature}
    
#     Create a simple, clear question to ask the user for the value of "{next_feature}".
    
#     Instructions:
#     - Ask only for the specific feature mentioned
#     - Keep the question short and easy to understand
#     - Do not ask for multiple values at once
#     - Make the question conversational and friendly
    
#     Question:
#     """
    
#     def chain(inputs: dict) -> str:
#         formatted_prompt = template.format(**inputs)
#         return generate_from_prompt(formatted_prompt, max_new_tokens=30)
    
#     return chain

# def create_explanation_chain():
#     template = """
#     You are providing a medical explanation based on the following information:

#     Dataset: {dataset_name}
#     Original Prediction: {original_prediction} ({original_class})
#     Alternative Prediction: {new_prediction} ({new_class})
#     Confidence Level: {confidence}

#     Changes needed to get different outcome:
#     {changes}

#     User's current data:
#     {user_data_str}

#     Please provide a clear, simple explanation that:
#     1. Explains what the current prediction means
#     2. Describes what changes would lead to a different outcome
#     3. Explains why these changes matter
#     4. Uses simple, non-technical language
#     5. Is supportive and informative

#     Keep the explanation concise but helpful.
    
#     Explanation:
#     """
    
#     def chain(inputs: dict) -> str:
#         formatted_prompt = template.format(**inputs)
#         return generate_from_prompt(formatted_prompt, max_new_tokens=150)\

##################################################################################


