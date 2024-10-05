
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

# 1. Load the FAISS index and embeddings model
def load_faiss_index(index_path, embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings_model = SentenceTransformer(embeddings_model_name)
    index = faiss.read_index(index_path)
    return index, embeddings_model

# 2. Function to query the FAISS index
def query_faiss(faiss_index, embeddings_model, query, k=5):
    query_embedding = embeddings_model.encode(query)
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    return indices, distances

# 3. Combine user's question with the returned results from FAISS
def combine_question_with_results(user_question, faiss_results, dataset):
    combined_question = user_question + ". Here are related symptoms and diseases:\n"
    for idx in faiss_results:
        disease = dataset.iloc[idx]['Disease']
        symptoms = dataset.iloc[idx][1:].dropna().tolist()
        combined_question += f"Disease: {disease}, Symptoms: {', '.join(symptoms)}\n"
    return combined_question

def load_llama_model(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Load the LLaMA model from huggingface
    """
    llm_pipeline = pipeline(
        "text-generation",
        model = model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        eos_token_id = [128001, 128009],
        pad_token_id = 128001,
    )
    return llm_pipeline

# 4. Function to interact with the LLaMA model
def query_llm_llama(combined_question, llm_pipeline, max_length=250):
    """
    Query the LLaMA LLM with a combined question.
    """
    message = [
        {
            "role": "system",
            "content": "You are a friendly doctor who is trying to understand what disease the user have",
        },
        {"role": "user", "content": f"{combined_question}"},
    ]
    response = llm_pipeline(combined_question, max_length=max_length, truncation=True)[0]['generated_text']
    return response

def run_chatbot_with_llama(index_path, dataset_path,):
    # 1. Load the FAISS index and dataset
    index, embeddings_model = load_faiss_index(index_path)
    dataset = pd.read_csv(dataset_path)

    # 2. Load LLaMA model
    llama_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    llm_pipeline = load_llama_model(llama_model_name)

    # 3. Get user input
    user_question = input("Please enter your symptoms: ")

    # 4. Query FAISS index
    faiss_results, distances = query_faiss(index, embeddings_model, user_question)

    # 5. Combine the user's question with retrieved FAISS results
    combined_question = combine_question_with_results(user_question, faiss_results[0], dataset)

    # 6. Query the LLaMA model with the combined question
    llm_response = query_llm_llama(combined_question, llm_pipeline) 

    # 7. Display the response
    print("\nDoctor's Response:\n")
    print(llm_response)

# Example usage
if __name__ == "__main__":
    run_chatbot_with_llama("vector_db\disease_symptoms_index.faiss", "dataset\symptoms_dataset_spaces.csv")