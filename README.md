# Symptoms AI - Your friendly GP doctor

**DISCLAMER: THIS IS JUST FOR EDUCATIONAL AND TESTING PURPOSES, IT HAS NO MEDICAL RELEVANCE**

This simple chatbot will:
1. Take as input symptoms
2. Query a FAISS embedding DB trained on a symptoms dataset
3. Enrich your query with the embeddings found
4. Set a System prompt to answer as a doctor
4. Query LLaMA 3.2 3B
5. Receive an answer from the LLM

## Example
Input:
```
Please enter your symptoms: I have chest pain and fever
```

Query sent to LLM:
```
I have chest pain and fever.
Context 
Here are related symptoms and diseases:
- Disease: Heart attack, Symptoms:  vomiting,  sweating,  chest pain
```

AI doctor's Response:
```
Based on the symptoms you've shared, chest pain and fever, I need to consider a few possibilities.

From the context you provided, it seems that chest pain is a common symptom of a Heart attack. However, you've also mentioned fever, which is not typically a primary symptom of a Heart attack.

One potential connection is that you might be experiencing a Heart attack, but the fever could be a secondary symptom. Fever can occur in some cases of Heart attack, especially if there's an associated infection.

However, it's also possible that the fever could be related to another underlying condition.

To narrow down the possibilities, I have a few more questions:

1. How long have you been experiencing chest pain and fever?
2. Are you experiencing any shortness of breath or difficulty breathing?
3. Have you recently had any strenuous physical activity or stress?
4. Have you noticed any other symptoms, such as vomiting or dizziness?

Please feel free to share any additional information that might help me better understand your symptoms.
```

## Create venv and download dependencies
```
# create venv
python -m venv venv
venv\Scripts\activate
# update pip
pip install --upgrade pip
pip install sentence-transformers faiss-cpu pandas transformers langchain sentencepiece protobuf accelerate torch torchvision torchaudio
# it has to match your cuda version
```

## Hugginface login
```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# you need an hf access token: https://huggingface.co/docs/hub/en/security-tokens
```