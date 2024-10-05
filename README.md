## Create venv and dependencies
```
# create venv
python -m venv venv
venv\Scripts\activate
# update pip
pip install --upgrade pip
pip install sentence-transformers faiss-cpu pandas transformers langchain sentencepiece protobuf accelerate>=0.26.0 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# it has to match your cuda version
```

## huggingface login
```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# you need an hf access token: https://huggingface.co/docs/hub/en/security-tokens
```

## Install LLaMA 3.2 models
DEPRECATED:
https://www.llama.com/llama-downloads/
```
pip install llama-stack
llama model list
llama model list --show-all
llama model download --source meta --model-id  Llama3.2-3B

# When the script asks for your unique custom URL, please paste the URL below
# You need your own from www.llama.com
https://llama3-2-lightweight.llamameta.net/*?Policy=[...]
```