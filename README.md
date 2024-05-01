# Hindi-Language-AI-Chatbot-for-Enterprises-using-Qdrant-MLFlow-and-LangChain
RAG powered AI chatbot for Indian Language (Hindi) using LangChain, Ollama, Qdrant, and MLFlow

This repository contains the source code for a Hindi Language AI Chatbot for Enterprises using Qdrant, MLFlow, and LangChain. The notebook contains step-by-step instructions to create and train the chatbot, and it can be run in any Python environment that supports Jupyter Notebook.

Requirements

Python 3.10.13
Jupyter Notebook
MLFlow
LangChain
Qdrant






```
git clone https://github.com/gayatrivenugopal/Hindi-Aesthetics-Corpus.git
pip install langchain transformers qdrant-client accelerate torch bitsandbytes
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:v0.2
ollama pull llama2
ollama list
ollama run <name-of-model>
sudo apt-get update
sudo apt install docker
docker info
docker pull qdrant/qdrant
docker run -p 6333:6333 -v /home/quamer23nasim38/Hindi-Language-AI-Chatbot-for-Enterprises-using-Qdrant-MLFlow-and-LangChain/:/qdrant/storage qdrant/qdrant
pip install mlflow
pip install gradio

```