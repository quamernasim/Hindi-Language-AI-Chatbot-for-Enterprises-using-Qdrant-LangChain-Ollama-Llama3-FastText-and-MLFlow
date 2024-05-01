# Hindi-Language-AI-Chatbot-for-Enterprises-using-Qdrant-MLFlow-and-LangChain
RAG powered AI chatbot for Indian Language (Hindi) using LangChain, Ollama, Qdrant, and MLFlow

<p align="center">
  <img src="assets/hindi-chatbot.jpeg" width=40%/>
</p>

## Introduction
This repository contains the source code for a Hindi Language AI Chatbot for Enterprises using Qdrant, MLFlow, and LangChain. The notebooks contains step-by-step instructions to create the chatbot, and it can be run in any Python environment that supports Jupyter Notebook.

This whole project is divided into 2 parts:
1. Indexing the data using Qdrant FastText, and LangChain
2. Building the chatbot using Llama-3, Ollama, FastText, LangChain, Qdrant, MLFlow, and Gradio

## Part 1: Indexing the data using Qdrant FastText, and LangChain
In the first part, I will index the data using Qdrant FastText, and LangChain. I will use the Hindi Aesthetics Corpus dataset for this purpose. The dataset contains 1000 Hindi text files, and I will index them using Qdrant and FastText.

The notebook for this part: `notebooks/indexing.ipynb`

Alternatively, I have also attached the python script for this part: `indexing.py`

### Installation
To install the Qdrant client, use the following command:
```
sudo apt-get update
sudo apt install docker
docker pull qdrant/qdrant
docker run -p 6333:6333 -v /home/quamer23nasim38/Hindi-Language-AI-Chatbot-for-Enterprises-using-Qdrant-MLFlow-and-LangChain/:/qdrant/storage qdrant/qdrant
```

To download the FastText model, use the following command:
```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hi.zip
unzip wiki.hi.zip
```

To download the data, clone the repository using the following command:
```
git clone https://github.com/gayatrivenugopal/Hindi-Aesthetics-Corpus.git
```

### Usage
To run the python script, use the following command:
```
# change the arguments as per your requirements
python indexing.py --data_path Hindi-Aesthetics-Corpus/Corpus --embedding_model_path 'wiki.hi.bin' --chunk_size 500 --chunk_overlap 50 --batch_size 4000 --host 'localhost' --port 6333 --collection_name 'my_collection'
```

This will index the data using Qdrant, FastText, and LangChain.

## Part 2: Building the chatbot using Llama-3, Ollama, FastText, LangChain, Qdrant, MLFlow, and Gradio
In the second part, I load the indexed data using Qdrant, embed queries using FastText, use Llama-3 as the language model, and build the chatbot using Ollama integrated with LangChain. I will also use MLFlow to track the parameters of the chatbot, and Gradio to create the user interface.

The notebook for this part: `notebooks/query.ipynb`

Alternatively, I have also attached the python script for this part: `query.py`

### Installation
To install the Olama and Llama-3 models, use the following command:
```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

If you want to test the Ollaama model, use the following command:
```
ollama list
ollama run llama3
```

argparser.add_argument("--mlflow_logging", type=bool, default=False)
argparser.add_argument("--host", type=str, default='localhost')
argparser.add_argument("--port", type=int, default=6333)
argparser.add_argument("--embed_model_path", type=str, default='wiki.hi.bin')
argparser.add_argument("--collection_name", type=str, default='my_collection')
argparser.add_argument("--collection_limit", type=int, default=1)
argparser.add_argument("--model_name", type=str, default='llama3')
argparser.add_argument("--num_predict", type=int, default=100)
argparser.add_argument("--num_ctx", type=int, default=3000)
argparser.add_argument("--num_gpu", type=int, default=2)
argparser.add_argument("--temperature", type=float, default=0.7)
argparser.add_argument("--top_k", type=int, default=50)
argparser.add_argument("--top_p", type=float, default=0.95)
argparser.add_argument("--query", type=str, default='किस तरह के किरदार और कहानी तत्व रचनाकारों और फिल्म निर्माताओं को आकर्षित करते हैं?')

### Usage
To run the python script, use the following command:
```
# change the arguments as per your requirements
python query.py --mlflow_logging True --host 'localhost' --port 6333 --embed_model_path 'wiki.hi.bin' --collection_name 'my_collection' --collection_limit 1 --model_name 'llama3' --num_predict 100 --num_ctx 3000 --num_gpu 2 --temperature 0.7 --top_k 50 --top_p 0.95 --query 'किस तरह के किरदार और कहानी तत्व रचनाकारों और फिल्म निर्माताओं को आकर्षित करते हैं?'
```

This will build the chatbot using Llama-3, Ollama, FastText, LangChain, Qdrant, MLFlow, and Gradio.

## Results
Here in the below image, you can see the results of the chatbot. The chatbot is able to provide the answers to the queries in Hindi language.

<p align="center">
  <img src="assets/output hindi.png" width=100%/>
</p>


## Requirements
```
Python==3.10.13
ollama==0.1.32
docker==20.10.17
```

To install the required packages, use the following command:
```
pip install -r requirements.txt
```

## Conclusion
This repository contains the source code for a Hindi Language AI Chatbot for Enterprises using Qdrant, MLFlow, and LangChain. The notebooks contains step-by-step instructions to create the chatbot, and it can be run in any Python environment that supports Jupyter Notebook. The chatbot can be used by enterprises to provide customer support in Hindi language.

## Related Blog
Please reed the related blog for in-depth undertsnading of each step.
[Hindi-Language AI Chatbot for Enterprises Using Qdrant, MLFlow, and LangChain](https://github.com/quamernasim/Hindi-Language-AI-Chatbot-for-Enterprises-using-Qdrant-LangChain-Ollama-Llama3-FastText-and-MLFlow)
