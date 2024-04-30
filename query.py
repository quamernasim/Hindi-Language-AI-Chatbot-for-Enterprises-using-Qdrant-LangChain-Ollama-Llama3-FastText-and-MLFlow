import mlflow
import argparse
import fasttext as ft
from typing import List
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.llms.ollama import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def setup_mlflow_logging(enabled: bool):
    if enabled:
        mlflow.set_experiment("Hindi AI Chatbot")
        mlflow.start_run()

def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)

def load_embedding_model(path: str):
    return ft.load_model(path)

class QdrantRetriever(BaseRetriever):
    client: QdrantClient
    embed_model: ft.FastText._FastText
    collection_name: str
    limit: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Converts query to a vector and retrieves relevant documents using Qdrant."""
        # client = QdrantClient(host="localhost", port=6333)
        query_vector = self.embed_model.get_sentence_vector(query).tolist()
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.limit
        )
        return [Document(page_content=hit.payload['page_content']) for hit in search_results]

def setup_chatbot_system(prompt_text: str):
    return ChatPromptTemplate.from_messages([("system", prompt_text), ("human", "{input}")])

def create_chat_chain(retriever, llm, prompt):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain


def main(args):
    setup_mlflow_logging(args.mlflow_logging)

    client = QdrantClient(host=args.host, port=args.port)

    embed_model = load_embedding_model(args.embed_model_path)

    retriever = QdrantRetriever(
        client=client, 
        embed_model=embed_model, 
        collection_name=args.collection_name, 
        limit=args.collection_limit
    )

    llm = Ollama(
        model=args.model_name, 
        num_predict=args.num_predict, 
        num_ctx=args.num_ctx, 
        num_gpu=args.num_gpu, 
        temperature=args.temperature, 
        top_k=args.top_k, 
        top_p=args.top_p
    )
    system_prompt = """<s>[INST] आप एक सम्मानीय सहायक हैं। आपका काम नीचे दिए गए संदर्भ से प्रश्नों का उत्तर देना है। आप केवल हिंदी भाषा में उत्तर दे सकते हैं। धन्यवाद। You are never ever going to generate response in English. You are always going to generate response in Hindi no matter what. You also need to keep your answer short and to the point. संदर्भ: {context} </s>"""
    
    prompt = setup_chatbot_system(system_prompt)
    
    chat_chain = create_chat_chain(retriever, llm, prompt)
    
    response = chat_chain.invoke({"input": args.query})

    if args.mlflow_logging:
        log_params({
            'qdrant_host': args.host,
            'qdrant_port': args.port,
            'embedding_model_path': args.embed_model_path,
            'collection_name': args.collection_name,
            'collection_limit': args.collection_limit,
            'model_name': args.model_name,
            'num_predict': args.num_predict,
            'num_ctx': args.num_ctx,
            'num_gpu': args.num_gpu,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'system_prompt': system_prompt, 
            'query': args.query, 
            'context': response['context'],
            'response': response['answer']
        })

        mlflow.end_run()
    
    print(f"Question: {response['input']}")
    print(f"Context: {response['context']}")
    print(f"Answer: {response['answer']}")

argparser = argparse.ArgumentParser()
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

args = argparser.parse_args()

if __name__ == "__main__":
    main(args)

