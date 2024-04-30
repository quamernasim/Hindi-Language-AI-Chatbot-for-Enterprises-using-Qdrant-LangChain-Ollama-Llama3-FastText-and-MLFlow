from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fasttext as ft
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import argparse
from qdrant_client.models import Batch

def load_docs_in_langchain(data_path, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    loader = DirectoryLoader(data_path, loader_cls=TextLoader)
    docs = loader.load_and_split(text_splitter=text_splitter)

    return docs

def convert_docs_to_embeddings_df(docs, embed_model):
    data = []
    for doc in docs:
        row_data = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        data.append(row_data)

    df = pd.DataFrame(data)

    df['page_content'] = df['page_content'].replace('\\n', ' ', regex=True)

    df['id'] = range(1, len(df) + 1)
    df['payload'] = df[['page_content', 'metadata']].to_dict(orient='records')
    df['embeddings'] = df['page_content'].apply(lambda x: (embed_model.get_sentence_vector(x)).tolist())

    return df

def create_qdrant_collections(collection_name, host, port):
    client = QdrantClient(host=host, port=port)

    client.delete_collection(collection_name=collection_name)

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=300, 
            distance=Distance.COSINE
        )
    )
    return client

def index_docs_in_qdrant(client, df, collection_name, batch_size):
    client.upsert(
        collection_name=collection_name,
        points=Batch(
            ids=df['id'].to_list()[:batch_size],
            payloads=df['payload'][:batch_size],
            vectors=df['embeddings'].to_list()[:batch_size],
        ),
    )
    return client

def main(args):

    embed_model = ft.load_model(args.embedding_model_path)
    docs = load_docs_in_langchain(args.data_path, args.chunk_size, args.chunk_overlap)
    df = convert_docs_to_embeddings_df(docs, embed_model)
    client = create_qdrant_collections(args.collection_name, args.host, args.port)
    client = index_docs_in_qdrant(client, df, args.collection_name, args.batch_size)
    client.close()

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='Hindi-Aesthetics-Corpus/Corpus')
argparser.add_argument('--embedding_model_path', type=str, default='wiki.hi.bin')
argparser.add_argument('--chunk_size', type=int, default=500)
argparser.add_argument('--chunk_overlap', type=int, default=50)
argparser.add_argument('--batch_size', type=int, default=4000)
argparser.add_argument('--host', type=str, default='localhost')
argparser.add_argument('--port', type=int, default=6333)
argparser.add_argument('--collection_name', type=str, default='my_collection')

args = argparser.parse_args()

if __name__ == '__main__':
    main(args)