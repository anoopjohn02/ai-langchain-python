import chromadb
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.HttpClient(host='localhost', port=8000) 

default_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

collection = chroma_client.get_collection(name="my-collection", embedding_function=default_ef)

results = collection.query(
    query_texts=["What is spice"],
    n_results=2,
    include=["metadatas"],
    where={"userId": "teena"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

print(results)