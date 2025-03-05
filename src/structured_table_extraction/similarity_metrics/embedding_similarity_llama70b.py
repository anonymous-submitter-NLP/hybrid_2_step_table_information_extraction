from sentence_transformers import SentenceTransformer

from langchain_ollama import OllamaEmbeddings

from typing import List, Tuple
from functools import lru_cache
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from src.classes.Available_embedding_models import AvailableEmbeddingModels
from src.similarity_measures.cosine_similarity import cosine_similarity
from src.structured_table_extraction.structured_table_extraction_helper import cosine_similarity_to_0_1

from dotenv import load_dotenv

load_dotenv()

def calculate_similarity(query: str, cell: str, embedding_model_enum: AvailableEmbeddingModels = AvailableEmbeddingModels.LLAMA3_70B) -> float: 
    """Create a function to calculate the similarity between two strings. 
    
    Return a float value between 0 and 1.

    Args:
        query (str): Query, that is compared to the cell
        cell (str): Cell value
        n (int, optional): Length of string element. Defaults to 2.

    Returns:
        float: _description_
    """

    ollama_base_url = "http://" + os.getenv("OLLAMA_HOST") + ":" + os.getenv("OLLAMA_PORT")

    embedding_model = OllamaEmbeddings(
                model=embedding_model_enum.value, 
                # base url needs to be added here for an unknown reason
                base_url = ollama_base_url
                )
    
    @lru_cache(maxsize=None)
    def _my_encode(sentence: str) -> np.array:
        embedding_query = embedding_model.embed_query(sentence)
        return embedding_query

    embedding_query = _my_encode(query)
    embedding_cell = _my_encode(cell)

    similarities = cosine_similarity(embedding_query, embedding_cell)

    return cosine_similarity_to_0_1(similarities)


if __name__ == "__main__":
    query = "What is the Capital of France?"
    cell = "Paris is the Capital of France."
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")
    
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")

    cell = "Berlin is the Capital of Germany."
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")
    