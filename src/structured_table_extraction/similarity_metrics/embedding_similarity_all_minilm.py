from sentence_transformers import SentenceTransformer

from typing import List, Tuple
from functools import lru_cache
import numpy as np
import os
import sys
import time

sys.path.append(os.getcwd())

from src.classes.Available_embedding_models import AvailableEmbeddingModels

def calculate_similarity(query: str, cell: str, embedding_model_enum: AvailableEmbeddingModels = AvailableEmbeddingModels.ALL_MINILM_L6_V2) -> float: 
    """Create a function to calculate the similarity between two strings. 
    
    Return a float value between 0 and 1.

    Args:
        query (str): Query, that is compared to the cell
        cell (str): Cell value
        n (int, optional): Length of string element. Defaults to 2.

    Returns:
        float: _description_
    """

    model = _get_model(embedding_model_enum)

    query_embedding = _my_encode((query,), model)

    cell_embeddings = _my_encode((cell,), model)

    similarities = model.similarity(query_embedding, cell_embeddings)

    return similarities[0][0]


@lru_cache(maxsize=None)
def _my_encode(sentences: Tuple[str], model: SentenceTransformer) -> np.array:
    return model.encode(list(sentences))

@lru_cache(maxsize=3)
def _get_model(embedding_model_enum: AvailableEmbeddingModels) -> SentenceTransformer: 
    
    model = SentenceTransformer(embedding_model_enum.value)

    return model 


if __name__ == "__main__":
    query = "What is the Capital of France?"
    cell = "Paris is the Capital of France."
    start_time = time.time()
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")
    print(f"Time taken: {time.time() - start_time} seconds")

    start_time = time.time()
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")
    print(f"Time taken: {time.time() - start_time} seconds")

    cell = "Berlin is the Capital of Germany."
    start_time = time.time()
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")
    print(f"Time taken: {time.time() - start_time} seconds")