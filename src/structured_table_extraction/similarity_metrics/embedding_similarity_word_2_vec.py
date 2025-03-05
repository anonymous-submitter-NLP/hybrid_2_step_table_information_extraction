from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as api

from typing import List, Tuple
from functools import lru_cache
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from src.similarity_measures.cosine_similarity import cosine_similarity
from src.classes.Available_embedding_models import AvailableEmbeddingModels
from src.structured_table_extraction.structured_table_extraction_helper import cosine_similarity_to_0_1
import re

@lru_cache(None)
def _get_model():
    return api.load("glove-wiki-gigaword-50")

def calculate_similarity(query: str, cell: str) -> float: 
    """Create a function to calculate the similarity between two strings. 
    
    Return a float value between 0 and 1.

    #https://stackoverflow.com/questions/65852710/text-similarity-using-word2vec

    Args:
        query (str): Query, that is compared to the cell
        cell (str): Cell value

    Returns:
        float: Similarity score between 0 and 1
    """

    model = _get_model()

    def preprocess(s):
        s = re.sub(r'[^a-zA-Z0-9\s]', '', s)

        s = re.sub(r'(\d+)', r' \1 ', s)

        return [i.lower() for i in s.split()]

    def get_vector(s):
        
        vectors = []
        for i in preprocess(s):
            try: 
                vectors.append(model[i])
            except KeyError: 
                pass
            
        return np.sum(np.array(vectors), axis=0)

    vector_query = get_vector(query)
    vector_cell = get_vector(cell)

    if vector_cell is not None and vector_query is not None and np.any(vector_cell) and np.any(vector_query):
        return cosine_similarity_to_0_1(cosine_similarity(vector_cell, vector_query))
    else: 
        return 0


if __name__ == "__main__":
    cell = "Scope 1 Emission"
    query = "3,213"
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")
    
    cell = "Scope 2 Emission"
    query = "Scope 1"
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")
