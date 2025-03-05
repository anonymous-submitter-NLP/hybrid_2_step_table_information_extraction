from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Tuple
from functools import lru_cache
import numpy as np
import os
import sys
import time

sys.path.append(os.getcwd())

from src.structured_table_extraction.structured_table_extraction_helper import cosine_similarity_to_0_1
from src.classes.Available_embedding_models import AvailableEmbeddingModels

@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    """
    Load FinBERT model and tokenizer with caching.
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def calculate_similarity(query: str, cell: str, embedding_model_enum: AvailableEmbeddingModels = AvailableEmbeddingModels.FINBERT) -> float: 
    """Create a function to calculate the similarity between two strings. 
    
    Return a float value between 0 and 1.

    Args:
        query (str): Query, that is compared to the cell
        cell (str): Cell value
        n (int, optional): Length of string element. Defaults to 2.

    Returns:
        float: _description_
    """
    # Load FinBERT model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    def get_embedding(text):
        """
        Generate an embedding from FinBERT for a given text.
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract hidden state (we use logits as a rough embedding)
        embedding = outputs.logits.squeeze().numpy()
        
        return embedding

    def calculate_cosine_similarity(text1, text2):
        """
        Compute the cosine similarity between the FinBERT embeddings of two texts.
        """
        emb1 = get_embedding(text1).reshape(1, -1)
        emb2 = get_embedding(text2).reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity

    similarity_score = calculate_cosine_similarity(query, cell)
    return cosine_similarity_to_0_1(similarity_score)


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



    # Load FinBERT model and tokenizer
    tokenizer, model = load_model_and_tokenizer()
    def get_embedding(text):
        """
        Generate an embedding from FinBERT for a given text.
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)        
        with torch.no_grad():
            outputs = model(**inputs)        

        # Extract hidden state (we use logits as a rough embedding)
        embedding = outputs.logits.squeeze().numpy()        
        return embedding

    query = "What is the Capital of France?"
    cell = "Paris is the Capital of France."
    embedding_query = get_embedding(query)
    embedding_cell = get_embedding(cell)

    print(embedding_query)
    print(embedding_cell)
    print(embedding_query.shape)
    print(embedding_cell.shape)

    start_time = time.time()
    similarity = calculate_similarity(query, cell)
    print(f"Similarity: {similarity}")
    print(f"Time taken: {time.time() - start_time} seconds")

    # start_time = time.time()
    # similarity = calculate_similarity(query, cell)
    # print(f"Similarity: {similarity}")
    # print(f"Time taken: {time.time() - start_time} seconds")

    # cell = "Berlin is the Capital of Germany."
    # start_time = time.time()
    # similarity = calculate_similarity(query, cell)
    # print(f"Similarity: {similarity}")
    # print(f"Time taken: {time.time() - start_time} seconds")