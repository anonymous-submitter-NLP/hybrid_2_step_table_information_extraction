from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate

from typing import List, Tuple
from functools import lru_cache
import numpy as np
import os
import sys
import json

sys.path.append(os.getcwd())

from src.classes.Available_embedding_models import AvailableEmbeddingModels
from src.structured_table_extraction.similarity_metrics import regular_expression_complete_word_matching
from src.llm_extraction.call_llm import call_llm
from src.classes.Available_llms import AvailableLLMs

def calculate_similarity(query: str, cell: str, embedding_model_enum: AvailableEmbeddingModels = AvailableEmbeddingModels.ALL_MP_NET_BASE_V2) -> float: 
    """Create a function to calculate the similarity between two strings. 
    
    Return a float value between 0 and 1.

    Args:
        query (str): Query, that is compared to the cell
        cell (str): Cell value
        n (int, optional): Length of string element. Defaults to 2.

    Returns:
        float: _description_
    """

    # prefilter with regex

    if regular_expression_complete_word_matching.calculate_similarity(query, cell) == 0:
        return 0

    # Define the prompt template and input
    prompt_template = ChatPromptTemplate.from_template("Is the following text a description for {query} Emission? Text: '''{cell}'''. Answer with true or false in a json object: {{\"answer\": true}} or {{\"answer\": false}}")
    prompt_input = {"query": query, "cell": cell}

    # Call the LLM
    response = call_llm(AvailableLLMs.LLAMA3_8B, prompt_template, prompt_input)
    
    print(prompt_template.format(query=query, cell=cell))
    print(response)
    print()

    response_json = json.loads(response.content)
    print(response_json)

    if response_json["answer"] == True: 
        return 1
    else:
        return 0