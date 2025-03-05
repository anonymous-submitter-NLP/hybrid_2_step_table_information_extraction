from typing import List

import ast
import json
from src import config
from src.formatting import formatting
from src.classes.Available_llms import AvailableLLMs
from src.llm_extraction import call_llm
from src.llm_extraction.prompting import create_prompts
from src.helper import join_chunks_to_str
from langchain_core.prompts import ChatPromptTemplate
from src.classes.Emission_types import Emission_type

def basic_rag_extraction(chunks: List[dict], year: int, report_name: str, emission_type: Emission_type, llm: AvailableLLMs):
    """
    Run basic rag extraction from chunks and return 

    - llm output: 
    - llm input: formatted prompt template

    """ 
    
    text = join_chunks_to_str(chunks)

    prompt_template = create_prompts.create_prompt_extract_y(prompts_folder="src/llm_extraction/prompting/benchmark_for_prompts_select_from_preselection_2/prompt_extract_y.txt")
    
    prompt_input = {"text": text,
                    "year": year, 
                    "report_name": report_name}

    llm_output = call_llm.call_llm(llm, prompt_template, prompt_input)

    return llm_output, prompt_template.format(**prompt_input)


def extraction_choosing(chunks: List[dict], year: int, report_name: str, emission_type: Emission_type, llm: AvailableLLMs, preselection: str):
    """
    Run basic rag extraction from chunks and return 

    - llm output: 
    - llm input: formatted prompt template

    """ 
    
    text = join_chunks_to_str(chunks)

    user_prompt = create_prompts._load_prompt_txt("src/llm_extraction/prompting/prompts_select_from_preselection_2/prompt_extract_y.txt")

    preselection_list = ast.literal_eval(preselection)

    preselection = [{"Scope 3": element} for i, element in enumerate(preselection_list)]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that extracts information in a robsut way, so return multiple in case you are not certain. Do not use prior knowledge."),
            ("user", user_prompt),
        ]
    )

    prompt_input = {"text": text,
                    "year": year, 
                    "report_name": report_name, 
                    "emission_type": emission_type,
                    "preselection": preselection}

    llm_output = call_llm.call_llm(llm, prompt_template, prompt_input)
    
    return llm_output, prompt_template.format(**prompt_input)

