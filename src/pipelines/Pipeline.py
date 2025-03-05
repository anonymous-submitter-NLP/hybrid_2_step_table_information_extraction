import os
import sys
import json
import dotenv
from typing import Callable
import numpy as np
from langchain.text_splitter import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter

sys.path.append(os.path.join(os.getcwd()))

from src import config
from src.pipelines import config_enums
from src.similarity_measures import cosine_similarity, euclidean_similarity

from src.classes import Company, Emission_types
from src.classes.Available_llms import AvailableLLMs
from src.classes.Available_embedding_models import AvailableEmbeddingModels

# basic formatting functions: 
from src.formatting import formatting
from src.formatting import text_splitting

# import the modules of pdt_to_text
from src.pdf_to_text import pypdf_extraction,  llama_parse_wrapper, pdf_to_tables
#from src.pdf_to_text import pdf_to_tables

# import the modules for filtering
from src.filter import filter_chunks

# import the modules for validation
from src.pipelines import validation_helper 

from src.pipelines import data_extraction_helper, chunk_extraction_helper

class Pipeline():

    def __init__(self, 
                 extract: config_enums.LLM_Extract, 
                 ) -> None:
        self.extract = extract

    def run_pipeline(self, pdf_path: str, emission_type: Emission_types.Emission_type, year: int, company: Company.Company, llm: AvailableLLMs, text: str = None, preselection: str = None) -> dict: 
        
        prior_x=company.get_report_of_year(year-1).get_emission(emission_type).chunk
        prior_y={"quantity": company.get_report_of_year(year-1).get_emission(emission_type).quantity, 
                 "unit": company.get_report_of_year(year-1).get_emission(emission_type).unit}
                                                
        result = self.information_extraction(pdf_path=pdf_path, 
                                             emission_type=emission_type, 
                                             year=year,
                                             prior_y=prior_y, 
                                             prior_x=prior_x, 
                                             llm=llm,
                                             text=text, 
                                             )
        
        result["company"] = company
        
        return result 


    def information_extraction(self, 
                               pdf_path: str, 
                               emission_type: Emission_types.Emission_type, 
                               year: int, 
                               prior_y: dict, 
                               llm: AvailableLLMs, 
                               text: str=None, 
                               preselection: str = None) -> dict: 
        
        # setup run log
        run_log = dict()
        run_log["llm"] = llm.value

        report_name = os.path.basename(pdf_path)

        chunks = [{'page': 1, 'text': text, 'type': 'text'}]

        match self.extract: 
            case config_enums.LLM_Extract.BASIC: 

                llm_output, llm_input = data_extraction_helper.basic_rag_extraction(chunks=chunks,
                                                                               year=year, 
                                                                               report_name=report_name, 
                                                                               emission_type=emission_type, 
                                                                               llm=llm
                                                                               )
                
                run_log['5_extraction'] = {'llm_output_from_extraction': llm_output.content, 
                                            'llm_input_from_extraction': llm_input}

            case config_enums.LLM_Extract.CHOOSE_FROM_PRESELECTION: 

                llm_output, llm_input = data_extraction_helper.extraction_choosing(chunks=chunks,
                                                                               year=year, 
                                                                               report_name=report_name, 
                                                                               emission_type=emission_type.value, 
                                                                               llm=llm, 
                                                                               preselection=preselection,
                                                                               prior_y=prior_y
                                                                               )
                
                run_log['5_extraction'] = {'llm_output_from_extraction': llm_output.content, 
                                            'llm_input_from_extraction': llm_input}

        formatted_result = formatting.llm_output_string_formatting(llm_output.content)

        return {"config": self.get_configs(),
                "result": formatted_result,
                "run_log": run_log, 
                "emission_type": emission_type}

    def get_configs(self):
        return self.__dict__