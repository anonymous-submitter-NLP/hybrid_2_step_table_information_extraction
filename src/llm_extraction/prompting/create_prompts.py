import os
import sys
from typing import List

from langchain_core.prompts import ChatPromptTemplate

sys.path.append(os.path.join(os.getcwd()))

from src.classes.Emission_types import Emission_type
from src import config

def _load_prompt_txt(file_path: str) -> str:
    with open(file_path, encoding="utf-8") as file:
        return file.read()
    
def create_prompt_extract_y(prompts_folder: str = None):

    # build final user_prompt template
    user_prompt = _load_prompt_txt(os.path.join(prompts_folder, "y_extraction", "user_prompt_y_extraction.txt"))
    
    # create and return prompt
    basic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _load_prompt_txt(os.path.join(prompts_folder, "y_extraction", "system_prompt_y_extraction.txt"))),
            ("user", user_prompt),
        ]
    )
    return basic_prompt
