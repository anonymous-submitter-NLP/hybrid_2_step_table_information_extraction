from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from enum import Enum
import os
import sys
sys.path.append(os.path.join(os.getcwd()))

from src.classes.Available_llms import AvailableLLMs
from langchain_core.prompts import ChatPromptTemplate

def call_llm(llm_enum: AvailableLLMs, prompt_template: ChatPromptTemplate, prompt_input: dict):

    llm = get_llm(llm_enum=llm_enum)

    # Create the LLM chain
    llm_chain = prompt_template | llm

    # Run the LLM chain with user input
    response = llm_chain.invoke(prompt_input)

    return response


def get_llm(llm_enum: AvailableLLMs) -> BaseChatModel:
    
    # if AvailableLLMs.GPT4 or AvailableLLMs.GPT4OMINI:
    #     max_token_len = 8_192
    # elif llm == AvailableLLMs.GPT35TURBO:
    #     max_token_len = 16_385
    # elif llm == 'llama2':
    #     max_token_len = 4_096
    # elif llm == AvailableLLMs.LLAMA3_70B:
    #     max_token_len = 8_192


    # Initialize the LLM based on the enum selection using match-case
    match llm_enum:
        case AvailableLLMs.GPT4:
            llm = ChatOpenAI(model=llm_enum.value)
        case AvailableLLMs.GPT4OMINI:
            llm = ChatOpenAI(model=llm_enum.value)
        case AvailableLLMs.GPT4O:
            llm = ChatOpenAI(model=llm_enum.value)
        case AvailableLLMs.GPT35TURBO:
            llm = ChatOpenAI(model=llm_enum.value,
                             model_kwargs={"response_format": {"type": "json_object"}})
        case AvailableLLMs.LLAMA3_8B:
            llm = ChatOllama(
                model=llm_enum.value,
                temperature=0,
                format="json"
            )
        case AvailableLLMs.LLAMA3_70B:
            llm = ChatOllama(
                model=llm_enum.value,
                temperature=0,
                format="json"
            )
        case AvailableLLMs.DEEPSEEK_R1_70:
            llm = ChatOllama(
                model=llm_enum.value,
                temperature=0,
                format="json"
            )
        case AvailableLLMs.TABLE_LLAMA3_8B:
            
            model_id = "osunlp/TableLlama"

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_4bit=False,
                    device_map="auto",  # Automatically assigns layers to available GPUs
                #attn_implementation="flash_attention_2", # if you have an ampere GPU
            )
            
            pipe = pipeline("text-generation", model=model_id, tokenizer=tokenizer, max_new_tokens=100, top_k=50, temperature=0.1)
            llm = HuggingFacePipeline(pipeline=pipe)

        case _:
            raise ValueError("Unsupported LLM selected.")
        
    return llm

# Example usage
if __name__ == "__main__":

    # # Define the prompt template and input
    # prompt_template = ChatPromptTemplate.from_template("Translate the following English text to French: {text}")
    # prompt_input = {"text": "Hello, how are you?"}

    # # Call the LLM
    # response = call_llm(AvailableLLMs.LLAMA3_8B, prompt_template, prompt_input)
    # print(response)

    os.environ["OLLAMA_HOST"] = "10.80.20.127"
    os.environ["OLLAMA_PORT"] = "11434"

    llm = ChatOllama(
        model="deepseek-r1:70b",
        temperature=0, 
        format="json"
    )

    prompt_template = ChatPromptTemplate(("User", "Write an essay to describe the importance of bench pressing. Please use json to describe that!"))
    llm_chain = prompt_template | llm
    response = llm_chain.invoke({})
    print(response)