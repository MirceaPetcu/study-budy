import mlflow
import os
import torch
from huggingface_hub import login
from dotenv import load_dotenv
from generator_base import GeneratorBase
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex
from class_with_logger import BaseClassWithLogging


class Generator(GeneratorBase, BaseClassWithLogging):
    def __init__(self,
                 index: VectorStoreIndex = None,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 system_prompt: str = "You are a teacher explaining complex concepts to students with the "
                                      "help of Retrieval Augmented Generations pipeline. You will receive some "
                                      "context from where you will have to synthesize a well structured answer.",
                 max_new_tokens: int = 512,
                 quantization_config: any = None,
                 ):
        super().__init__(class_name="Generator")
        self.start_logging()
        load_dotenv()
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("system_prompt", system_prompt)
        mlflow.log_param("max_new_tokens", max_new_tokens)
        login(token=os.getenv("HF_READ_TOKEN"))
        # TODO: try with llama3.2 3b instruct and other inference optimization techniques
        generator_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                               torch_dtype=torch.bfloat16,
                                                               device_map='cuda',
                                                               use_cache=True,
                                                               quantization_config=quantization_config).eval()
        generator_model = generator_model.to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        llm = HuggingFaceLLM(
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            device_map="cuda:0",
            system_prompt=system_prompt,
            model=generator_model
        )

        self._query_engine = index.as_query_engine(llm=llm)
        mlflow.end_run()

    @staticmethod
    def post_process_response(response: str):
        return response

    def generate(self, prompt: str) -> str:
        raw_response = self._query_engine.query(prompt)
        processed_response = Generator.post_process_response(str(raw_response))
        return processed_response
