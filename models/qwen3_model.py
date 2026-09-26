import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseVLLM
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

class Qwen3Model(BaseVLLM):
    def __init__(self, model_name="Qwen/Qwen3-8B", **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        logger.info(f"Loading Qwen3 model: {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = 1.0  
            self.model.generation_config.top_p = 1.0  
            self.model.generation_config.top_k = 50
        
        logger.info(f"Model loaded: {self.model_name}")
        self._log_gpu_memory_usage()
    
    def _log_gpu_memory_usage(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB allocated")
    
    def generate_text(self, images: List = None, prompt: str = "", max_new_tokens: int = 50, debug: bool = False) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        try:
            with torch.no_grad():
                valid_params = {
                    'max_new_tokens': generation_config['max_new_tokens'],
                    'do_sample': False,
                    'pad_token_id': generation_config['pad_token_id'],
                    'eos_token_id': generation_config['eos_token_id'],
                    'num_return_sequences': 1,
                    'return_dict_in_generate': False
                }
                
                if debug:
                    logger.info(f"[Qwen3] Generation params: {valid_params}")
                
                generated_ids = self.model.generate(
                    **model_inputs,
                    **valid_params
                )
            
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            if debug:
                logger.info(f"[Qwen3] Generated response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Qwen3 generation: {e}")
            return ""
    
    def generate_image(self, prompt: str, context_images = None, **kwargs):
        raise NotImplementedError("Image generation not supported for pure LLM models")
    
    def load_image(self, image_path: str, max_num: int = 6):
        return None