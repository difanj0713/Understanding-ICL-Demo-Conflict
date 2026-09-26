from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import torch
from PIL import Image

class BaseVLLM(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate_text(self, images: List[Image.Image], prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def generate_image(self, prompt: str, context_images: List[Image.Image] = None, **kwargs) -> Image.Image:
        pass