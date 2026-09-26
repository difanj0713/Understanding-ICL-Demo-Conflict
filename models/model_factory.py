def create_model(model_type: str, model_name: str, **kwargs):
    if model_type == 'internvl':
        from .internvl_model import InternVLModel
        return InternVLModel(model_name, **kwargs)
    elif model_type == 'qwen25':
        from .qwen25_model import Qwen25Model
        return Qwen25Model(model_name, **kwargs) 
    elif model_type == 'qwen3':
        from .qwen3_model import Qwen3Model
        return Qwen3Model(model_name, **kwargs)   
    elif model_type == 'llama3':
        from .llama3_model import Llama3Model
        return Llama3Model(model_name, **kwargs)  
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: internvl, qwen25, qwen3, llama3")