import logging

logger = logging.getLogger(__name__)

def build_vl_icl_prompt(task, demonstrations, query, mode="constrained", warned=False, debug=False, max_images=20, task_type=None):
    if hasattr(task, 'get_task_instruction'):
        task_instruction_params = task.get_task_instruction.__code__.co_varnames
        kwargs = {}
        if 'mode' in task_instruction_params:
            kwargs['mode'] = mode
        if 'warned' in task_instruction_params:
            kwargs['warned'] = warned
        if 'task_type' in task_instruction_params and task_type is not None:
            kwargs['task_type'] = task_type
        instruction = task.get_task_instruction(**kwargs)
    else:
        instruction = task.get_task_instruction()
    
    prompt_parts = [instruction]
    
    if demonstrations:
        support_parts = []
        for demo in demonstrations:
            if hasattr(task, 'format_demonstration') and 'mode' in task.format_demonstration.__code__.co_varnames:
                demo_text = task.format_demonstration(demo, include_image_token=True, mode=mode)
            else:
                demo_text = task.format_demonstration(demo, include_image_token=True)
            support_parts.append(demo_text)
        
        support_set = "\n\n".join(support_parts) if mode == "free" else "\n".join(support_parts)
        prompt_parts.append(f"Support Set:\n{support_set}")
    
    if hasattr(task, '_build_conversational_prompt'):
        conversational_prompt = task._build_conversational_prompt(demonstrations, query, mode)
        prompt_parts = [instruction, conversational_prompt]
        full_prompt = "\n\n".join(prompt_parts)
    else:
        if hasattr(task, 'format_query'):
            format_query_params = task.format_query.__code__.co_varnames
            query_kwargs = {'include_image_token': True}
            if 'mode' in format_query_params:
                query_kwargs['mode'] = mode
            if 'task_type' in format_query_params and task_type is not None:
                query_kwargs['task_type'] = task_type
            query_text = task.format_query(query, **query_kwargs)
        else:
            query_text = task.format_query(query, include_image_token=True)
        prompt_parts.append(f"Question:\n{query_text}")
        
        if mode == "free":
            prompt_parts.append("Answer:")
        else:
            prompt_parts.append("Answer:")
        
        full_prompt = "\n\n".join(prompt_parts)
    
    if debug:
        logger.info(f"Built VL-ICL prompt ({mode} mode):\n{full_prompt}")
    
    return full_prompt

def collect_images_for_prompt(task, demonstrations, query, debug=False, max_images=20):
    images = []
    
    for demo in demonstrations:
        if 'image' in demo:
            if isinstance(demo['image'], list):
                for img_path in demo['image']:
                    images.append(task.load_image(img_path))
            else:
                images.append(task.load_image(demo['image']))
    
    if 'image' in query:
        if isinstance(query['image'], list):
            for img_path in query['image']:
                images.append(task.load_image(img_path))
        else:
            images.append(task.load_image(query['image']))
    
    if debug:
        logger.info(f"Collected {len(images)} images total")
    
    return images
