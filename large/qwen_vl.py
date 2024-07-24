from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import time
def init_qwenvl(device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True, cache_dir='./cache').eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    return model, tokenizer
def inference_qwenvl(model, tokenizer, image_file, detection_prompt='get the location of the handclap'):
    # 1st dialogue turn
    t_start = time.time()
    query = tokenizer.from_list_format([
        # {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
        # {'text': '这是什么?'},
        {'image': image_file},
        {'text': 'Describe the image.'},
    ])    
    response, history = model.chat(tokenizer, query=query, history=None)
    if detection_prompt is not None:
        response, history = model.chat(tokenizer, detection_prompt, history=history)
    print(f"Time elapsed: {time.time() - t_start:.2f}s")
    return response