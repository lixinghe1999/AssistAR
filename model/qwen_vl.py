from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import time
def init_qwenvl(device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True, cache_dir='./cache').eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    return (model, tokenizer)
def inference_qwenvl(model, image_file, prompt='get the location of the handclap'):
    model, tokenizer = model
    # 1st dialogue turn
    t_start = time.time()
    query = tokenizer.from_list_format([
        # {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
        # {'text': '这是什么?'},
        {'image': image_file},
        {'text': 'Describe the image.'},
    ])    
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    if prompt is not None:
        response, history = model.chat(tokenizer, 'Locate the {} in the image'.format(prompt), history=history)
    print(f"Time elapsed: {time.time() - t_start:.2f}s")
    return response

def parser_qwenvl(response):
    '''
    Unified format:
    ref: [string] - class name
    box: [x1, y1, x2, y2] in [0, 1]
    '''
    if '<ref>' not in response:
        box = []; ref = []
    else:
        ref = response.split('<ref>')[1].split('</ref>')[0]
        box = response.split('<box>')[1].split('</box>')[0]
        box = box.replace('(', '').replace(')', '').split(',')
        box = [int(x) for x in box]
        box = [x/1000 for x in box]

    return ref, box