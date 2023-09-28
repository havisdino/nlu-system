import torch
import json
import jsonlines as jsl


def get_key_padding_mask(tensor, binary=False):
    mask = tensor == 0
    if binary:
        return mask
    
    mask = torch.where(mask, -float('inf'), 0.)
    mask[:, 0] = 0.
    return mask
    
    
def get_config(config_path):
    with open(config_path) as file:
        configs = json.load(file)
    return configs


def count_jsonl_lines(path):
    with jsl.open(path) as reader:
        n = 0
        for line in reader:
            n += 1
    return n


__config = get_config('configs/model_configs.json')
D_MODEL = __config['d_model']
WARMUP_STEP = __config['warmup_step']

def lr_schedule(step):
    step += 1
    return D_MODEL ** (-0.5) * min(step ** (-0.5), step * WARMUP_STEP ** (-1.5))
