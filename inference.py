from utils import get_config, get_key_padding_mask
from model import load_model_from_checkpoint, Transformer, SARER
import torch
from argparse import ArgumentParser
from sentencepiece import SentencePieceProcessor
import jsonlines as jsl
import json
import os
from time import time


def recognize_entities(sentence_ids: list, labels: list, entity_configs: dict, tokenizer):    
    id_to_type = {id: type for type, id in entity_configs.items()}
    neutral = entity_configs['neutral']
    entities = []
    i = 0
    j = -1
    while i < len(labels):
        if labels[i] != neutral:
            try:
                next_label = labels[i + 1]
            except IndexError:
                next_label = -1
                
            if labels[i] != next_label:
                filler_ids = sentence_ids[j + 1:i + 1]
                filler = tokenizer.decode(filler_ids)
                type = id_to_type[labels[i]]
                entities.append(dict(type=type, filler=filler))
            else:
                i += 1
                continue
        j = i
        i += 1
    return entities
            

def infer(
    intent_model: Transformer, entity_model: SARER,
    tokenizer: SentencePieceProcessor, sentence, maxlen, device,
    entity_configs, audio_file_name=None, write_single_result=True):
    
    intent_model.eval()
    entity_model.eval()
    
    sentence_ids = tokenizer.encode(sentence)[:maxlen]
    sentence_ids = torch.tensor(sentence_ids).unsqueeze(0).to(device)
    
    encoder_output = intent_model(sentence_ids, None, encoder_output_only=True)
    
    # Entity recognition
    predicted_labels = [0]
    while len(predicted_labels) <= 1 + sentence_ids.size(-1):
        target_ids = torch.tensor([predicted_labels]).to(device)
        
        logits = entity_model(
            input=encoder_output,
            key_padding_mask=None,
            tgt_ids=target_ids,
            tgt_key_padding_mask=None
        )[0, -1, :]
        predicted_label = logits.argmax(dim=-1).cpu().item()
        predicted_labels.append(predicted_label)
        
    entities = recognize_entities(
        sentence_ids.squeeze().tolist(), predicted_labels[1:], entity_configs, tokenizer)
    
    # Intent prediction
    predicted_ids = [tokenizer.bos_id()]
    while len(predicted_ids) <= maxlen:
        target_ids = torch.tensor([predicted_ids]).to(device)
        output = intent_model(sentence_ids, target_ids, inference=True, encoder_output=encoder_output)
        
        logits = output['decoder_output'][0, -1, :]
        predicted_id = logits.argmax(dim=-1).cpu().item()
        if predicted_id == tokenizer.eos_id():
            break
        predicted_ids.append(predicted_id)
        
    intent = tokenizer.decode(predicted_ids)
    
    try:
        os.mkdir('results')
    except FileExistsError:
        pass
    
    result = dict(intent=intent, entities=entities)
    if audio_file_name is not None:
        result.update(file=audio_file_name)

    if write_single_result:
        with jsl.open(f'results/{time()}.jsonl', 'w') as file:
            file.write(result)

    return result
        

if __name__ == '__main__':
    from tqdm.autonotebook import tqdm
    from utils import count_jsonl_lines
    
    parser = ArgumentParser()
    parser.add_argument('-s', '--sentence', type=str, default=None)
    parser.add_argument('-j', '--jsonl', type=str, default=None)
    parser.add_argument('-mi', '--intent-model-path', type=str, required=True)
    parser.add_argument('-me', '--entity-model-path', type=str, required=True)
    parser.add_argument('-dv', '--device', type=str, default=None)
    parser.add_argument('-c', '--config-path', type=str, default='configs/model_configs.json')
    parser.add_argument('-t', '--tokenizer-path', type=str, default='tokenizer/tok.model')
    parser.add_argument('-ec', '--entity-config-path', type=str, default='configs/entity_types.json')
    
    args = parser.parse_args() 
    assert (args.sentence is not None) ^ (args.jsonl is not None)
    
    tokenizer = SentencePieceProcessor(args.tokenizer_path)
    
    if args.device in ['cuda', 'cpu', 'mps']:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    intent_model = load_model_from_checkpoint(args.intent_model_path, device, args.config_path)
    intent_model.to(device)
    intent_model.eval()
    
    entity_model = load_model_from_checkpoint(
        args.entity_model_path, device, args.config_path,
        model_type='entity', entity_config_path=args.entity_config_path
    )
    entity_model.to(device)
    entity_model.eval()
    
    with open(args.config_path) as file:
        maxlen = json.load(file)['max_sequence_length']
        
    entity_configs = get_config(args.entity_config_path)
    
    if args.jsonl is not None:
        results = []
        
        with jsl.open(args.jsonl) as ds:
            for sample in tqdm(ds, total=count_jsonl_lines(args.jsonl)):
                sentence = sample['sentence'].lower()
                file = sample['file']
                
                results.append(
                    infer(
                        intent_model=intent_model,
                        entity_model=entity_model,
                        tokenizer=tokenizer,
                        sentence=sentence,
                        maxlen=maxlen,
                        device=device,
                        entity_configs=entity_configs,
                        audio_file_name=file,
                        write_single_result=False
                    )
                )
        
        with jsl.open(f'results/all_{time()}.jsonl', 'w') as f:
            f.write_all(results)
    
    else:
        result = infer(
            intent_model=intent_model,
            entity_model=entity_model,
            tokenizer= tokenizer,
            sentence=args.sentence.lower(),
            maxlen=maxlen,
            device=device,
            entity_configs=entity_configs
        )
    