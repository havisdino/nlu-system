import train_entity
import train_intent
from tqdm.autonotebook import tqdm
import os


def validate_single_task(data_loader, device, entity_model=None, intent_model=None):
    assert intent_model is not None
    intent_model.eval()
    
    mode = 'intent'
    if entity_model is not None:
        entity_model.eval()
        mode = 'entity'
        
    loss = 0
    N = len(data_loader)
    for sample in tqdm(data_loader):
        src_ids = data_loader['sentence_ids'].to(device)
        if mode == 'entity':
            labels = data_loader['labels'].to(device)
            shifted_labels = data_loader['shifted_labels'].to(device)
            _loss = train_entity.get_masked_loss(entity_model, intent_model, src_ids, labels, shifted_labels)
            _loss = _loss.cpu().item()
            loss += _loss / N
            
        else:
            tgt_ids = sample['intent_ids'].to(device)
            shifted_tgt_ids = sample['shifted_intent_ids'].to(device)
            _loss = train_intent.get_masked_loss(intent_model, src_ids, tgt_ids, shifted_tgt_ids)
            _loss = _loss.cpu().item()
            loss += _loss / N
    
    return dict(loss=loss)


def validate_multitask(data_path):
    pass

