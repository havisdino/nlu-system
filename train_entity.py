from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from utils import get_config, lr_schedule
from model import (get_entity_model, EntityRecognizer,
                   Transformer, SARER, get_intent_model)
from dataset import Dataset, DataLoader, collate_fn
from utils import get_key_padding_mask
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from sentencepiece import SentencePieceProcessor
from tqdm.autonotebook import tqdm
from typing import Union
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter


def get_masked_loss(
    entity_model: Union[EntityRecognizer, SARER],
    intent_model: Transformer,
    src_ids, labels, shifted_labels):
    
    encoder_output = intent_model(
        src_ids=src_ids, tgt_ids=None, encoder_output_only=True)
    
    src_mask = get_key_padding_mask(src_ids)

    logits = entity_model(
        input=encoder_output,
        key_padding_mask=src_mask,
        tgt_ids=labels,
        tgt_key_padding_mask=src_mask
    )
    
    mask = 1 - get_key_padding_mask(src_ids, binary=True).float()
    loss = F.cross_entropy(
        input=logits.view(-1, logits.size(-1)),
        target=shifted_labels.view(-1), reduction='none'
    )

    loss = (loss * mask.view(-1)).sum() / mask.sum()
    return loss


def train_step(entity_model, intent_model, optimizer, src_ids, labels, shifted_labels):
    entity_model.train()
    intent_model.train()
    
    loss = get_masked_loss(entity_model, intent_model, src_ids, labels, shifted_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().item()


def train_epoch(entity_model, intent_model, optimizer, lr_scheduler, data_loader, device, writer, step, epoch=None):
    losses = []
    for sample in (bar := tqdm(data_loader)):
        src_ids = sample['sentence_ids'].to(device)
        labels = sample['labels'].to(device)
        shifted_labels = sample['shifted_labels'].to(device)

        loss = train_step(entity_model, intent_model, optimizer, src_ids, labels, shifted_labels)
        lr_scheduler.step()
        losses.append(loss)
        step[0] += 1
        writer.add_scalar('train/step_loss', loss, step[0])
        _lr = optimizer.param_groups[0]['lr']
        bar.set_description(f'epoch: {epoch} - loss: {loss:.6f} - lr: {_lr:.6f}')

    epoch_loss = sum(losses) / len(losses)
    return dict(step_losses=losses, epoch_loss=epoch_loss)


def fit(entity_model, intent_model, optimizer, lr_scheduler, data_loader, device, writer, epochs=1, checkpoint_epoch=1):
    epoch_losses = []
    step_losses = []
    step = [0]
    for epoch in range(1, 1 + epochs):
        output = train_epoch(entity_model, intent_model, optimizer, lr_scheduler, data_loader, device, writer, step, epoch)
        step_losses += output['step_losses']
        epoch_losses.append(output['epoch_loss'])
        writer.add_scalar('train/epoch_loss', epoch_losses[-1], epoch)
        if epoch % checkpoint_epoch == 0:
            torch.save(
                dict(statedict=entity_model.state_dict(),
                     optimizer=optimizer,
                     lr_scheduler=lr_scheduler,
                     last_step=step,
                     losses=epoch_losses),
                f=f'logs/checkpoints/entity_{epoch}.pth'
            )
            
            torch.save(
                dict(statedict=intent_model.state_dict()),
                f=f'logs/checkpoints/base_{epoch}.pth'
            )
            
    return dict(step_losses=step_losses, epoch_losses=epoch_losses)


if __name__ == '__main__':
    import pickle as pkl
    import os
    from torch import nn
    

    try:
        os.makedirs('logs/checkpoints/')
    except FileExistsError:
        pass
    
    try:
        os.makedirs('logs/train_info')
    except FileExistsError:
        pass
    
    parser = ArgumentParser()
    parser.add_argument('-l', '--log-dir', type=str, default='logs/tensorboard/entity')
    parser.add_argument('-d', '--data-path', type=str, default='labels/train.jsonl')
    parser.add_argument('-ec', '--entity-config-path', type=str, default='configs/entity_types.json')
    parser.add_argument('-mc', '--model-config-path', type=str, default='configs/model_configs.json')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-dv', '--device', type=str, default=None)
    parser.add_argument('-ch', '--checkpoint_epoch', type=int, default=1)
    args = parser.parse_args()
    
    
    if args.device in ['cuda', 'cpu', 'mps']:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    intent_model = get_intent_model(args.model_config_path)
    intent_model.to(device)
    entity_model = get_entity_model(args.entity_config_path, args.model_config_path)
    entity_model.to(device)
    
    wrapper = nn.ModuleList([intent_model, entity_model])
    
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1.)
    lr_scheduler = LambdaLR(optimizer, lr_schedule)

    tokenizer = SentencePieceProcessor('tokenizer/tok.model')
        
    maxlen = get_config(args.model_config_path)['max_sequence_length']
        
    dataset = Dataset(label_path=args.data_path, tokenizer=tokenizer,
                      config_path=args.entity_config_path, maxlen=maxlen)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, prefetch_factor=2, collate_fn=collate_fn)
    writer = SummaryWriter(args.log_dir)

    history = fit(entity_model, intent_model, optimizer, lr_scheduler, data_loader, device,
                  writer, epochs=args.epochs, checkpoint_epoch=args.checkpoint_epoch)
    
    with open('logs/train_info/train_entity_history.bin', 'wb') as file:
        pkl.dump(history, file)
