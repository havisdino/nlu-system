import torch
from torch.utils.data import DataLoader
from torchtext import transforms
import jsonlines as jsl
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self, label_path, tokenizer, config_path, maxlen):
        super().__init__()
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        with jsl.open(label_path) as file:
            self.data = list(file)

        with open(config_path) as file:
            self.type_to_id = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return self.preprocess(sample)

    def preprocess(self, sample: dict):
        sentence = sample['sentence'].lower()
        intent = sample['intent'].lower()

        sentence_ids = self.tokenizer.encode(sentence, out_type=int)[:self.maxlen]
        
        intent_ids = self.tokenizer.encode(intent, out_type=int, add_bos=True, add_eos=True)
        shifted_intent_ids = intent_ids[1:1 + self.maxlen]
        intent_ids = intent_ids[:len(shifted_intent_ids)]
        
        # labels = self.label(sentence, sample['entities'])[:self.maxlen]
        labels = [0] + self.label(sentence, sample['entities'])
        shifted_labels = labels[1:1 + self.maxlen]
        labels = labels[:len(shifted_labels)]
        
        return dict(
            sentence_ids=sentence_ids, intent_ids=intent_ids,
            shifted_intent_ids=shifted_intent_ids, labels=labels,
            shifted_labels=shifted_labels
        )
    
    def label(self, sentence, entities):
        sentence_ids = self.tokenizer.encode(sentence)
        label_ids = [0 for _ in range(len(sentence_ids))]
        for entity in entities:
            filler = entity['filler'].lower()
            type = entity['type'].lower()
            filler_ids = self.tokenizer.encode(filler)

            L = len(filler_ids)
            for i in range(len(sentence_ids) - L):
                if sentence_ids[i:i + L] == filler_ids:
                    label_ids[i:i + L] = [self.type_to_id[type] for _ in range(L)]
        return label_ids


def collate_fn(batch):
    sentence_ids = [sample['sentence_ids'] for sample in batch]
    intent_ids = [sample['intent_ids'] for sample in batch]
    shifted_intent_ids = [sample['shifted_intent_ids'] for sample in batch]
    labels = [sample['labels'] for sample in batch]
    shifted_labels = [sample['shifted_labels'] for sample in batch]

    sentence_ids = transforms.F.to_tensor(sentence_ids, padding_value=0)
    intent_ids = transforms.F.to_tensor(intent_ids, padding_value=0)
    shifted_intent_ids = transforms.F.to_tensor(shifted_intent_ids, padding_value=0)
    labels = transforms.F.to_tensor(labels, padding_value=0)
    shifted_labels = transforms.F.to_tensor(shifted_labels, padding_value=0)

    return dict(
        sentence_ids=sentence_ids, intent_ids=intent_ids,
        shifted_intent_ids=shifted_intent_ids, labels=labels,
        shifted_labels=shifted_labels
    )
