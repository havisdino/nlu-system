import torch
from torch import nn
import json
from utils import get_key_padding_mask
from utils import get_config


def load_model_from_checkpoint(
    path, map_location, config_path, model_type='intent', entity_config_path=None):
    # model_type: 'intent' or 'entity'
    
    assert model_type == 'intent' or model_type == 'entity'
    
    state_dict = torch.load(path, map_location)['statedict']
    if model_type == 'intent':
        model = get_intent_model(config_path)
    else:
        assert entity_config_path is not None
        model = get_entity_model(entity_config_path, config_path)
        
    model.load_state_dict(state_dict)
    return model


def get_intent_model(config_path='configs/model_configs.json'):
    configs = get_config(config_path)
    return Transformer(configs)


def get_entity_model(
    entity_config_path='configs/entity_types.json',
    model_config_path='configs/model_configs.json'):
    
    model_configs = get_config(model_config_path)
    entity_configs = get_config(entity_config_path)
    n_types = len(entity_configs.keys())
    
    return SARER(n_types, model_configs)


class Transformer(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.configs = configs
        d_model = configs['d_model']
        vocab_size = configs['vocab_size']
        dff = configs['dim_feedforward']
        nhead = configs['nhead']
        dropout = configs['dropout']
        batch_first = configs['batch_first']
        norm_first = configs['norm_first']
        maxlen = configs['max_sequence_length']

        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )
        self.position_embed = PositionalEmbedding(
            maxlen=maxlen, d_model=d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dff,
                dropout=dropout,
                batch_first=batch_first,
                norm_first=norm_first
            ),
            num_layers=configs['enc_num_layers'],
            norm=nn.LayerNorm(d_model)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dff,
                dropout=dropout,
                batch_first=batch_first,
                norm_first=norm_first
            ),
            num_layers=configs['dec_num_layers'],
            norm=nn.LayerNorm(d_model)
        )
        self.last_ff = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, vocab_size)
        )
        self.register_buffer(
            name='causal_mask',
            tensor=nn.Transformer.generate_square_subsequent_mask(
                sz=maxlen
            )
        )

    def get_causal_mask(self, size):
        return self.causal_mask[:size, :size]

    def forward(self, src_ids, tgt_ids, encoder_output_only=False, inference=False, encoder_output=None):
        assert inference + (encoder_output is None) == 1
        
        src_key_padding_mask = get_key_padding_mask(src_ids)
        if not inference:
            src_pe = self.position_embed(src_ids)
            src = self.embed(src_ids) + src_pe
            
            encoder_output = self.encoder(
                src=src, src_key_padding_mask=src_key_padding_mask, is_causal=False
            )
            if encoder_output_only:
                return encoder_output

        tgt_key_padding_mask = get_key_padding_mask(tgt_ids)
        tgt_pe = self.position_embed(tgt_ids)
        tgt = self.embed(tgt_ids) + tgt_pe
        
        causal_mask = self.get_causal_mask(tgt_ids.size(1))
        logits = self.decoder(
            tgt=tgt, memory=encoder_output, tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        logits = self.last_ff(logits)  
        return dict(encoder_output=encoder_output, decoder_output=logits)


class PositionalEmbedding(nn.Module):
    def __init__(self, maxlen, d_model, n=10000.0):
        super().__init__()
        assert maxlen % 2 == 0
        self.n = n
        pemat = torch.zeros(maxlen, d_model)
        pos = torch.arange(0, maxlen).unsqueeze(-1)
        denom = n ** (torch.arange(0, d_model, 2) / d_model)
        pemat[:, 0::2] = torch.sin(pos / denom)
        pemat[:, 1::2] = torch.cos(pos / denom)
        self.register_buffer('pemat', pemat)

    def forward(self, x):
        return self.pemat[None, :x.size(1)]


class EntityRecognizer(nn.Module):
    def __init__(self, n_types, configs: dict):
        super().__init__()
        d_model = configs['d_model']
        dff = configs['dim_feedforward']
        self.top_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=configs['nhead'],
                dim_feedforward=dff,
                dropout=configs['dropout'],
                batch_first=configs['batch_first'],
                norm_first=configs['norm_first']
            ),
            num_layers=configs['num_markov_layers'],
            norm=nn.LayerNorm(d_model)
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, n_types)
        )

    def forward(self, input, key_padding_mask, **kwargs):
        logits = self.top_encoder(input, src_key_padding_mask=key_padding_mask)
        logits = self.ff(logits)
        return logits


# ==================== Experimental =========================
# Semi-autoregressive Entity Recognizer

class SARER(nn.Module):
    def __init__(self, num_types, configs: dict):
        super().__init__()
        self.configs = configs
        d_model = configs['d_model']
        dff = configs['dim_feedforward']
        nhead = configs['nhead']
        dropout = configs['dropout']
        batch_first = configs['batch_first']
        norm_first = configs['norm_first']
        maxlen = configs['max_sequence_length']

        self.type_embed = nn.Embedding(
            num_embeddings=num_types,
            embedding_dim=d_model
        )
        self.position_embed = PositionalEmbedding(
            maxlen=maxlen, d_model=d_model)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dff,
                dropout=dropout,
                batch_first=batch_first,
                norm_first=norm_first
            ),
            num_layers=configs['num_markov_layers'],
            norm=nn.LayerNorm(d_model)
        )
        self.last_ff = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, num_types)
        )
        self.register_buffer(
            name='causal_mask',
            tensor=nn.Transformer.generate_square_subsequent_mask(
                sz=maxlen
            )
        )

    def get_causal_mask(self, size):
        return self.causal_mask[:size, :size]

    def forward(self, input, key_padding_mask, tgt_ids, tgt_key_padding_mask, **kwargs):
        tgt_pe = self.position_embed(tgt_ids)
        tgt = self.type_embed(tgt_ids) + tgt_pe

        causal_mask = self.get_causal_mask(tgt_ids.size(1))
        logits = self.decoder(
            tgt=tgt, memory=input, tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=key_padding_mask
        )
        logits = self.last_ff(logits)
        return logits
