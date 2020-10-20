import os
import logging

logging.getLogger().setLevel(logging.CRITICAL)

import torch
from torch.utils.data import DataLoader
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from data import synoDataset
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
import torch
import torch.nn.functional as F
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
import gluonnlp

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()

print(model.state_dict().keys())

pytorch_kogpt2 = {
    'url':
        './checkpoint/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000
}

model_syno = torch.load("trained_models/nipa/gpt2_medium_syno_95.pt", map_location="cuda")
model_story = torch.load("trained_models/gpt2_medium_syno_95.pt", map_location="cuda")
# print(model_syno.keys())
model_final = {k: v for k, v in zip(model_story.keys(), model_syno.values())}

# KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
kogpt2model.load_state_dict(model_final)
print('success')

kogpt2mod