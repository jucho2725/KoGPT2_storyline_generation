import os
import logging

logging.getLogger().setLevel(logging.CRITICAL)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import download, tokenizer, get_tokenizer
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from data import storyDataset
import gluonnlp
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

batch_size = 16
epochs = 100
learning_rate = 3e-5
wamup_steps = 5000
max_seq_len = 400

dataset = storyDataset('./data/korean_naver_2.csv', vocab, tok)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

from transformers import AdamW, get_linear_schedule_with_warmup

model = torch.nn.DataParallel(model)
model = model.to(device)
# print("devcie :", device)
model.train()
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=wamup_steps, num_training_steps=-1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0
model.zero_grad()

tmp_synos_tens = None
models_folder = 'trained_models'
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(epochs):
    print(f"Epoch {epoch} started" + '=' * 30)

    for idx, syno in enumerate(data_loader):
        syno_tens = torch.tensor(syno).unsqueeze(0).to(device)

        # torch.Size([1, number fo tokens])
        # skip sample from dataset if it is longer than max_seq_len
        if syno_tens.size()[1] > max_seq_len:
            continue

        # The first sequence in the sequence
        if not torch.is_tensor(tmp_synos_tens):
            tmp_synos_tens = syno_tens
            continue
        else:
            # The next syno does not fit in so we process the sequence and leave the last syno
            # as the start for next sequence
            if tmp_synos_tens.size()[1] + syno_tens.size()[1] > max_seq_len:
                work_synos_tens = tmp_synos_tens
                tmp_synos_tens = syno_tens
            else:
                # Add the syno to sequence, continue and try to add more
                tmp_synos_tens = torch.cat([tmp_synos_tens, syno_tens[:, 1:]], dim=1)
                continue

        # sequence ready, process it through the model
        outputs = model(work_synos_tens, labels=work_synos_tens)
        loss, logits = outputs[:2]
        loss.backward()
        sum_loss = sum_loss + loss.detach().data

        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == batch_size:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

    # Store the model after each epoch to compare the performance of them
    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_syno_{epoch}.pt"))


