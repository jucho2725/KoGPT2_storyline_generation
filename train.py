import os
import logging

logging.getLogger().setLevel(logging.CRITICAL)

import torch
from torch.utils.data import DataLoader
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from dataset import synoDataset
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
max_seq_len = 1024

print("Dataset Loading... ", end=' ')
dataset = synoDataset('./data/korean_naver_2.csv', vocab, tok)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
print("[[[Done]]]")

from transformers import AdamW, get_linear_schedule_with_warmup

# model = torch.nn.DataParallel(model)
torch.cuda.device("cuda:1")
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
        # """  max 시퀀스가 넘으면 슬라이싱 """
        if len(syno) > max_seq_len:
            syno = syno[:max_seq_len]

        syno_tensor = torch.tensor(syno).unsqueeze(0).to(device)

        outputs = model(syno_tensor, labels=syno_tensor)
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
            print(f"average loss for 100 epoch {sum_loss / 1000}")
            batch_count = 0
            sum_loss = 0.0

    # Store the model after each epoch to compare the performance of them
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_genre_pad_{epoch}.pt"))


