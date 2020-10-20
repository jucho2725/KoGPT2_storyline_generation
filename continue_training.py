import os
import logging

logging.getLogger().setLevel(logging.CRITICAL)

import torch
from torch.utils.data import DataLoader
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from data import synoDataset
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model

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

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)
org_path = "trained_models/gpt2_j20_1007.pt"
load_path = "trained_models/gpt2_medium_syno_95.pt"

checkpoint = torch.load(load_path, map_location=device)
# 1013: special token 학습한 뒤로 keys 값이 달라져서 이와 같은 작업 필요
checkpoint_org = torch.load(org_path, map_location=device)

ckpt_final = {k: v for k, v in
              zip(checkpoint_org.keys(), checkpoint.values())}  # 원래 state_dict 에 value 를 새로운 학습 결과로 바꿔줌

# KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
model.load_state_dict(ckpt_final)
model.to(device)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

batch_size = 16
epochs = 100
learning_rate = 3e-5
wamup_steps = 5000
max_seq_len = 400

print("Dataset Loading... ", end=' ')
dataset = synoDataset('./data/korean_naver_2.csv', vocab, tok)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
print("[[[Done]]]")

from transformers import AdamW, get_linear_schedule_with_warmup

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
        # data = torch.stack(syno)  # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
        # data = data.transpose(1, 0)
        # syno_tens = data.to(device)

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


