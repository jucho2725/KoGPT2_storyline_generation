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
import re

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
dataset = storyDataset('./data/korean_naver_1.csv', vocab, tok)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

batch_size = 16
epochs = 100
learning_rate = 3e-5
wamup_steps = 5000
max_seq_len = 400

from transformers import AdamW, get_linear_schedule_with_warmup

# model = torch.nn.DataParallel(model)
model = model.to(device)
print("device :", device)
model.train()
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=wamup_steps, num_training_steps=-1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0
model.zero_grad()

tmp_jokes_tens = None
models_folder = 'trained_models'
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(epochs):
    print(f"Epoch {epoch} started" + '=' * 30)

    for idx, joke in enumerate(data_loader):
        joke_tens = torch.tensor(joke).unsqueeze(0).to(device)

        # torch.Size([1, number fo tokens])
        # skip sample from dataset if it is longer than max_seq_len
        if joke_tens.size()[1] > max_seq_len:
            continue

        # The first joke sequence in the sequence
        if not torch.is_tensor(tmp_jokes_tens):
            tmp_jokes_tens = joke_tens
            continue
        else:
            # The next joke does not fit in so we process the sequence and leave the last joke
            # as the start for next sequence
            if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > max_seq_len:
                work_jokes_tens = tmp_jokes_tens
                tmp_jokes_tens = joke_tens
            else:
                # Add the joke to sequence, continue and try to add more
                tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:, 1:]], dim=1)
                continue

        # sequence ready, process it through the model
        outputs = model(work_jokes_tens, labels=work_jokes_tens)
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
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_joker_{epoch}.pt"))
        print(f"========= {epoch} 번째 문장 생성 ========")
        sent = "해리와 그의 연인 포터는 한적한 마을 해그리드에 살고 있었다."
        toked = tok(sent)
        print("input sentence: ", sent)
        while 1:
            if len(sent) > 100:
                break
            input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0).to(device)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=50, repetition_penalty=1.2, do_sample=True, eos_token_ids=-1, num_return_sequences=3)

            for i in range(3):
                toked = vocab.to_tokens(outputs[0][i].squeeze().tolist())
                ret = re.sub(r'(<s>|</s>)', '', ''.join(toked).replace('▁', ' ').strip())
                print('Generated {}: {}'.format(i, ret))
            # gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
            # sent += gen.replace('▁', ' ')
            # toked = tok(sent)
        print("final sentence:", sent)


# def generate_some_text(input_str, text_len=250):
#     cur_ids = torch.tensor(tokenizer.encode(input_str)).unsqueeze(0).long().to(device)
#
#     model.eval()
#     with torch.no_grad():
#         for i in range(text_len):
#             outputs = model(cur_ids, labels=cur_ids)
#             loss, logits = outputs[:2]
#             softmax_logits = torch.softmax(logits[0, -1],
#                                            dim=0)  # Take the first(only one) batch and the last predicted embedding
#             next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(),
#                                             n=10)  # Randomly(from the given probability distribution) choose the next word from the top n words
#             cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id],
#                                 dim=1)  # Add the last word
#
#         output_list = list(cur_ids.squeeze().to('cpu').numpy())
#         output_text = tokenizer.decode(output_list)
#         print(output_text)
#
#
# # Function to first select topN tokens from the probability list and then based on the selected N word distribution
# # get random token ID
# def choose_from_top(probs, n=5):
#     ind = np.argpartition(probs, -n)[-n:]
#     top_prob = probs[ind]
#     top_prob = top_prob / np.sum(top_prob)  # Normalize
#     choice = np.random.choice(n, 1, p=top_prob)
#     token_id = ind[choice][0]
#     return int(token_id)


model.eval()
sent = "해리와 그의 연인 포터는 한적한 마을 해그리드에 살고 있었다."
toked = tok(sent)
print("input sentence: ", sent)
while 1:
    if len(sent) > 100:
        break
    input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0).to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=50, repetition_penalty=1.2, do_sample=True, eos_token_ids=-1, num_return_sequences=3)

    for i in range(3):
        toked = vocab.to_tokens(outputs[0][i].squeeze().tolist())
        ret = re.sub(r'(<s>|</s>)', '', ''.join(toked).replace('▁', ' ').strip())
        print('Generated {}: {}'.format(i, ret))
    # gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
    # sent += gen.replace('▁', ' ')
    # toked = tok(sent)
print("final sentence:", sent)

