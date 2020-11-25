import os
import logging
import argparse

import torch
from torch.utils.data import DataLoader

from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from dataset import synoDataset
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from transformers import AdamW, get_linear_schedule_with_warmup

logging.getLogger().setLevel(logging.CRITICAL)


def define_argparser():
    """
    Define argument parser
    :return: configuration object
    """
    parser = argparse.ArgumentParser(description="run argparser")
    parser.add_argument(
        "--data_path",
        required=True,
        default="",
        help="storyline data path (csv format), must include content, genre columns",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        default="./train_models/",
        help="where to save model checkpoint",
    )

    parser.add_argument(
        "--max_len", type=int, default=1024, help="max sequence length for each data"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument(
        "--print_every",
        type=int,
        default=100,
        help="print average loss at every n step",
    )
    parser.add_argument(
        "--save_every", type=int, default=1, help="save the model at every N epoch"
    )
    args = parser.parse_args()
    return args


def main(args):
    tok_path = get_tokenizer()
    model, vocab = get_pytorch_kogpt2_model()
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = args.batch_size
    epochs = args.n_epochs
    learning_rate = 3e-5
    wamup_steps = 2000
    max_seq_len = 1024

    print("Dataset Loading... ", end=" ")
    dataset = synoDataset("./data/korean_naver_2.csv", vocab, tok)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("[[[Done]]]")

    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=wamup_steps, num_training_steps=-1
    )
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    model.zero_grad()

    models_folder = "trained_models"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    for epoch in range(epochs):
        print(f"Epoch {epoch} started" + "=" * 30)

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

            if batch_count == args.print_every:
                print(f"average loss for 100 epoch {sum_loss // args.print_every}")
                batch_count = 0
                sum_loss = 0.0

        # Store the model after each epoch to compare the performance of them
        if epoch % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"gpt2_genre_pad_{epoch}.pt"),
            )


if __name__ == "__main__":
    args = define_argparser()
    main(args)
