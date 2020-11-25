from torch.utils.data import Dataset

from kogpt2.utils import download, tokenizer, get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp
import numpy as np
import pandas as pd


def sentencePieceTokenizer():
    tok_path = get_tokenizer()
    sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

    return sentencepieceTokenizer


def koGPT2Vocab():
    cachedir = "~/kogpt2/"

    # download vocab
    vocab_info = tokenizer
    vocab_path = download(
        vocab_info["url"], vocab_info["fname"], vocab_info["chksum"], cachedir=cachedir
    )

    koGPT2_vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(
        vocab_path,
        mask_token=None,
        sep_token=None,
        cls_token=None,
        unknown_token="<unk>",
        padding_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    return koGPT2_vocab


def toString(list):
    if not list:
        return ""
    result = ""

    for i in list:
        result = result + i
    return result


class storyDataset(Dataset):
    """script dataset"""

    def __init__(self, file_path, vocab, tokenizer):
        self.file_path = file_path
        self.sentence_list = []
        self.vocab = vocab
        self.tokenizer = tokenizer

        df = pd.read_csv(self.file_path)

        for line in df["content"]:
            tokenized_line = tokenizer(str(line))
            print(tokenized_line[-1])

            index_of_words = (
                [vocab[vocab.bos_token],]
                + vocab[tokenized_line]
                + [vocab[vocab.eos_token]]
            )
            self.sentence_list.append(index_of_words)
        print("sentence list length :", len(self.sentence_list))

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, index):
        return self.sentence_list[index]


class synoDataset(Dataset):
    """synopsis dataset"""

    def __init__(self, file_path, vocab, tokenizer):
        self.file_path = file_path
        self.sentence_list = []
        self.vocab = vocab
        self.tokenizer = tokenizer

        df = pd.read_csv(self.file_path)

        df["genre"] = df["genre"].str.strip("[]").str.split(",")
        # df['genre'] = df['genre'].fillna('none')

        ### gen_to_idx, genre_to_vocab 설정
        gen_to_vocab = {}
        genres = [
            "SF",
            "TV영화",
            "공포",
            "느와르",
            "다큐멘터리",
            "드라마",
            "멜로",
            "로맨스",
            "모험",
            "무협",
            "뮤지컬",
            "미스터리",
            "범죄",
            "서부",
            "서스펜스",
            "스릴러",
            "애니메이션",
            "액션",
            "멜로/로맨스",
            "가족",
            "서사",
            "전쟁",
            "코미디",
            "판타지",
        ]
        print(f"We have {len(genres)} genres")
        gen_to_idx = {}
        for idx, gen in enumerate(genres):
            gen_to_idx[gen] = idx + 6
        idx_to_gen = {v: k for k, v in gen_to_idx.items()}

        for idx, gen in idx_to_gen.items():
            gen_to_vocab[gen] = vocab.idx_to_token[idx]

        count = 0
        err = 0
        for idx in range(len(df)):
            line = df.loc[idx, "content"]
            genres = df.loc[idx, "genre"]
            tokenized_line = tokenizer(str(line))
            if genres == "'none'":
                print(genres)
                index_of_words = (
                    [vocab[vocab.bos_token],]
                    + vocab[tokenized_line]
                    + [vocab[vocab.eos_token]]
                )
            else:
                tmp = []

                for gen in genres:
                    try:
                        tmp.append(gen_to_vocab[gen.strip("' '")])
                    except Exception as e:
                        pass
                if len(tmp) > 0:
                    count += 1
                else:
                    err += 1
                index_of_words = (
                    [vocab[vocab.bos_token],]
                    + vocab[tmp]
                    + vocab[tokenized_line]
                    + [vocab[vocab.eos_token]]
                )
            self.sentence_list.append(index_of_words)

        print(f"average length of data : {sum(df['content'].str.len()) / len(df)}")

        print("total data :", len(self.sentence_list))

        print("=== test genres ===")
        print(f"we got {count} synos which have genres.")
        print(f"we lose {err} synos because their genres are not included.")
        print(f"match full == count + err {len(self.sentence_list) == count+err}")

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, index):
        return self.sentence_list[index]


if __name__ == "__main__":
    import torch
    from gluonnlp.data import SentencepieceTokenizer
    from kogpt2.utils import get_tokenizer
    from dataset import synoDataset
    from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
    from time import time

    tok_path = get_tokenizer()
    _, vocab = get_pytorch_kogpt2_model()
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

    start = time()
    print("Dataset Loading... ", end=" ")
    dataset = synoDataset("./data/korean_naver_3.csv", vocab, tok)
    end = time()
    print(f"{start - end}")
