import os
import torch
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.model.sample import sample_sequence
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
import gluonnlp
import argparse

pytorch_kogpt2 = {
    'url':
        'checkpoint/pytorch_kogpt2_676e9bcfa7.params',
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


def auto_enter(text):
    text = (text.replace("   ", "\n"))
    text = text.split("\n")

    text = [t.lstrip() for t in text if t != '']
    return "\n\n".join(text)


def generation(temperature=0.7, top_p=0.8, top_k=40, tmp_sent="", text_size=100,
               loops=0, load_path=""):
    ctx = 'cuda'
    cachedir = '~/kogpt2/'
    save_path = './checkpoint/'

    # download vocab
    vocab_info = tokenizer
    vocab_path = download(vocab_info['url'],
                          vocab_info['fname'],
                          vocab_info['chksum'],
                          cachedir=cachedir)
    # Device 설정
    device = torch.device(ctx)
    # 저장한 Checkpoint 불러오기
    checkpoint = torch.load(load_path, map_location=device)

    # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    kogpt2model.load_state_dict(checkpoint)

    kogpt2model.eval()
    vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                              mask_token=None,
                                                              sep_token=None,
                                                              cls_token=None,
                                                              unknown_token='<unk>',
                                                              padding_token='<pad>',
                                                              bos_token='<s>',
                                                              eos_token='</s>')

    tok_path = get_tokenizer()
    model, vocab = kogpt2model, vocab_b_obj
    tok = SentencepieceTokenizer(tok_path)

    if loops:
        num = 1
    else:
        num = 0

    try:
        load_path.split("/")[-2]
    except:
        pass
    else:
        load_path = load_path.split("/")[-2]

    print("ok : ", load_path)

    if not (os.path.isdir("samples/" + load_path)):
        os.makedirs(os.path.join("samples/" + load_path))

    for _ in range(5):

        sent = ''
        if tmp_sent == "":
            tmp_sent = input('input : ')
        sent = sent + tmp_sent

        toked = tok(sent)

        if len(toked) > 1022:
            break

        sent = sample_sequence(model, tok, vocab, sent, text_size, temperature, top_p, top_k)
        sent = sent.replace("//", "\n")  # 비효율적이지만 엔터를 위해서 등장
        sent = sent.replace("</s>", "")
        sent = auto_enter(sent)
        print(sent)

        """ sample 해서 파일로 저장하는 부분. 저장안할거면 필요없음 """
        now = [int(n) for n in os.listdir("./samples/" + load_path)]

        try:
            now = max(now)
        except:
            now = 1

        f = open(f"samples/" + load_path + f"/{temperature}{top_p}{top_k}" + str(now + 1), 'w', encoding="utf-8")

        head = [load_path, tmp_sent, text_size, temperature, top_p, top_k]
        head = [str(h) for h in head]
        f.write(",".join(head))
        f.write(",")
        f.write(sent)
        f.close()

        # tmp_sent = ""

        if num != 0:
            num += 1
            if num >= loops:
                print("good")
                return

if __name__ == "__main__":
    args = argparse.

    generation(temperature=0.7, top_p=0.8, top_k=40, text_size=200, load_path="")