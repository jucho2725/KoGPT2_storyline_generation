
import torch
import torch.nn.functional as F
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2LMHeadModel
from kogpt2.configuration_gpt2 import GPT2Config
import gluonnlp


import argparse
import re

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
    "vocab_size": 50000,
}
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_p_logits(logits, top_p=0.0, filter_value=-float('Inf')):
    """Nucleus sampling"""
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits

def auto_enter(text):
    text = (text.replace("   ", "\n"))
    text = text.split("\n")

    text = [t.lstrip() for t in text if t != '']
    return "\n\n".join(text)

class GPT2:
    def __init__(self, load_path):
        ctx = 'cuda'
        cachedir = '~/kogpt2/'
        org_path = "trained_models/gpt2_j20_1007.pt"

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
        # 1013: special token 학습한 뒤로 keys 값이 달라져서 이와 같은 작업 필요
        checkpoint_org = torch.load(org_path, map_location=device)
        ckpt_final = {k:v for k, v in zip(checkpoint_org.keys(), checkpoint.values())} # 원래 state_dict 에 value 를 새로운 학습 결과로 바꿔줌

        # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
        self.kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
        self.kogpt2model.load_state_dict(ckpt_final)
        self.kogpt2model.to(device)


        self.kogpt2model.eval()
        self.vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                                  mask_token=None,
                                                                  sep_token=None,
                                                                  cls_token=None,
                                                                  unknown_token='<unk>',
                                                                  padding_token='<pad>',
                                                                  bos_token='<s>',
                                                                  eos_token='</s>')

        tok_path = get_tokenizer()
        self.tok = SentencepieceTokenizer(tok_path)

    def generation_fromutil(self, genres, input_sentence, temperature=0.7, top_p=0.8, top_k=40, text_size=100):
        ctx = 'cuda'
        device = torch.device(ctx)

        def get_info(vocab):
            ### gen_to_idx, genre_to_vocab 설정
            gen_to_vocab = {}
            genres = ['SF', 'TV영화', '공포', '느와르', '다큐멘터리', '드라마', '멜로', '로맨스', '모험', '무협', '뮤지컬',
                      '미스터리', '범죄', '블랙코미디', '서부', '서스펜스', '스릴러', '실험', '애니메이션', '액션', '웹무비',
                      '전쟁', '코미디', '판타지']
            gen_to_idx = {}
            for idx, gen in enumerate(genres):
                gen_to_idx[gen] = idx + 6
            idx_to_gen = {v: k for k, v in gen_to_idx.items()}

            for idx, gen in idx_to_gen.items():
                gen_to_vocab[gen] = vocab.idx_to_token[idx]
            return gen_to_vocab

        gen_toks = []
        for gen in genres:
            gen_to_vocab = get_info(self.vocab)
            gen_tok = gen_to_vocab[gen]
            gen_toks.append(gen_tok)

        total = []
        fmt = 'Genre: {:<6} Input Sentence: {:<4}'


        sent = ''
        sent = sent + input_sentence
        toked = self.tok(sent)

        input_ids = torch.tensor(
            [self.vocab[self.vocab.bos_token], ] + self.vocab[gen_toks] + self.vocab[toked]).unsqueeze(0)
        input_ids = input_ids.to(ctx)
        outputs = self.kogpt2model.generate(input_ids=input_ids, eos_token_id=1, pad_token_id=3, do_sample=True, num_return_sequences=1,
                                            max_length=text_size, min_length=50,
                                            top_p=top_p, top_k=top_k, temperature=temperature,
                                            repetition_penalty=1.2)

        generated_text = ''
        gen = self.vocab.to_tokens(outputs[0].squeeze().tolist())
        for tk in gen:
            generated_text += tk.replace('▁', ' ')
        sent = generated_text.replace("//", "\n")  # 비효율적이지만 엔터를 위해서 등장
        for unused_tok in list(gen_to_vocab.values()):
            sent = sent.replace(f"{unused_tok}", "")
        sent = sent.replace("<s>", "")
        sent = sent.replace("</s>", "")
        sent = auto_enter(sent)
        print(fmt.format(str(genres), sent))
        total.append(sent)
        return total

    def generation_fnc(self, genre, input_sentence, temperature=0.7, top_p=0.8, top_k=40, text_size=100):
        def get_info(vocab):
            ### gen_to_idx, genre_to_vocab 설정
            gen_to_vocab = {}
            genres = ['SF', 'TV영화', '공포', '느와르', '다큐멘터리', '드라마', '멜로', '로맨스', '모험', '무협', '뮤지컬',
                      '미스터리', '범죄', '블랙코미디', '서부', '서스펜스', '스릴러', '실험', '애니메이션', '액션', '웹무비',
                      '전쟁', '코미디', '판타지']
            gen_to_idx = {}
            for idx, gen in enumerate(genres):
                gen_to_idx[gen] = idx + 6
            idx_to_gen = {v: k for k, v in gen_to_idx.items()}

            for idx, gen in idx_to_gen.items():
                gen_to_vocab[gen] = vocab.idx_to_token[idx]
            return gen_to_vocab
        fmt = 'Genre: {:<6} Input Sentence: {:<4}'
        print(fmt.format(genre, input_sentence))
        gen_to_vocab = get_info(self.vocab)
        gen_tok = gen_to_vocab[genre]

        total = []
        for _ in range(5):

            sent = ''
            sent = sent + input_sentence

            toked = self.tok(sent)

            if len(toked) > 1022:
                break

            sent, generated_text = self.sample_sequence_with_genre(self.kogpt2model, self.tok, self.vocab, gen_tok, sent, text_size, temperature, top_p, top_k)
            # sent = sent.replace("//", "\n")  # 비효율적이지만 엔터를 위해서 등장
            # sent = sent.replace("</s>", "")
            # sent = auto_enter(sent)
            generated_text = generated_text.replace("//", "\n")  # 비효율적이지만 엔터를 위해서 등장
            generated_text = generated_text.replace("</s>", "")
            generated_text = auto_enter(generated_text)
            print(generated_text)
            total.append(generated_text)

        return total


    def sample_sequence_with_genre(self, model, tok, vocab, gen_tok, sent, text_size, temperature, top_p, top_k):
        ctx = 'cuda'
        device = torch.device(ctx)

        toked = tok(sent)  # 받은 문장
        count = 0
        generated_text = ''

        if len(toked) > 1024:
            return 0

        while 1:  # 이부분도 적절하게 바꾸기.
            # 시작 토큰 넣기
            input_ids = torch.tensor([vocab[vocab.bos_token], ] + [vocab[gen_tok], ] + vocab[toked]).unsqueeze(0)

            input_ids = input_ids.to(ctx)
            model = model.to(ctx)

            predicts = model(input_ids)
            pred = predicts[0]

            # temperature 적용
            logits = pred
            logits = logits[:, -1, :] / temperature
            # top k
            logits = top_k_logits(logits, top_k)
            # top p
            logits = top_p_logits(logits, top_p=top_p)

            # logits = logits.to(ctx)

            # 확률적을 뽑고
            log_probs = F.softmax(logits, dim=-1)
            # 이전 것들 저장해서 다음 학습에 사용
            prev = torch.multinomial(log_probs, num_samples=1)
            # 결과 나오게 (사전에서 gpt2가 뽑은 결과)
            gen = vocab.to_tokens(prev.squeeze().tolist())

            # 끝나면 본격적으로 만들어 놓기.
            if gen == '</s>' or count > text_size:
                # print('length:', count)
                # print('to_tokens:', vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist()))
                sent += gen.replace('▁', ' ')
                generated_text += gen.replace('▁', ' ')
                sent += '\n'
                generated_text += '\n'
                toked = tok(sent)
                count = 0
                break

            sent += gen.replace('▁', ' ')
            generated_text += gen.replace('▁', ' ')
            toked = tok(sent)
            count += 1
        return sent, generated_text


    def generation_byt(self, genre, input_sentence, temperature=0.7, top_p=0.8, top_k=40, text_size=100):
        ctx = 'cuda'
        device = torch.device(ctx)
        def get_info(vocab):
            ### gen_to_idx, genre_to_vocab 설정
            gen_to_vocab = {}
            genres = ['SF', 'TV영화', '공포', '느와르', '다큐멘터리', '드라마', '멜로', '로맨스', '모험', '무협', '뮤지컬',
                      '미스터리', '범죄', '블랙코미디', '서부', '서스펜스', '스릴러', '실험', '애니메이션', '액션', '웹무비',
                      '전쟁', '코미디', '판타지']
            gen_to_idx = {}
            for idx, gen in enumerate(genres):
                gen_to_idx[gen] = idx + 6
            idx_to_gen = {v: k for k, v in gen_to_idx.items()}

            for idx, gen in idx_to_gen.items():
                gen_to_vocab[gen] = vocab.idx_to_token[idx]
            return gen_to_vocab

        gen_to_vocab = get_info(self.vocab)
        gen_tok = gen_to_vocab[genre]

        total = []
        fmt = 'Genre: {:<6} Input Sentence: {:<4}'
        print(fmt.format(genre, input_sentence))
        # for _ in range(5):

        sent = ''
        sent = sent + input_sentence
        toked = self.tok(sent)

        input_ids = torch.tensor([self.vocab[self.vocab.bos_token], ] + [self.vocab[gen_tok], ] + self.vocab[toked]).unsqueeze(0)
        input_ids = input_ids.to(ctx)
        outputs = self.kogpt2model.generate(input_ids=input_ids, eos_token_id=1, pad_token_id=3, do_sample=True, num_return_sequences=1,
                                            max_length=text_size, min_length=50,
                                            top_p=top_p, top_k=top_k, temperature=temperature,
                                            repetition_penalty=1.2)

        generated_text = ''
        gen = self.vocab.to_tokens(outputs[0].squeeze().tolist())
        # print(gen)
        for tk in gen:
            generated_text += tk.replace('▁', ' ')
        sent = generated_text.replace("//", "\n")  # 비효율적이지만 엔터를 위해서 등장
        for unused_tok in list(gen_to_vocab.values()):
            sent = sent.replace(f"{unused_tok}", "")
        sent = sent.replace("<s>", "")
        sent = sent.replace("</s>", "")
        sent = auto_enter(sent)
        print(sent)
        total.append(sent)

        return total

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(dest='modelpath', help='load model path')
    #
    # args = parser.parse_args()

    ex1 = "해리는 이별의 아픔을 딛고 새 출발을 하고자 한다."
    gen1 = ["로맨스", "멜로"]
    ex2 = "원하는 결과가 나오지 않자, 브라운 박사는 빠르게 탈출 준비를 시작했다."
    gen2 = ["스릴러", "공포"]

    model = GPT2("trained_models/gpt2_genre_pad_50.pt")

    model.generation_fromutil(genres= gen1, input_sentence=ex1, temperature=0.9, top_p=0.95, top_k=30, text_size=200)
    model.generation_fromutil(genres= gen2, input_sentence=ex2, temperature=0.9, top_p=0.95, top_k=30, text_size=200)
    model.generation_fromutil(genres= gen2, input_sentence=ex1, temperature=0.9, top_p=0.95, top_k=30, text_size=200)
    model.generation_fromutil(genres= gen1, input_sentence=ex2, temperature=0.9, top_p=0.95, top_k=30, text_size=200)