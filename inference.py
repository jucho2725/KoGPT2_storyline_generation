
import torch
import torch.nn.functional as F
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2LMHeadModel
from kogpt2.configuration_gpt2 import GPT2Config
import gluonnlp

from koalanlp.Util import initialize, finalize
from koalanlp.proc import Tagger
from koalanlp import API
from collections import Counter

initialize(EUNJEON='LATEST')
tagger = Tagger(API.EUNJEON)
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

def get_modelinfo(model):
    for (name, info) in iter(model.named_parameters()):
        print(f"{name} : {info.shape}")
    print(f"_metadata {model.state_dict()._metadata}")
    print(f"_params {model._parameters.items()}")
    print(f"_buffer {model._buffers.items()}")
    print(f"_module {model._modules.items()}")
    print(f"_state_dict_hooks {model._state_dict_hooks.values()}")
    print("=" * 50)

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

def heuristic_topk(outputs, total, k):
    """
    :param outputs: list of list of token numbers
    :return: list of list of token numbers
    select k best performed sentences
    """
    result = []
    score = []

    # 1. 가장 길게 나온 것이 가장 좋은 결과일 것이다.
    for i in range(len(outputs)):
        score.append([i, len(outputs[i])])

    # 2. 같은 문장에서 같은 명사가 두 번이상 나오면 점수를 -1/3
    for i in range(len(total)):
        sents = tagger(total[i])
        for sent in sents:
            tag_info = [] # 문장 마다 확인
            for word in sent:
                for morph in word:
                    tag_info.append((morph.getSurface(), str(morph.getTag())))
            c = Counter(tag_info)
            for ((word, pos), v) in c.items():
                if pos.startswith("NNP") and v >= 2:
                    score[i][1] -= 70
    # 3. 문장 정렬
    score = sorted(score, key=lambda x: x[1], reverse=True)
    print(score[k:])
    for info in score[:k]:
        print(info[0])
        result.append(total[info[0]])
    return result


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

    def generation_fromutil(self, genres, input_sentence, temperature=0.9, top_p=0.8, top_k=20, text_size=200, repeat=10):
        ctx = 'cuda'
        device = torch.device(ctx)

        def get_info(vocab):
            ### gen_to_idx, genre_to_vocab 설정
            gen_to_vocab = {}
            genres = ['SF', 'TV영화', '공포', '느와르', '다큐멘터리', '드라마', '멜로', '로맨스', '모험', '무협', '뮤지컬',
                      '미스터리', '범죄', '서부', '서스펜스', '스릴러', '애니메이션', '액션',
                      '멜로/로맨스', '가족', '서사', '전쟁', '코미디', '판타지']
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
        outputs = []
        sent = ''
        sent = sent + input_sentence
        toked = self.tok(sent)

        input_ids = torch.tensor(
            [self.vocab[self.vocab.bos_token], ] + self.vocab[gen_toks] + self.vocab[toked]).unsqueeze(0)
        input_ids = input_ids.to(ctx)

        for _ in range(repeat):
            output = self.kogpt2model.generate(input_ids=input_ids, eos_token_id=1, pad_token_id=3, do_sample=True,
                                                num_return_sequences=1,
                                                max_length=text_size, min_length=50,
                                                top_p=top_p, top_k=top_k, temperature=temperature,
                                                repetition_penalty=1.2)
            outputs.append(output[0].squeeze().tolist())
            generated_text = ''
            gen = self.vocab.to_tokens(output[0].squeeze().tolist())

            for tk in gen:
                generated_text += tk.replace('▁', ' ')
            sent = generated_text.replace("//", "\n")  # 비효율적이지만 엔터를 위해서 등장
            for unused_tok in list(gen_to_vocab.values()):
                sent = sent.replace(f"{unused_tok}", "")
            sent = sent.replace("<s>", "")
            sent = sent.replace("</s>", "")
            sent = auto_enter(sent)
            total.append(sent)


        result = heuristic_topk(outputs, total, 1)
        return result

if __name__ == "__main__":

    gen = [['멜로/로맨스'], ['공포', '스릴러'], ["SF"]]
    ex = ["반복되는 일상을 못 견디던 엘리엇은 세탁소에서 만난 제인과 사랑에", "헬렌과 함께 만나게 된 일행들은 지하실로 내려가지만, 그곳에는", "지구에서 탈출한 키모아와 그의 친구들은 수수께끼의 행성 N95로"]

    model = GPT2("trained_models/gpt2_genre_30.pt")

    from datetime import datetime
    cur = datetime.now().strftime(r"%m%d_%H%M")
    f = open(f"samples/result_{cur}.txt", 'w', encoding="utf-8")
    for i in range(3):
        # print(i)
        fmt = 'Genre: {:<6} Input Sentence: {:<4}'
        res_l = model.generation_fromutil(genres= gen[i], input_sentence=ex[i], temperature=0.9, top_p=0.8, top_k=20, text_size=200)
        f.write(fmt.format(str(gen[i]), ex[i]))
        f.write("\n")
        for sent in res_l:
            f.write("\n")
            f.write(sent)

    f.close()
    finalize()