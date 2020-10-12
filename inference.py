
import torch
import torch.nn.functional as F
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
import gluonnlp
import argparse

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
        self.model, self.vocab = kogpt2model, vocab_b_obj
        self.tok = SentencepieceTokenizer(tok_path)



    def generation(self, input_sentence, temperature=0.7, top_p=0.8, top_k=40, text_size=100):
        total = []
        for _ in range(5):

            sent = ''
            sent = sent + input_sentence

            toked = self.tok(sent)

            if len(toked) > 1022:
                break

            sent = self.sample_sequence(self.model, self.tok, self.vocab, sent, text_size, temperature, top_p, top_k)
            sent = sent.replace("//", "\n")  # 비효율적이지만 엔터를 위해서 등장
            sent = sent.replace("</s>", "")
            sent = auto_enter(sent)
            total.append(sent)

        return total

    def sample_sequence(self, model, tok, vocab, sent, text_size, temperature, top_p, top_k):
        ctx = 'cuda'
        device = torch.device(ctx)

        toked = tok(sent)  # 받은 문장
        count = 0
        generated_text = ''

        if len(toked) > 1024:
            return 0

        while 1:  # 이부분도 적절하게 바꾸기.
            # 시작 토큰 넣기
            input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)

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
                # print(sent)
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
        return sent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='modelpath', help='load model path')

    args = parser.parse_args()

    ex1 = "해리는 이별의 아픔을 딛고 새 출발을 하고자 한다."
    ex2 = "원하는 결과가 나오지 않자, 브라운 박사는 빠르게 탈출 준비를 시작했다."

    model = GPT2(args.modelpath)

    model.generation(input_sentence=ex1, text_size=200)
    model.generation(input_sentence=ex2, text_size=200)