import json
import logging
import pandas as pd
import numpy as np
import re
from flask import Flask, render_template, request
from flask import make_response
import os
import torch
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.model_copy.sample import sample_sequence
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from kogpt2.model_copy.torch_gpt2 import GPT2Config, GPT2LMHeadModel
import gluonnlp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
app = Flask(__name__)


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


ctx = 'cuda'
cachedir = './kogpt2/'
save_path = './checkpoint/'



tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)
model.load_state_dict(torch.load('./trained_models/201015_gpt2_story_add_10.pt'))
# model.load_state_dict(torch.load('./trained_models/201010_gpt2_story_35.pt'))

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:1'

model.to(device)
model.eval()

def next_tok(tokens, candidate_list):
    for i in range(len(candidate_list)):
        if candidate_list[i] not in tokens:
            return candidate_list[i]

# syno model
@app.route('/webhook', methods=['GET', 'POST'])
def get_answer():
    data = request.get_json(silent=True)
    sessionId = data['session']
    input_text = data['queryResult']['queryText']
    print(f"intput Text : {input_text}")
    text = sentence_generation(input_text)
    print(f"output Text : {text}")
    response = fulfilment_text(text)
    response = create_response(response)
    return response


def create_response(response):
    """ Creates a JSON with provided response parameters """

    # convert dictionary with our response to a JSON string
    res = json.dumps(response, indent=4)

    logger.info(res)

    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'

    # print(r)
    return r


def fulfilment_text(text):
    "intent parsing해서 해당 response 생성"
    response = {
        "fulfillmentText":
            text
    }
    return response


def sentence_generation(input_text):
    temperature = 0.7
    top_p = 0.8
    top_k = 40
    tmp_sent = ""
    text_size = 100
    loops = 5
    sent = input_text
    generated = []
    for i in range(5):
        toked = tok(sent)
        sent = sample_sequence(model, tok, vocab, sent, text_size, temperature, top_p, top_k)
        sent = sent.replace("//", "\n") # 비효율적이지만 엔터를 위해서 등장
        sent = sent.replace("</s>", "") 
        sent = auto_enter(sent)
        print(sent)
        generated.append(sent)
    
    for line in generated:
        if input_text != line.strip():
            return line
    return "적절한 대사(지문)가 없습니다."


if __name__ == '__main__':
    app.run('0.0.0.0', port=8088, threaded=True)