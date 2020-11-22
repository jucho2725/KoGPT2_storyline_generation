import json
import logging
import pandas as pd
import numpy as np
import re
import torch
from flask import Flask, render_template, request
from flask import make_response
from inference_genre import GPT2
# from flask_bootstrap import Bootstrap

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
app = Flask(__name__)


path = "../trained_models/gpt2_genre3_30.pt"
model = GPT2(load_path = path)

def parse_genre(text):
    parsed = [x.lstrip('[ ') for x in text.split(']')]
    genres = parsed[:-1]
    input_text = parsed[-1]
    return genres, input_text

# syno model
@app.route('/webhook', methods=['GET', 'POST'])
def get_answer():
    print("test1")
    data = request.get_json(silent=True)
    sessionId = data['session']
    text = data['queryResult']['queryText']
    genres, input_text = parse_genre(text)
    print(f"intput Text : {input_text}")
    text = model.generation_fromutil(genres=genres, input_sentence=input_text)[0]
    print(f"output Text : {text}")
    response = fulfilment_text(text)
    response = create_response(response)
    print(response)
    return response


def create_response(response):
    print("test2")
    """ Creates a JSON with provided response parameters """

    # convert dictionary with our response to a JSON string
    res = json.dumps(response, indent=4)

    logger.info(res)

    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'

    # print(r)
    return r


def fulfilment_text(text):
    print("test3")
    "intent parsing해서 해당 response 생성"
    response = {
        "fulfillmentText":
            text
    }
    return response



if __name__ == '__main__':
    print("hi")
    app.run('0.0.0.0', port=8089, threaded=True)
