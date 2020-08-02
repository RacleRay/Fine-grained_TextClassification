#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   app.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

# tf serving预测结果没有网页显示界面，此处通过flask web页面展示结果。
# 只使用计算结果时，可以直接通过 REST api 或者 grpc 远程连接模型计算。

import requests
from flask import Flask, request, render_template
import numpy as np
from data_process import tokenize
from dataset import read_vocab, _tokenize


# text processing
def preprocess(text):
    return tokenize(text)

# formating
class Formatter:
    def __init__(self, vocab_file, max_len=1200):
        self.max_len = max_len
        self.vocab, self.w2i = read_vocab(vocab_file)
        self.tag_l2i = {"1": 0, "0": 1, "-1": 2, "-2": 3}
        self.tag_i2l = {v: k for k, v in self.tag_l2i.items()}

    def format(self, text):
        content = preprocess(text)
        content = _tokenize(content,
                            self.w2i,
                            self.max_len,
                            False,
                            True)
        length = [len(content)]
        return content, length


app = Flask(__name__)
formatter = Formatter('data/vocab.txt')

SERVER_URL = 'http://localhost:8501/v1/models/classifier:predict'
key_list = [
    "location_traffic_convenience",
    "location_distance_from_business_district",
    "location_easy_to_find",
    "service_wait_time",
    "service_waiters_attitude",
    "service_parking_convenience",
    "service_serving_speed",
    "price_level",
    "price_cost_effective",
    "price_discount",
    "environment_decoration",
    "environment_noise",
    "environment_space",
    "environment_cleaness",
    "dish_portion",
    "dish_taste",
    "dish_look",
    "dish_recommendation",
    "others_overall_experience",
    "others_willing_to_consume_again"
]
value_dict = {0: "好评", 1: "中评", 2: "差评", 3: "未提及"}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def inference():
    '''
    For rendering results on HTML GUI
    '''
    # 处理输入
    content = request.form.get('content')
    content_ids, length = formatter.format(content)
    inputs_id_list = [content_ids]

    # 从tf serving获取深度模型预测结果
    predict_request = '{"signature_name": "predict_labels", ' + \
            '"inputs": {' + f'"text_ids": {inputs_id_list}, "text_lens": {length}' + "}}"
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

    prediction = response.json()['outputs']['predicts']
    # logits = response.json()['outputs']['logits']

    prediction = {key: value_dict[np.argmax(each_label)] for key, each_label in zip(key_list,
                                                                                 prediction[0])}
    print(content)
    print(prediction)
    return render_template('index.html', results=prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000, debug=False)
    # inference("这家餐厅环境挺不错的")