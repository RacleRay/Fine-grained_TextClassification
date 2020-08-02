#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   grpc_infer.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import grpc
import numpy as np
import tensorflow as tf
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from data_process import tokenize
from dataset import read_vocab, _tokenize


# 作为远程客户端，通过grpc访问tensorflow serving

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


formatter = Formatter('data/vocab.txt')


def client_gRPC(data):
    # 链接远端服务器
    channel = grpc.insecure_channel('127.0.0.1:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    content_ids, length = formatter.format(data)
    inputs_id_list = np.array([content_ids])

    # 初始化请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'classifier' #指定模型名称
    request.model_spec.signature_name = "predict_labels" #指定模型签名
    request.inputs['text_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(inputs_id_list))
    request.inputs['text_lens'].CopyFrom(tf.contrib.util.make_tensor_proto(length))

    #开始调用远端服务。执行预测任务
    start_time = time.time()
    result = stub.Predict(request)

    #输出预测时间
    print("花费时间: {}".format(time.time()-start_time))

    #解析结果并返回
    result_dict = {}
    for key in result.outputs:
        tensor_proto = result.outputs[key]
        nd_array = tf.contrib.util.make_ndarray(tensor_proto)
        result_dict[key] = nd_array

    return result_dict


if __name__=="__main__":
    inputs = input("输入文本：")
    result = client_gRPC(inputs)
    print(result['predicts'])
