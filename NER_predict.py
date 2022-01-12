# -*- coding: utf-8 -*-
from DataProcess.process_data import DataProcess
from DataProcess.vocab import *
from Model import BERT_BILSTM_CRF as BERTBILSTMCRF
import numpy as np
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy,crf_viterbi_accuracy
from keras_contrib.losses import crf_loss
from keras_bert import get_custom_objects
# from keras.preprocessing.sequence import pad_sequences


max_len = 100
model_path = './Model/trained_model.h5'

custom_objects = get_custom_objects()
my_objects = {"CRF": CRF, 'crf_loss': crf_loss,'crf_viterbi_accuracy': crf_viterbi_accuracy}
custom_objects.update(my_objects)
model = load_model(model_path, custom_objects=custom_objects)

model.summary()
try:
    tag_to_ind = get_tag2index()
    word_to_ind = get_w2i()
    num_to_tag = dict(zip(tag_to_ind.values(), tag_to_ind.keys()))

    unk_flag = '[UNK]'
    pad_flag = '[PAD]'
    cls_flag = '[CLS]'
    sep_flag = '[SEP]'
    unk_index = word_to_ind.get(unk_flag, 101)
    pad_index = word_to_ind.get(pad_flag, 1)
    cls_index = word_to_ind.get(cls_flag, 102)
    sep_index = word_to_ind.get(sep_flag, 103)

    data_ids = []
    data_types = []

    user_inp = "浦发银行的东西太贵了,昨天林大强说还想买"
    input_list = [word for word in user_inp]
    line_data_ids = []
    for w in input_list:
        w_index = word_to_ind.get(w, unk_index)
        line_data_ids.append(w_index)

    # input_wordind = [[word_to_ind[word] for word in input_list]]
    # input_data_id = pad_sequences(maxlen=max_len, sequences=input_wordind, padding='post')

    line_data_ids = [cls_index] + line_data_ids + [sep_index]
    pad_num = max_len - len(line_data_ids)
    line_data_ids = [pad_index] * pad_num + line_data_ids
    data_ids.append(np.array(line_data_ids))

    line_data_types = [0] * max_len
    data_types.append(np.array(line_data_types))

    input_data = [np.array(data_ids), np.array(data_types)]
    pre = model.predict(input_data)
    pre_ = np.argmax(pre, axis=2)
    # print(pre_)

    ner_tag = []
    input_len = len(input_list)
    for i in range(0, input_len):
        j = max_len-input_len-1+i
        ner_tag.append(pre_[0][j])
    print(ner_tag)

    ner_ans = [num_to_tag[i] for i in ner_tag]

    print(input_list)
    print(ner_ans)

except:
    print("Recognition error. Please input again!")
















