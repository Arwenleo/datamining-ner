#!/usr/bin/env python
# coding: utf-8

import os
from Public.path import path_mydata_dir
# current_dir = os.path.dirname(os.path.abspath(__file__))
# path_mydata_dir = os.path.join(current_dir, '../data/mydata/')


def mydata_preprocess(split_rate: float = 0.8) -> None:
    '''ignore_exist: bool = False'''

    path = os.path.join(path_mydata_dir, "all_2.txt")
    path_train = os.path.join(path_mydata_dir, "train.txt")
    path_test = os.path.join(path_mydata_dir, "test.txt")
    print("Open the preprocessed data")

    texts = []
    with open(path, 'r', encoding='UTF-8') as f:
        # count = 0
        for l in f.readlines():
            # count += 1
            # if(len(l) == 1):
            #     print(1)
            #     print(count)
            texts.append(l)

    split_index = int(len(texts) * split_rate)
    # print(split_index)
    train_texts = texts[:split_index]
    test_texts = texts[split_index:]

    train_texts[-1] = train_texts[-1].strip('\n')
    test_texts[-1] = test_texts[-1].strip('\n')

    with open(path_train, 'w', encoding='UTF-8') as f:
        f.write("".join(train_texts))
    with open(path_test, 'w', encoding='UTF-8') as f:
        f.write("".join(test_texts))
    print("Split the data successfully!")



if __name__ == '__main__':
    mydata_preprocess()



