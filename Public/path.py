import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前地址

# bert预训练模型词表
path_vocab = os.path.join(current_dir, '../data/vocab/vocab.txt')

path_mydata_dir = os.path.join(current_dir, '../data/mydata/')

# bert 预训练文件
path_bert_dir = os.path.join(current_dir, '../data/chinese_L-12_H-768_A-12/')

path_log_dir = os.path.join(current_dir, "../log")

