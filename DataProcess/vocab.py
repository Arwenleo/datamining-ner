# 获取词典

from Public.path import path_vocab

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# word to index 词典
def get_w2i(vocab_path = path_vocab):
    w2i = {}
    with open(vocab_path, 'r',encoding='UTF-8') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i


# tag to index 词典
def get_tag2index():
    return {"O": 0,
            "B-BANK": 1, "I-BANK": 2,
            "B-PRODUCT": 3, "I-PRODUCT": 4,
            "B-COMMENTS_N": 5, "I-COMMENTS_N": 6,
            'B-COMMENTS_ADJ':7, 'I-COMMENTS_ADJ':8
            }

if __name__ == '__main__':
    get_w2i()






















