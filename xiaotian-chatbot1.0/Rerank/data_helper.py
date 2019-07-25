# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 20:06
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_helper.py
# @Software: PyCharm

import sys
import numpy as np


# random.seed(1337)
# np.random.seed(1337)


def load_embedding(dstPath, embedding_size, vocab_file):

    rng = np.random.RandomState(None)
    # pad_embedding = rng.uniform(-0.1, 0.1, size=(1, embedding_size))
    pad_embedding = np.zeros((1, embedding_size))
    unk_embedding = rng.uniform(-0.1, 0.1, size=(1, embedding_size))

    print('uniform_init...')

    embed_dic = {}
    count = 0
    with open(dstPath, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                count +=1
                if count == 1:
                    continue
                line_info = line.strip().split()
                word = line_info[0]
                embedding = [float(val) for val in line_info[1:]]
                embed_dic[word] = embedding
            except:
                print('Error while loading line: {}'.format(line.strip()))

        # print(embed['UNK'].reshape(-1).tolist())
        if 'PAD' not in embed_dic:
            embed_dic['PAD'] = pad_embedding.reshape(-1).tolist()
        if 'UNK' not in embed_dic:
            embed_dic['UNK'] = unk_embedding.reshape(-1).tolist()


    embeddings = []
    with open(vocab_file, 'r', encoding='utf-8') as f1:
        for line in f1:
            word = line.strip().split()[0]
            if word in embed_dic:
                embeddings.append(embed_dic[word])
            else:
                embeddings.append(rng.uniform(-0.1, 0.1, size=(1, embedding_size)).reshape(-1).tolist())
    print("load embedding finish! embedding shape:{}".format(np.shape(embeddings)))
    embeddings = np.array(embeddings)
    return embeddings


class Batch:
    # batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.quest_id = []
        self.ans_id = []
        self.quest = []
        self.ans = []
        self.quest_mask = []
        self.ans_mask = []
        self.label = []


class Iterator(object):
    """
    数据迭代器
    """
    def __init__(self, x):
        self.x = x
        self.sample_num = len(self.x)

    def next_batch(self, batch_size, shuffle=True):
        # produce X, Y_out, Y_in, X_len, Y_in_len, Y_out_len
        if shuffle:
            np.random.shuffle(self.x)
        l = np.random.randint(0, self.sample_num - batch_size + 1)
        r = l + batch_size
        x_part = self.x[l:r]
        return x_part

    def next(self, batch_size, shuffle=False):
        if shuffle:
            np.random.shuffle(self.x)
        l = 0
        while l < self.sample_num:
            r = min(l + batch_size, self.sample_num)
            batch_size = r - l
            x_part = self.x[l:r]
            l += batch_size
            yield x_part


if __name__ == '__main__':

    embeding = '../word2vec/70000-small.txt'
    embeding_size = 200
    vocab_file = '../data/corpus1/project-data/word_vocab.txt'
    load_embedding(embeding, embeding_size, vocab_file)
