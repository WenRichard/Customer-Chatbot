# -*- coding: utf-8 -*-
# @Time    : 2019/5/27 19:10
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_preprocess3.py
# @Software: PyCharm


import json
import jieba
import numpy as np
import pickle
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split


ques_len = 315
ans_len = 315


def preprocess(data_file):
    all_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i in f:
            id, s1, s2, label = i.split('|')
            s1 = ''.join(s1.strip().split(" "))
            s2 = ''.join(s2.strip().split(" "))
            label = int(label.strip('\n'))
            all_data.append((s1, s2, label))
    return all_data


def load_stopword(file):
    stopwords = set()
    file_obj = open(file, 'r', encoding='utf-8')
    while True:
        line = file_obj.readline()
        line = line.strip('\r\n')
        if not line:
            break
        stopwords.add(line)
    return stopwords


def cut(sentence, stopwords, stopword=True, cut_all=False):
    # 加载外部词典
    jieba.load_userdict('./userdict/userdict.txt')
    seg_list = jieba.cut(sentence, cut_all)
    results = []
    for seg in seg_list:
        if stopword and seg in stopwords:
            continue
        results.append(seg)
    return results


def read_train(preprocess_data, stopword_file,  vocab_file):
    stopwords = load_stopword(stopword_file)
    all_data = []
    word_vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for i in f:
            word, idx = i.strip().split()
            word_vocab[word] = idx
    for i in preprocess_data:
        s1, s2, label = i

        s1_seg = cut(s1, stopwords)
        s2_seg = cut(s2, stopwords)
        A = [word_vocab.get(i, 1) for i in s1_seg]
        B = [word_vocab.get(i, 1) for i in s2_seg]
        all_data.append((A, B, label))
    return all_data


# 线上得到[(query, s1), (query, s2), ...]的列表
def read_test(list_data, stopword_file,  vocab_file):
    stopwords = load_stopword(stopword_file)
    all_data = []
    word_vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for i in f:
            word, idx = i.strip().split()
            word_vocab[word] = idx
    for i in list_data:
        s1, s2 = i
        s1_seg = cut(s1, stopwords)
        s2_seg = cut(s2, stopwords)
        A = [word_vocab.get(i, 1) for i in s1_seg]
        B = [word_vocab.get(i, 1) for i in s2_seg]
        all_data.append((A, B))
    return all_data


def load_train_data(transformed_corpus, ques_len, ans_len):
    """
        load train data
        """
    pointwise_corpus = []
    for sample in transformed_corpus:
        q, a, label = sample
        q_pad, q_len = padding(q, ques_len)
        a_pad, a_len = padding(a, ans_len)
        pointwise_corpus.append((q_pad, a_pad, q_len, a_len, label))
    return pointwise_corpus


def load_test_data(transformed_corpus, ques_len, ans_len):
    """
        load train data
        """
    pointwise_corpus = []
    for sample in transformed_corpus:
        q, a = sample
        q_pad, q_len = padding(q, ques_len)
        a_pad, a_len = padding(a, ans_len)
        pointwise_corpus.append((q_pad, a_pad, q_len, a_len))
    return pointwise_corpus


def padding(sent, sequence_len):
    """
     convert sentence to index array
    """
    if len(sent) > sequence_len:
        sent = sent[:sequence_len]
    padding = sequence_len - len(sent)
    sent2idx = sent + [0]*padding
    return sent2idx, len(sent)


def pointwise(file, file2):
    with open(file, 'r', encoding='utf-8') as f, open(file2, 'w', encoding='utf-8') as f2:
        for i in f:
            s1, s2, s3 = i.strip().split('\t')
            f2.write(s1 + '\t' + s2 + '\t' + str(1) + '\n')
            f2.write(s1 + '\t' + s3 + '\t' + str(0) + '\n')



if __name__ == '__main__':
    stopwords = '../stopwordList/stopword.txt'
    word_vocab_file = '../data/corpus1/project-data/word_vocab.txt'
    datafile = '../data/corpus1/raw/train.txt'

    train_data = preprocess(datafile)
    read_train(train_data, stopwords, word_vocab_file)

    # load_train_data(train_pkl, 315, 315)

    # pointwise(in_file, pointwise_file)
    # gen_test(pointwise_file, stopwords, word_vocab_file)