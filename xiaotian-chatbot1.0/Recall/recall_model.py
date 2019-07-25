# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 16:48
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : tmodel.py
# @Software: PyCharm

import pandas as pd
import matplotlib as mpl
import numpy as np
from nltk.probability import FreqDist
import time

from .jiebaSegment import *
from .sentenceSimilarity import SentenceSimilarity

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese

# 设置外部词
seg = Seg()
seg.load_userdict('./userdict/userdict.txt')


def read_corpus1():
    qList = []
    # 问题的关键词列表
    qList_kw = []
    aList = []
    data = pd.read_csv('./data/corpus1/faq/qa_.csv', header=None)
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(t[0])
        qList_kw.append(seg.cut(t[0]))
        aList.append(t[1])
    return qList_kw, qList, aList


def read_corpus2():
    qList = []
    # 问题的关键词列表
    qList_kw = []
    aList = []
    with open('./data/corpus1/chat/chat-small2.txt', 'r', encoding='utf-8') as f2:
        for i in f2:
            t = i.split('\t')
            s1 = ''.join(t[0].split(' '))
            s2 = ''.join(t[1].strip('\n'))
            qList.append(s1)
            qList_kw.append(seg.cut(s1))
            aList.append(s2)

    return qList_kw, qList, aList


def plot_words(wordList):
    fDist = FreqDist(wordList)
    #print(fDist.most_common())
    print("单词总数: ",fDist.N())
    print("不同单词数: ",fDist.B())
    fDist.plot(10)


def invert_idxTable(qList_kw):  # 定一个一个简单的倒排表
    invertTable = {}
    for idx, tmpLst in enumerate(qList_kw):
        for kw in tmpLst:
            if kw in invertTable.keys():
                invertTable[kw].append(idx)
            else:
                invertTable[kw] = [idx]
    return invertTable


def filter_questionByInvertTab(inputQuestionKW, questionList, answerList, invertTable):
    idxLst = []
    questions = []
    answers = []
    for kw in inputQuestionKW:
        if kw in invertTable.keys():
            idxLst.extend(invertTable[kw])
    idxSet = set(idxLst)
    for idx in idxSet:
        questions.append(questionList[idx])
        answers.append(answerList[idx])
    return questions, answers


def main(question, top_k, task='faq'):
    # 读取数据
    if task == 'chat':
        qList_kw, questionList, answerList = read_corpus2()
    else:
        qList_kw, questionList, answerList = read_corpus1()

    """简单的倒排索引"""
    # 计算倒排表
    invertTable = invert_idxTable(qList_kw)
    inputQuestionKW = seg.cut(question)

    # 利用关键词匹配得到与原来相似的问题集合
    questionList_s, answerList_s = filter_questionByInvertTab(inputQuestionKW, questionList, answerList,
                                                              invertTable)
    # 初始化模型
    ss = SentenceSimilarity(seg)
    ss.set_sentences(questionList_s)
    ss.TfidfModel()  # tfidf模型
    # ss.LsiModel()         # lsi模型
    # ss.LdaModel()         # lda模型
    question_k = ss.similarity_k(question, top_k)
    return question_k, questionList_s, answerList_s


if __name__ == '__main__':
    # 设置外部词
    seg = Seg()
    seg.load_userdict('./userdict/userdict.txt')
    # 读取数据
    List_kw, questionList, answerList = read_corpus1()
    # 初始化模型
    ss = SentenceSimilarity(seg)
    ss.set_sentences(questionList)
    ss.TfidfModel()         # tfidf模型
    # ss.LsiModel()         # lsi模型
    # ss.LdaModel()         # lda模型

    while True:
        question = input("请输入问题(q退出): ")
        if question == 'q':
            break
        time1 = time.time()
        question_k = ss.similarity_k(question, 5)
        print("亲，我们给您找到的答案是： {}".format(answerList[question_k[0][0]]))
        for idx, score in zip(*question_k):
            print("same questions： {},                score： {}".format(questionList[idx], score))
        time2 = time.time()
        cost = time2 - time1
        print('Time cost: {} s'.format(cost))








