# -*- coding: utf-8 -*-
# @Time    : 2019/7/22 9:19
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : predict.py
# @Software: PyCharm


import sys
stdout = sys.stdout
import warnings

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .data_helper import *
from .data_preprocess2 import *
from .model import SiameseQACNN
from .model_utils import *
from .metrics import *
warnings.filterwarnings("ignore")

best_path = "./Rerank/model/bestval"
stopword_file = './stopwordList/stopword.txt'
vocab_file = './data/corpus1/project-data/word_vocab.txt'
max_q_length = 15
max_a_length = 15
batch_size = 32


class NNConfig(object):
    def __init__(self, embeddings):
        self.ans_length = 15
        # 循环数
        self.num_epochs = 100
        # batch大小
        # 输入问题(句子)长度
        self.ques_length = 15
        # 输入答案长度
        self.batch_size = 32
        # 不同类型的filter，对应不同的尺寸
        self.window_sizes = [1, 2, 3, 5]
        # 隐层大小
        self.hidden_size = 128
        self.output_size = 128
        self.keep_prob = 0.5
        # 每种filter的数量
        self.n_filters = 128
        # 词向量大小
        self.embeddings = np.array(embeddings).astype(np.float32)
        # self.embeddings = None
        self.vocab_size = 3258
        self.embedding_size = 300
        # 学习率
        self.learning_rate = 0.001
        # 优化器
        self.optimizer = 'adam'
        self.clip_value = 5
        self.l2_lambda = 0.00001
        # 评测
        self.eval_batch = 100


def test(corpus, config):
    process_data = read_test(corpus, stopword_file, vocab_file)
    test_corpus = load_test_data(process_data, max_q_length, max_a_length)
    # tf.reset_default_graph() 可以防止模型重载时报错
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = SiameseQACNN(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(best_path))

        iterator = Iterator(test_corpus)
        res = []
        for batch_x in iterator.next(config.batch_size, shuffle=False):
            batch_q, batch_a, batch_qmask, batch_amask = zip(*batch_x)
            batch_q = np.asarray(batch_q)
            batch_a = np.asarray(batch_a)
            predictions = sess.run([model.res], feed_dict={model._ques: batch_q,
                                                model._ans: batch_a,
                                                model.dropout_keep_prob: 1.0})
            res.append([i for i in predictions])
        return res


def main(corpus, embedding):
    res = test(corpus, NNConfig(embedding))[0][0]
    res = [i[0] for i in res]
    return res




