# -*- coding: utf-8 -*-
# @Time    : 2019/5/25 16:08
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : QACNN_CAIL2.py
# @Software: PyCharm


import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import os
import tqdm
import sys
from copy import deepcopy
stdout = sys.stdout

from .data_helper import *
from .data_preprocess2 import *
from .model import SiameseQACNN
from .model_utils import *
from .metrics import *
from sklearn.metrics import accuracy_score

# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
timestamp = str(int(time.time()))
fh = logging.FileHandler('./log/log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
# logger.addHandler(ch)


class NNConfig(object):
    def __init__(self, embeddings):
        self.ans_length = 315
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


# TODO
def test(corpus, config):
    with tf.Session() as sess:
        model = SiameseQACNN(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(best_path))


def train(train_corpus, test_corpus, config):
    iterator = Iterator(train_corpus)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    # 定义计算图
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
        with tf.Session(config=session_conf) as sess:
            # training
            print('Start training and evaluating ...')

            outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
            print("Writing to {}\n".format(outDir))

            trainSummaryDir = os.path.join(outDir, "train")
            trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

            evalSummaryDir = os.path.join(outDir, "eval")
            evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

            model = SiameseQACNN(config)
            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            ckpt = tf.train.get_checkpoint_state(save_path)

            print('Configuring TensorBoard and Saver ...')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reloading model parameters ...')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Created new model parameters ...')
                sess.run(tf.global_variables_initializer())

            # count trainable parameters
            total_parameters = count_parameters()
            print('Total trainable parameters : {}'.format(total_parameters))

            def trainStep(batch_q, batch_a, batchY):
                _, loss, summary, step, predictions = sess.run(
                    [model.train_op, model.total_loss, model.summary_op, model.global_step, model.res],
                    feed_dict={model._ques: batch_q,
                               model._ans: batch_a,
                               model._y: label,
                               model.dropout_keep_prob: config.keep_prob})
                predictions = [1 if i >= 0.5 else 0 for i in predictions]
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
                trainSummaryWriter.add_summary(summary, step)

                return loss, acc, prec, recall, f_beta

            def devStep(corpus):
                iterator = Iterator(corpus)
                dev_Loss = []
                dev_Acc = []
                dev_Prec = []
                dev_Recall = []
                dev_F_beta = []
                for batch_x in iterator.next(config.batch_size, shuffle=False):
                    batch_q, batch_a, batch_qmask, batch_amask, label = zip(*batch_x)
                    batch_q = np.asarray(batch_q)
                    batch_a = np.asarray(batch_a)
                    loss, summary, step, predictions = sess.run(
                        [model.total_loss, model.summary_op, model.global_step, model.res],
                        feed_dict={model._ques: batch_q,
                                   model._ans: batch_a,
                                   model._y: label,
                                   model.dropout_keep_prob: 1.0})
                    predictions = [1 if i >= 0.5 else 0 for i in predictions]
                    acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=label)
                    dev_Loss.append(loss)
                    dev_Acc.append(acc)
                    dev_Prec.append(prec)
                    dev_Recall.append(recall)
                    dev_F_beta.append(f_beta)
                    evalSummaryWriter.add_summary(summary, step)

                return mean(dev_Loss), mean(dev_Acc), mean(dev_Recall), mean(dev_Prec), mean(dev_F_beta)

            best_acc = 0.0
            for epoch in range(config.num_epochs):
                train_time1 = time.time()
                print("----- Epoch {}/{} -----".format(epoch + 1, config.num_epochs))
                train_Loss = []
                train_Acc = []
                train_Prec = []
                train_Recall = []
                train_F_beta = []
                for batch_x in iterator.next(config.batch_size, shuffle=True):
                    batch_q, batch_a, batch_qmask, batch_amask, label = zip(*batch_x)
                    batch_q = np.asarray(batch_q)
                    batch_a = np.asarray(batch_a)
                    train_loss, train_acc, train_prec, train_recall, train_f_beta = trainStep(batch_q, batch_a, label)
                    train_Loss.append(train_loss)
                    train_Acc.append(train_acc)
                    train_Prec.append(train_prec)
                    train_Recall.append(train_recall)
                    train_F_beta.append(train_f_beta)
                print(
                    "---epoch %d  -- test loss %.3f -- test acc %.3f -- test recall %.3f -- test precision %.3f"
                           "-- test f_beta %.3f".
                    format(epoch, np.mean(train_Loss), np.mean(train_Acc), np.mean(train_Recall),
                           np.mean(train_Prec), np.mean(train_F_beta)))
                # 写入文件中去
                logger.info("---epoch %d  -- test loss %.3f -- test acc %.3f -- test recall %.3f -- test precision %.3f"
                           "-- test f_beta %.3f".
                      format(epoch, np.mean(train_Loss), np.mean(train_Acc), np.mean(train_Recall),
                             np.mean(train_Prec), np.mean(train_F_beta)))
                train_time2 = time.time()
                print('train time cost: {}'.format(train_time2-train_time1))

                test_time1 = time.time()
                test_loss, test_acc, test_recall, test_prec, test_f_beta = devStep(test_corpus)
                print("---epoch %d  -- test loss %.3f -- test acc %.3f -- test recall %.3f -- test precision %.3f"
                           "-- test f_beta %.3f" % (
                           epoch + 1, test_loss, test_acc, test_recall, test_prec, test_f_beta))
                # 写入文件中去
                logger.info("\nTest:")
                logger.info("---epoch %d  -- test loss %.3f -- test acc %.3f -- test recall %.3f -- test precision %.3f"
                            "-- test f_beta %.3f" % (epoch + 1, test_loss, test_acc, test_recall, test_prec, test_f_beta))

                test_time2 = time.time()
                print('test time cost: {}'.format(test_time2 - test_time1))

                checkpoint_path = os.path.join(save_path, 'acc{:.3f}_{}.ckpt'.format(test_acc, epoch + 1))
                bestcheck_path = os.path.join(best_path, 'acc{:.3f}_{}.ckpt'.format(test_acc, epoch + 1))
                saver.save(sess, checkpoint_path, global_step=epoch)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_saver.save(sess, bestcheck_path, global_step=epoch)
            logger.info("\nBest and Last:")
            logger.info('--- best_acc %.3f '% (best_acc))


def main():

    embedding = load_embedding(embeding, embeding_size, vocab_file)
    preprocess_data1 = preprocess(train_file)
    preprocess_data2 = preprocess(test_file)

    train_data = read_train(preprocess_data1, stopword_file, vocab_file)
    test_data = read_train(preprocess_data2, stopword_file, vocab_file)
    train_corpus = load_train_data(train_data, max_q_length, max_a_length)
    test_corpus = load_train_data(test_data, max_q_length, max_a_length)

    config = NNConfig(embedding)
    config.ques_length = max_q_length
    config.ans_length = max_a_length
    # config.embeddings = embedding
    train(deepcopy(train_corpus), test_corpus, config)


if __name__ == '__main__':
    save_path = "./model/checkpoint"
    best_path = "./model/bestval"
    train_file = '../data/corpus1/raw/train.txt'
    test_file = '../data/corpus1/raw/test.txt'
    stopword_file = '../stopwordList/stopword.txt'
    embeding = '../word2vec/70000-small.txt'
    vocab_file = '../data/corpus1/project-data/word_vocab.txt'
    max_q_length = 15
    max_a_length = 15
    embeding_size = 200
    main()


