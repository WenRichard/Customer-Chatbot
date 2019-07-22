# -*- coding: utf-8 -*-
# @Time    : 2019/7/21 16:48
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : tmodel.py
# @Software: PyCharm


import time
from Rerank.data_helper import *
from Recall import recall_model
from Rerank import rerank_model


class SmartQA:
    def __init__(self):
        self.top_k = 5
        self.min_sim = 0.25
        self.max_sim = 0.95
        self.embeding_size = 200
        self.vocab_file = './data/corpus1/project-data/word_vocab.txt'
        self.embed_file = './word2vec/70000-small.txt'
        self.embedding = load_embedding(self.embed_file, self.embeding_size, self.vocab_file)

    '''问答主函数'''
    # 分为 recall + rerank 2个部分
    # 设定2个阈值，max_sim 和 min_sim，分为3种情况,缩减响应时间：
    #   1. recall_score < min_sim，说明问答库数量少或者问句噪声大，需要复查分析
    #   2. min_sim < recall_score < max_sim， 进行 recall + rerank
    #   3. recall_score > max_sim，只进行recall，直接得出答案
    # TODO 把问答对字典化，提升寻址速度
    def search_main(self, question, task='faq'):
        # 粗排
        candi_questions, questionList, answerList = recall_model.main(question, self.top_k, task)
        answer_dict = {}
        corpus = []
        indxs = []
        matchmodel_simscore = []
        sim_questions = []

        for indx, candi in zip(*candi_questions):
            # 如果在粗排阶段就已经找到了非常相似的问题，则马上返回这个答案,终止循环
            if candi > self.max_sim:
                indxs.append(indx)
                break
            else:
                # 如果召回的数据噪声很大，不做精确匹配
                # TODO 把噪声大的数据生成一个文件，复查分析
                if candi < self.min_sim:
                    continue
                matchmodel_simscore.append(candi)
                corpus.append((question, questionList[indx]))
                indxs.append(indx)
                sim_questions.append(questionList[indx])
        if len(indxs) == 1:
            sim = [questionList[indx] for indx, candi in zip(*candi_questions)]
            return answerList[indxs[0]], sim
        else:
            # 精确匹配
            if len(indxs) != 0:
                deepmodel_simscore = rerank_model.main(corpus, self.embedding)
            else:
                return '您好，暂时无法解决这个问题，请您稍等，正在为您转接人工客服.....'
            final = list(zip(indxs, matchmodel_simscore, deepmodel_simscore))
            for id, score1, score2 in final:
                final_score = (score1 + score2) / 2
                answer_dict[id] = final_score
            if answer_dict:
                answer_dict = sorted(answer_dict.items(), key=lambda asd: asd[1], reverse=True)
                final_answer = answerList[answer_dict[0][0]]
            else:
                final_answer = '您好，暂时无法解决这个问题，请您稍等，正在为您转接人工客服....'
            return final_answer, sim_questions


if __name__ == "__main__":
    handler = SmartQA()
    print('hello，我是您的助理小天，闲聊请输入chat，咨询家居问题请输入faq，如果要结束对话请输入 end 哦！')
    question = input('task:')
    # 进入闲聊模式
    if question == 'chat':
        print('hello， 我是人见人爱的小天哦')
        while(1):
            question = input()
            if question == 'end':
                print('byebye~ 小天期待和你下次再见哦！')
                break
            s1 = time.time()
            try:
                final_answer, sim_questions = handler.search_main(question, 'chat')
                s2 = time.time()
                print('', final_answer)
                print('time cost:{} sec'.format(s2 - s1))
            except:
                print('不太理解你说的话哦')
    # 进入任务问答模式
    else:
        while(1):
            question = input('question: ')
            if question == 'end':
                print('byebye~ 小天期待和你下次再见哦！')
                break
            s1 = time.time()
            try:
                final_answer, sim_questions = handler.search_main(question)
                s2 = time.time()
                print('answers:', final_answer)
                print('您好， 您可能对以下问题感兴趣！')
                i = 0
                for j in sim_questions:
                    if j == question:
                        continue
                    i += 1
                    print("{}.{}".format(i, j))
                print('time cost:{} sec'.format(s2 - s1))
            except:
                print('请您把问题说的详细一点哦')
