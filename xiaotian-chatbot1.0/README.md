# 聊天机器人小天1.0（XiaoTian Chatbot1.0）
    
### The introduction of the project in english will be introduced after the chinese introduction, Please read patiently ！    
## 介绍
本项目由两个部分组成，一是基于tf-idf检索的召回模型，二是基于CNN的精排模型，本项目将两者融合，构建 召回+排序 的客服聊天机器人。系统支持闲聊模式和FAQ问答模式，采取的数据分别为小黄鸡闲聊数据集和垂直领域的FAQ问答数据集。该聊天机器人的版本为小天1.0，速度提升的小天2.0版本会在后期陆续上传。  
  
目前该系统的优点在于：  
一、 召回+排序 2个模块互不干扰，便于自定义修改以及维护；  
二、系统采取了排序规则优化，提升了检索速度。  
三、加入了简单的倒排索引，优化了检索流程。  
  
本项目依靠route函数进行问答任务转换，分为 chat模式 和 faq 模式，这样做的目的主要是系统可以根据不同的任务设置不同的情景对话，同时系统将2个语料集分开管理，避免了搜索时间的增加。目前的效果是如果你不输入end终止对话，那么你可以在对话中进行chat模式和faq模式的随意转化，随心所欲！  
## 环境配置  
  
  Python版本为3.6  
  tensorflow版本为1.13  
  
### 目录说明
    
    Recall
      - recall_model.py  
    Rerank
      - data_preprocess1.py  根据语料生成embedding词表文件
      - data_preprocess2.py  加载训练集，测试集等信息
      - data_helper.py  加载batch信息
      - model.py  深度学习模型构建
      - qacnn.py  支持模型训练，调参
      - rerank_model.py  加载训练好的模型，进行预测
      
### 使用说明

    - recall_model.py 不需要训练，需保证输入正确语料，根据不同的任务调用不同的语料集，支持单独测试
    - qacnn.py  先训练深度学习模型，得到checkpoint等文件，支持训练和测试，这一步确保得到一个高效的rerank模型
    - rerank_model.py 不支持单独使用，是系统的调用文件
    - qa-control.py  
      系统主要控制文件，集成 recall + rerank 2个部分
      需设定2个阈值，max_sim 和 min_sim，分为3种情况,缩减响应时间：
          1. recall_score < min_sim，说明问答库数量少或者问句噪声大，需要复查分析
          2. min_sim < recall_score < max_sim， 进行 recall + rerank
          3. recall_score > max_sim，只进行recall，直接得出答案

## Introduction
The project consists of two parts, one is **recall model based on the tf-idf**, and the other is **rerank model based on CNN**, this project combines the two to build a "recall + rerank customer chatbot". The system supports **chat mode** and **FAQ question mode**. datas are based on the Xiao Huang ji dataset and the vertical field FAQ data set. This version is the first version, and other performances such as speed have yet to be improved. It is worth noting that some excellent jobs will be uploaded later. At present, the advantages of the system are as follows: First, the recall + sorting two modules do not interfere with each other, which is convenient for custom modification and maintenance; Second, the system adopts the optimization of the sorting rules and improves the retrieval speed. According to the current feedback, the difficulty of the system lies in constructing a The high-precision and short time-consuming re-ranking model. the system needs to be changed if it is will be used in industry; however, if you just wants to be familiar with the QA system, this project can meet 100% of the demand.

## configuration  
  
  Python 3.6  
  tensorflow 1.13  
  
### Directory description
    
    Recall
      - recall_model.py  
    Rerank
      - data_preprocess1.py   # Generate embedding vocabulary files based on corpus
      - data_preprocess2.py   # Load training datasets, test datasets, etc.
      - data_helper.py   # load batch of data
      - model.py   # construct deep model
      - qacnn.py   # Support model training, coordinate
      - rerank_model.py   # Load trained models and make predictions
      
### How to use

    - recall_model.py 
    - qacnn.py  
      $ No training is required, it is necessary to ensure that the correct corpus is input, different corpus sets are   called according to different tasks, and individual testing is supported.
    - rerank_model.py 
      $ No support separate use, is the system call file
    - qa-control.py  
      $ System main control file, integrated recall + rerank 2 parts
      $ Two thresholds, max_sim and min_sim, are required to be divided into three cases to reduce the response time:
          1. recall_score < min_sim，
          Explain that the number of question and answer libraries is small or the question is noisy, and it needs to be reviewed and analyzed.
          2. min_sim < recall_score < max_sim， 
          just recall + rerank
          3. recall_score > max_sim，
          Just recall model and get the answer directly
 
## 实验结果 $ Experimental result
###  FAQ问答模式：只有recall阶段 (faq model, only consists recall)  
![FAQ问答图]( https://github.com/WenRichard/Customer-Chatbot/raw/master/smart-chatbot-zero/data/corpus1/chat/image/faq.png "闲聊图") 
   
如果想要获取更详细的效果图，可以看  
[问答系统实践（二）构建聊天机器人小天1.0](https://zhuanlan.zhihu.com/p/75108562)  
对于recall模型的具体介绍在：  
[问答系统实践（一）：中文检索式问答机器人初探](https://zhuanlan.zhihu.com/p/61513395)  
[基于tf-idf的问答机器人](https://github.com/WenRichard/QAmodel-for-Retrievalchatbot/tree/master/QAdemo_base1)  

 --------------------------------------------------------------
**如果觉得我的工作对您有帮助，请不要吝啬右上角的小星星哦！欢迎Fork和Star！也欢迎一起建设这个项目！**    
**有时间就会更新问答相关项目，有兴趣的同学可以follow一下**  
**留言请在Issues或者email xiezhengwen2013@163.com**
