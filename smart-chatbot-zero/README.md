# 中文智能客服聊天机器人 
## Chinese Customer Chatbot(first version)
    
## The English version of the project is introduced behind the Chinese introduction, Please read patiently ！    

## 介绍
本项目由两个部分组成，一是**基于tf-idf检索的召回模型**，二是**基于CNN的精排模型**，本项目将两者融合，构建 召回+排序 的客服聊天机器人。系统支持**闲聊模式**和**FAQ问答模式**，采取的数据分别为小黄鸡闲聊数据集和垂直领域的FAQ问答数据集。该版本为第一版本，速度等其他性能还有待提升，这些工作会在后期陆续上传。目前该系统的优点在于：一、 召回+排序 2个模块互不干扰，便于自定义修改以及维护； 二、 系统采取了排序规则优化，提升了检索速度。根据目前的反馈，系统的难点在于构建一个精度高且耗时短的rerank模型，如果要在工业上使用，需要大改；如果是想要熟悉问题系统的一个整套流程，这个项目百分之百能满足需求。

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
     
 ## 实验结果
 ### 1.1 闲聊模式：只有recall阶段  
 ![闲聊图]( https://github.com/WenRichard/Customer-Chatbot/raw/master/smart-chatbot-zero/data/corpus1/chat/image/recall1.png "闲聊图") 
 ![闲聊图]( https://github.com/WenRichard/Customer-Chatbot/raw/master/smart-chatbot-zero/data/corpus1/chat/image/recall2.png "闲聊图") 
 ### 1.2 闲聊模式：recall+rerank阶段 
 ![闲聊图]( https://github.com/WenRichard/Customer-Chatbot/raw/master/smart-chatbot-zero/data/corpus1/chat/image/rerank1.png "闲聊图") 
   
 ### 2.1 FAQ问答模式：只有recall阶段 
 ![FAQ问答图]( https://github.com/WenRichard/Customer-Chatbot/raw/master/smart-chatbot-zero/data/corpus1/chat/image/faq.png "闲聊图") 
   
 ## 3.1 聊天结束
 ![结束图]( https://github.com/WenRichard/Customer-Chatbot/raw/master/smart-chatbot-zero/data/corpus1/chat/image/chat-end.png "闲聊图") 
