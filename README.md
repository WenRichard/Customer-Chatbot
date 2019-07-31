# Customer-Chatbot
#### 中文智能客服机器人demo，包含闲聊和专业问答2个部分，支持自定义组件  
Chinese intelligent customer chatbot Demo, including the gossip and the professional Q&amp;A(FAQ) , support for custom components！

## [smart-chatbot-zero (First version)](https://github.com/WenRichard/Customer-Chatbot/tree/master/smart-chatbot-zero)     
- ### 介绍
一、 本项目由两个部分组成，一是**基于tf-idf检索的召回模型**，二是**基于CNN的精排模型**，本项目将两者融合，构建 召回+排序 的客服聊天机器人。系统支持**闲聊模式**和**FAQ问答模式**，采取的数据分别为小黄鸡闲聊数据集和垂直领域的FAQ问答数据集。该版本为第一版本，速度等其他性能还有待提升，这些工作会在后期陆续上传。根据目前的反馈，系统的难点在于构建一个精度高且耗时短的rerank模型，如果要在工业上使用，需要大改；如果是想要熟悉问题系统的一个整套流程，这个项目百分之百能满足需求。  
  
二、 只有recall阶段的系统可查看：  
[基于tf-idf的问答机器人](https://github.com/WenRichard/QAmodel-for-Retrievalchatbot/tree/master/QAdemo_base1)

-------------------------------------------------
## [xiaotian-chatbot1.0 (Second version)](https://github.com/WenRichard/Customer-Chatbot/tree/master/xiaotian-chatbot1.0) 
- ### 介绍
该项目是在 First version 的基础上进行改进，加入了一些规则  
目前该系统的优点在于：  
一、召回+排序 2个模块互不干扰，便于自定义修改以及维护    
二、系统采取了排序规则优化，提升了检索速度  
三、加入了简单的倒排索引，优化了检索流程  
  
本项目依靠route函数进行问答任务转换，分为 chat模式 和 faq 模式，这样做的目的主要是系统可以根据不同的任务设置不同的情景对话，同时系统将2个语料集分开管理，避免了搜索时间的增加。目前的效果是如果你不输入end终止对话，那么你可以在对话中进行chat模式和faq模式的随意转化，随心所欲！
