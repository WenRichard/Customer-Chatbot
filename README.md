# Customer-Chatbot
#### 中文智能客服机器人demo，包含闲聊和专业问答2个部分，支持自定义组件  
Chinese intelligent customer chatbot Demo, including the gossip and the professional Q&amp;A(FAQ) , support for custom components！

## Chinese Customer Chatbot(First version)      
## 介绍
一、 本项目由两个部分组成，一是**基于tf-idf检索的召回模型**，二是**基于CNN的精排模型**，本项目将两者融合，构建 召回+排序 的客服聊天机器人。系统支持**闲聊模式**和**FAQ问答模式**，采取的数据分别为小黄鸡闲聊数据集和垂直领域的FAQ问答数据集。该版本为第一版本，速度等其他性能还有待提升，这些工作会在后期陆续上传。目前该系统的优点在于：一、 召回+排序 2个模块互不干扰，便于自定义修改以及维护； 二、 系统采取了排序规则优化，提升了检索速度。根据目前的反馈，系统的难点在于构建一个精度高且耗时短的rerank模型，如果要在工业上使用，需要大改；如果是想要熟悉问题系统的一个整套流程，这个项目百分之百能满足需求。  
  
二、 只有recall阶段的系统可查看：  
[基于tf-idf的问答机器人](https://github.com/WenRichard/QAmodel-for-Retrievalchatbot/tree/master/QAdemo_base1)
  
## Chinese Customer Chatbot(Second version)      
敬请期待~~
