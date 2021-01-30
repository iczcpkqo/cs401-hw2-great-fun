#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import nltk
# nltk.download()

####################################
import re
import os
import pandas as pd
from gensim import corpora, models
import numpy as np
from nltk.corpus import stopwords


# In[195]:


#读取文件，提取字段
file_path="./news-data/"
fileList = os.listdir(file_path)

file_List=[]
SOURCE_List=[]
AGENT_List=[]
GOAL_List=[]
DATA_List=[]
METHODS_List=[]
RESULTS_List=[]
ISSUES_List=[]
SCORE_List=[]
COMMENTS_List=[]

for f in fileList:
    print(file_path+f)
    with open(file_path+f,encoding='iso-8859-1') as file_obj:
        content = file_obj.read()
        file_List.append(f)
        SOURCE ="".join(re.findall('SOURCE(.*?)AGENT', content, re.S)).replace("\n","").replace("==","")
    #     print(SOURCE)
        SOURCE_List.append(SOURCE)
        AGENT = "".join(re.findall('AGENT(.*?)GOAL', content, re.S)).replace("\n","").replace("==","")
#         print(AGENT)
        AGENT_List.append(AGENT)
        GOAL = "".join(re.findall('GOAL(.*?)DATA', content, re.S)).replace("\n","").replace("==","")
#         print(GOAL)
        GOAL_List.append(GOAL)
        DATA = "".join(re.findall('DATA(.*?)METHODS', content, re.S)).replace("\n","").replace("==","")
#         print(DATA)
        DATA_List.append(DATA)
        METHODS = "".join(re.findall('METHODS(.*?)RESULTS', content, re.S)).replace("\n","").replace("==","")
#         print(METHODS)
        METHODS_List.append(METHODS)
        RESULTS = "".join(re.findall('RESULTS(.*?)ISSUES', content, re.S)).replace("\n","").replace("==","")
#         print(RESULTS)
        RESULTS_List.append(RESULTS)
        ISSUES = "".join(re.findall('ISSUES(.*?)SCORE', content, re.S)).replace("\n","").replace("==","")
#         print(ISSUES)
        ISSUES_List.append(ISSUES)
        SCORE = "".join(re.findall('SCORE(.*?)COMMENTS', content, re.S)).replace("\n","").replace("==","")
#         print(SCORE)
        SCORE_List.append(SCORE)
        COMMENTS = "".join(re.findall('COMMENTS(.+)', content, re.S)).replace("\n","").replace("==","")
#         print(COMMENTS)
        COMMENTS_List.append(COMMENTS)


# In[196]:


#构造为df
data=pd.DataFrame({"file":file_List,"SOURCE":SOURCE_List,"AGENT":AGENT_List,"GOAL":GOAL_List,"DATA":DATA_List,"METHODS":METHODS_List,"RESULTS":RESULTS_List,"ISSUES":ISSUES_List,"SCORE":SCORE_List,"COMMENTS":COMMENTS_List})


# In[197]:


data


# In[198]:


list_stopWords=list(set(stopwords.words('english')))


# In[199]:


#对要聚类的字段做清洗，大写转小写，只保留单词，去除多个空格
data["GOAL_clean"]=data.apply(lambda x:re.sub('\s+', ' ',re.sub('[^a-z ]+', ' ',x["GOAL"].lower())),axis=1)
data["DATA_clean"]=data.apply(lambda x:re.sub('\s+', ' ',re.sub('[^a-z ]+', ' ',x["DATA"].lower())),axis=1)
data["METHODS_clean"]=data.apply(lambda x:re.sub('\s+', ' ',re.sub('[^a-z ]+', ' ',x["METHODS"].lower())),axis=1)
data["RESULTS_clean"]=data.apply(lambda x:re.sub('\s+', ' ',re.sub('[^a-z ]+', ' ',x["RESULTS"].lower())),axis=1)
data["ISSUES_clean"]=data.apply(lambda x:re.sub('\s+', ' ',re.sub('[^a-z ]+', ' ',x["ISSUES"].lower())),axis=1)
data["COMMENTS_clean"]=data.apply(lambda x:re.sub('\s+', ' ',re.sub('[^a-z ]+', ' ',x["COMMENTS"].lower())),axis=1)


# In[200]:


data


# In[201]:


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.cluster import KMeans


# In[202]:


# #遍历要聚类的列
# cluster_cols=["GOAL_clean","DATA_clean","METHODS_clean","RESULTS_clean","ISSUES_clean","COMMENTS_clean"]
# cluster_num=[3,3,3,3,3,3]
# for i in range(len(cluster_cols)):
#     #将文本中的词语转换为词频矩阵
#     vectorizer = CountVectorizer()
#     #计算个词语出现的次数
#     X = vectorizer.fit_transform(data[cluster_cols[i]])
#     #类调用
#     transformer = TfidfTransformer()
#     #将词频矩阵X统计成TF-IDF值
#     tfidf = transformer.fit_transform(X)
#     #查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
#     clf = KMeans(n_clusters=cluster_num[i])
#     s = clf.fit(tfidf.toarray())
#     data[cluster_cols[i]+"_cluster_id"]=clf.labels_


# In[203]:


#去除停用词
data["GOAL_clean"]=data.apply(lambda x:[w for w in x["GOAL_clean"].split(" ") if not w in list_stopWords],axis=1)


# In[204]:


data["DATA_clean"]=data.apply(lambda x:[w for w in x["DATA_clean"].split(" ") if not w in list_stopWords],axis=1)
data["METHODS_clean"]=data.apply(lambda x:[w for w in x["METHODS_clean"].split(" ") if not w in list_stopWords],axis=1)
data["RESULTS_clean"]=data.apply(lambda x:[w for w in x["RESULTS_clean"].split(" ") if not w in list_stopWords],axis=1)
data["ISSUES_clean"]=data.apply(lambda x:[w for w in x["ISSUES_clean"].split(" ") if not w in list_stopWords],axis=1)
data["COMMENTS_clean"]=data.apply(lambda x:[w for w in x["COMMENTS_clean"].split(" ") if not w in list_stopWords],axis=1)


# In[205]:


data


# In[ ]:





# In[206]:


# cluster_id=["GOAL_clean_cluster_id","DATA_clean_cluster_id","METHODS_clean_cluster_id","RESULTS_clean_cluster_id","ISSUES_clean_cluster_id","COMMENTS_clean_cluster_id"]


# In[207]:


# for col in cluster_id:
#     print(data.groupby(col,as_index=False)["file"].count())


# In[208]:


#保存主题词结果
column=[]
cluster=[]
term=[]
prob=[]


# In[209]:


#遍历要聚类的列
cluster_cols=["GOAL_clean","DATA_clean","METHODS_clean","RESULTS_clean","ISSUES_clean","COMMENTS_clean"]
cluster_num=[3,3,4,3,3,3]
topic_num=[10,10,10,10,10,10]
for i in range(len(cluster_cols)):
    dictionary = corpora.Dictionary(data[cluster_cols[i]].values)
    # 计算文本向量
    corpus = [dictionary.doc2bow(text) for text in data[cluster_cols[i]].values]  # 每个text对应的稀疏向量
    # 计算文档TF-IDF
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    num_topics = cluster_num[i]  # 定义主题数
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha=0.01, eta=0.01, minimum_probability=0.001,
                          update_every=1, chunksize=100, passes=1)
    doc_topic = [a for a in lda[corpus_tfidf]]
    id=[]
    for d in doc_topic:
        id.append(np.argmax([i[1] for i in d]))
    data[cluster_cols[i]+"_cluster_id"]=id

    num_show_term = topic_num[i]  # 每个主题显示几个词
    print(cluster_cols[i])
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id)
        column.append(cluster_cols[i])
        cluster.append(topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)  # 所有词的词分布
        term_distribute = term_distribute_all[:num_show_term]  # 只显示前几个词
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        print('词：', end="")
        words=""
        for t in term_id:
            words=words+dictionary.id2token[t]+" "
            print(dictionary.id2token[t], end=' ')
        term.append(words)
        print('概率：', end="")
        print(term_distribute[:, 1])
        prob.append(term_distribute[:, 1])


# In[210]:


data


# In[211]:


data.to_csv("./result.csv",index=None)


# In[216]:


result2=pd.DataFrame({"字段":column,"主题":cluster,"词":term,"概率":prob})


# In[217]:


result2.to_csv("./result2.csv",index=None)


# In[ ]:




