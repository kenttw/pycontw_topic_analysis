
# coding: utf-8

# In[1]:

import csv

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer , TfidfTransformer
from sklearn.decomposition import NMF
from collections import defaultdict
from bs4 import BeautifulSoup, NavigableString

from lxml.html.soupparser import fromstring
from bs4 import UnicodeDammit

import lxml.etree as ET

import jieba
import jieba.posseg as pseg


jieba.load_userdict("new.dict_all")
# import jieba.analyse
# jieba.analyse.load_stop_words("stop_words_list.txt")


def uri_to_file_name(uri):
    return uri.replace("/", "-")

sessions = {}

xpath_abstract = '''//div[@class='panel-body']/div[1]/text()'''

with open("data/sc.csv", "r") as sessions_file:
    for row in csv.DictReader(sessions_file, ['title', 'link', 'speaker']):  
        session_id = (row['title'])
        filename = "data/sessions/" + uri_to_file_name(row['link']) + '.html'
        page = open(filename).read()
        soup = fromstring(page)
        ab = soup.xpath(xpath_abstract)
        if len(ab) >0 :
            abstract = ab[0].replace('\n', ' ').replace('\r', '').replace('\r\n', '').encode('utf-8','ignore')
            title = row['title']
            sessions[row['link']] = {'title':title , 'abstract':abstract}
           


# In[5]:


        
corpus = []
titles = []
for id, session in sorted(sessions.iteritems(), key=lambda t: t[0]):
    wordlist=pseg.cut(session["abstract"] + session["title"])
    words = ''
    for key in wordlist:  
         words  = words + ' ' + key.word
    corpus.append(words)
    titles.append(session["title"])

n_topics = 10
n_top_words = 10
n_features = 6000

# vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
# vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')


if False :
#     vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
#     transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
#     tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
#     word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
#     weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
    tfidf =  vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    
else :
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
    tfidf = vectorizer.fit_transform(corpus)
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  



import lda
import numpy as np
# 
vocab = word



if True:
    model = lda.LDA(n_topics=n_topics, n_iter=500, random_state=1)
    model.fit(tfidf)
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        s = ''
        for word in topic_words :
            s = s + ' ' + word.encode('utf-8')
        print('Topic ' + str(i) + ' : ' + s )
else :
    # Fit the NMF model
    print("Fitting the NMF model with n_samples=%d and n_features=%d..."% (tfidf.shape[0], n_features))
    nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    feature_names = vectorizer.get_feature_names()
    
    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()


# 
# doc_topic = model.doc_topic_
# for i in range(0, len(titles)):
#     print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
#     print(doc_topic[i].argsort()[::-1][:3])




