
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
from jieba import analyse

jieba.load_userdict("new.dict_all")
analyse.set_stop_words('./    stopword.txt')

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

from gensim import corpora, models, similarities


        
words = []
titles = []
for id, session in sorted(sessions.iteritems(), key=lambda t: t[0]):
    words.append(list(jieba.cut(session["abstract"] + ' ' + session["title"])))

dic = corpora.Dictionary(words)
# print dic
# print dic.token2id
# 
# for word,index in dic.token2id.iteritems():
#     print word.encode('utf-8') +" 编号为:"+ str(index)

corpus = [dic.doc2bow(text) for text in words]
# print corpus

tfidf = models.TfidfModel(corpus)


corpus_tfidf = tfidf[corpus]
# for doc in corpus_tfidf:
#     print doc
# 
# lsi = models.LsiModel(corpus_tfidf, id2word=dic, num_topics=2)
# lsiout=lsi.print_topics(2)
# print lsiout[0]
# print lsiout[1]


lda = models.LdaModel(corpus_tfidf, id2word=dic, num_topics=5)
ldaOut=lda.print_topics(5)
for topic in ldaOut :
    print topic
    
corpus_lda = lda[corpus_tfidf]


for session , doc in zip(sessions,  corpus_lda):
    print session + ' >>> ' + str(doc)

