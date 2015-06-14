
# coding: utf-8

# In[4]:


import ijson
from gensim import corpora, models, similarities
import jieba
from jieba import analyse
import util


jieba.load_userdict("new.dict_all")
stop_words = util.load_stop_words('stopword.txt')


f = open("./spam-articles-half-a.json")
import itertools
from BeautifulSoup import BeautifulSoup



# In[5]:

objects = ijson.items(f, 'item')


# In[6]:

count = 0
words = []
for obj in objects :
    if obj['is_spam'] == 0 and obj['content'] != None :
        soup = BeautifulSoup(obj['content'])
        
        l =[]
        for item in jieba.cut(soup.getText(),cut_all = False) :
            if len(item) < 2 or item in stop_words: continue
            else : l.append(item)
        words.append(l)
        count = count + 1
    if count > 100: break

dic = corpora.Dictionary(words)
corpus = [dic.doc2bow(text) for text in words]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel(corpus_tfidf, id2word=dic, num_topics=5)
ldaOut=lda.print_topics(5)
for topic in ldaOut :
    print topic
    
corpus_lda = lda[corpus_tfidf]


# In[7]:

print page


# In[ ]:



