#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import stem_text
from nltk.corpus import stopwords
import pickle
import en_core_web_sm
import matplotlib.pyplot as plt


# In[2]:


url='https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json'
df=pd.read_json(url)
#It has three columns: content, target and target_names. It's email text from people in different universities
df.head()


# In[3]:


#distribution of target_names: 20 categories can be found in target_name.
#It's about sports, religions, autos,science etc.most of categories have around 600 records, accounting for 5% of records.
target_name_unique=df['target_names'].value_counts(normalize=True)
target_name_unique.plot(kind='bar')
plt.show()


# In[4]:


#It has 11,314 entries.
df.count()


# In[5]:


#the first content value
#it matches with the target_name rec.autos
print(df['content'][0])
#business question:  
# the popular/demanding specializes/areas that the university needs to put more efforts on. 
# hobbies -> clubs


# In[6]:


get_ipython().run_line_magic('pinfo', 're')


# In[7]:


#Remove the lines beginning with any of the following: ‘From:’, ‘Article-I.D.:’, ‘Organization:’, ‘Lines:’, ‘NNTP-Posting-Host:’, ‘Distribution:’, ‘Reply-To:’, ‘X-Newsreader:’, ‘Expires:’
#I also removed 'Nntp-Posting-Host:'
df['content2']=df['content']
for i in range(0,len(df['content'])):
               regex=re.sub(r'^(From:|Lines:|Subject:|Organization:|Article-I.D.:|Nntp-Posting-Host:|Distribution:|Reply-To:|X-Newsreader:|Expires:|NNTP-Posting-Host:).*\n?', '', str(df['content'][i]), flags=re.MULTILINE | re.IGNORECASE
)
               df['content2'][i]=regex


# In[8]:


#check the result
df['content2'][3]


# In[9]:


#Remove any of the words ‘Subject:’, ‘Summary:’ or ‘Keywords:’.
for i in range(0,len(df['content'])):
               regex=re.sub(r'^(Subject:|Summary:|Keywords:)*?', '', str(df['content2'][i]), flags=re.IGNORECASE )
               df['content2'][i]=regex


# In[10]:


#check
df['content2'][10]


# In[11]:


#remove multiples (also one) of ‘-‘ preceded by space
for i in range(0,len(df['content'])):
               regex=re.sub(r'\s*-', '', str(df['content2'][i]))
               df['content2'][i]=regex


# In[12]:


#check
df['content2'][10]


# In[13]:


#we remove these because they don't convey information relevant for our anaysis and are instead noise.
get_ipython().run_line_magic('pinfo', 'strip_numeric')


# In[14]:


#part C. Apply strip_numeric, strip_punctuation and strip_multiple_whitespaces to data and override it because Punctuation improves the readability of the text, 
#but often does not convey information
#Apply strip_numeric:Remove digits from `s` using for text analysis.
for i in range(0,len(df['content'])):
               regex=strip_numeric(str(df['content2'][i]))
               df['content2'][i]=regex


# In[15]:


#test
df['content2'][10]


# In[16]:


#strip_punctuation:Replace punctuation characters with spaces in `s` 
for i in range(0,len(df['content'])):
               regex=strip_punctuation(str(df['content2'][i]))
               df['content2'][i]=regex


# In[17]:


#test
df['content2'][20]


# In[18]:


#strip_multiple_whitespaces: Remove repeating whitespace characters (spaces, tabs, line breaks) from `s`
#and turns tabs & line breaks into spaces
for i in range(0,len(df['content'])):
               regex=strip_multiple_whitespaces(str(df['content2'][i]))
               df['content2'][i]=regex


# In[19]:


#test
df['content2'][20]


# In[20]:


#Transform all letters to lower case ones
for i in range(0,len(df['content'])):
               regex=(str(df['content2'][i])).lower()
               df['content2'][i]=regex


# In[21]:


#test
df['content2'][1]


# In[22]:


#Part D stopwords in gensim
get_ipython().run_line_magic('pinfo', 'STOPWORDS')
print(sorted(list(STOPWORDS)))


# In[23]:


#stopwords in nltk
get_ipython().run_line_magic('pinfo', 'stopwords')
print(stopwords)


# In[24]:


from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words('english'))
nltk_stopwords


# In[25]:


type(nltk_stopwords)


# In[26]:


#179 words(fewer than genism STOPWORD), including contraction e.g. what's, you've which is not in gensim STOPWORD
len(sorted(list(nltk_stopwords)))


# In[27]:


#remove_stopwords in gensim
for i in range(0,len(df['content'])):
               regex=remove_stopwords((str(df['content2'][i])))
               df['content2'][i]=regex


# In[28]:


#test
df['content2'][0]


# In[29]:


get_ipython().run_line_magic('pinfo', 'strip_short')
#Remove words with length lesser than `minsize` from `s`.defualt min=3


# In[30]:


#use strip_short
for i in range(0,len(df['content'])):
               regex=strip_short((str(df['content2'][i])))
               df['content2'][i]=regex


# In[31]:


#test
df['content2'][1]


# In[32]:


#Part E
get_ipython().run_line_magic('pinfo', 'stem_text')
#Transform `s` into lowercase and stem it


# In[33]:


df['content_stem']=df['content2']


# In[34]:


#apply stem_text
for i in range(0,len(df['content'])):
               regex=stem_text((str(df['content2'][i])))
               df['content_stem'][i]=regex


# In[35]:


#test
df['content_stem'][0]


# In[36]:


#Initializing spacy’s 'en' model
import spacy
sp = spacy.load('en_core_web_sm')


# In[37]:


df['content_lemma']=df['content2']


# In[38]:


#apply lemmatization
for i in range(0,len(df['content'])):
    con=sp(df['content2'][i])
    df['content_lemma'][i]=" ".join([token.lemma_ for token in con])
    


# In[39]:


#test
df['content_lemma'][0]


# In[44]:


#store data using pickle. both makes sense because stemming is quick and simple but resulting word may not exist. 
#Lemmatization gets the true root but is sloweer and less words for grouping
with open('s_l.p','wb') as f:
    pickle.dumps((df['content_stem'],df['content_lemma']),f)


# In[41]:


#pickle file saves space, can save in any form, can export any objects and share with others. 
with open('s_l.p','rb') as f:
    content_stem,content_lemm = pickle.load(f)


# In[42]:


content_stem


# In[43]:


content_lemmab


# In[ ]:




