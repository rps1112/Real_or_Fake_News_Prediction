#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LogisticRegression


# In[2]:


news=pd.read_csv(r"C:\Users\rpsie\Downloads\Projects\ML Projects\train.csv\train.csv")
news.head(10)


# In[3]:


news.shape


# In[4]:


news.describe()


# In[5]:


news.info()


# In[6]:


news.isnull().sum()


# In[7]:


news.dropna(inplace=True)
news.info()


# In[8]:


news['label'].value_counts()


# In[9]:


top_authors=news['author'].value_counts().head(10)
top_authors


# In[10]:


top_authors.keys()


# In[11]:


news_counts=list(news['author'].value_counts().head(10))
news_counts


# In[12]:


plt.figure(figsize=(20,5))
plt.bar(top_authors.keys(),news_counts,color='blue')
plt.title('Authors vs News Counts')
plt.show()


# In[13]:


fake_news = news.groupby(['author', 'label']).size().unstack(level=-1)
fake_news.columns = ['Real News', 'Fake News']
fake_news.drop('Real News',axis=1,inplace=True)
fake_news.sort_values(by='Fake News',ascending=False).head(10)


# ## Separate Dataset

# In[28]:


x=news.drop('label',axis=1)
x


# In[29]:


y=news['label']
y


# ## Text Modification

# In[31]:


x['content']=x['author']+' '+x['title']+' '+x['text']
x['content']


# In[17]:


# Remove stopwords using NLTK
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Lemmatize words to their root form.(Lemmatization is the process of reducing words to their base or root form)
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


# In[33]:


import re

def stemming(text):
    # Use regular expressions to remove special characters, punctuation, and numbers
    stemmed_content = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert the cleaned text to lowercase
    stemmed_content = stemmed_content.lower()
    # Convert the cleaned text to a list by splitting
    stemmed_content = stemmed_content.split()
    # Lemmatize words and remove stopwords
    stemmed_content = [lemmatizer.lemmatize(word) for word in stemmed_content if word not in stop_words]
    # Join the filtered words back into a clean text
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[19]:


# Download the missing resource
nltk.download('omw-1.4') 
x['content'] = x['content'].apply(stemming)
print(x['content'] )


# In[34]:


x['content'].shape


# ## Train Model

# In[35]:


# convert the textual data to numerical data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
vectorizer.fit(x['content'])
x=vectorizer.fit_transform(x['content'])
print(x)


# In[52]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=30)


# In[53]:


model=LogisticRegression()
model.fit(x_train,y_train)


# In[54]:


from sklearn.metrics import accuracy_score,classification_report
y_test_pred=model.predict(x_test)
accuracy=accuracy_score(y_test_pred,y_test)
report=classification_report(y_test_pred,y_test)
print(accuracy)
print(report)


# In[55]:


y_train_pred=model.predict(x_train)
accuracy=accuracy_score(y_train_pred,y_train)
report=classification_report(y_train_pred,y_train)
print(accuracy)
print(report)


# ## Model Prediction

# In[59]:


input=x_test[39]
prediction=model.predict(input)
print(prediction)

if (prediction==0):
    print("REAL NEWS")
else:
    print("FAKE NEWS")


# In[ ]:




