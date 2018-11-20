
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[ ]:


df = pd.read_csv('profiles.csv')


# In[ ]:


#New Column
zodsign = list(['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio', 'sagittarius', 'capricorn', 'aquarius', 'pisces'])
new_sign = []
for zod in zodsign:
    for i in range(100):
        if zod in str(df.sign[i]):
           # print(df.sign[i])else:
            new_sign.append(zod)
new_signs = df.sign
new_signs = new_signs.replace(' and it&rsquo;s fun to think about', '', regex=True)
new_signs = new_signs.replace(' but it doesn&rsquo;t matter', '', regex=True)
new_signs = new_signs.replace(' and it matters a lot', '', regex=True)


# In[ ]:


#New column
body_cats = {"average":"average", "skinny":"ectomorph", "thin":"ectomorph", "a little extra":"endomorph", "full figured":"endomorph", "curvey":"endomorph", "overweight":"endomorph", "athletic":"mesomorph", "fit":"mesomorph", "jacked":"mesomorph", "used up":"other", "rather not say":"other"}
df["body_cat"] = df["body_type"].map(body_cats)


# In[ ]:


all_data = df.iloc[:, [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28]]


# In[ ]:


#New columns
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}

all_data['drinks_code'] = all_data.drinks.map(drink_mapping)
all_data['smokes_code'] = all_data.smokes.map(smokes_mapping)
all_data['drugs_code'] = all_data.drugs.map(drugs_mapping)


# In[ ]:


all_data = df.iloc[:, [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28]]


# In[ ]:


essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]


# In[ ]:


# Removing the NaNs
all_essays = all_data[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)


# In[ ]:


all_data["essay_len"] = all_essays.apply(lambda x: len(x))


# In[ ]:


all_essays[34513] = ''
all_essays[40640] = ''
all_essays[44214] = ''
all_essays[52093] = ''
all_essays[58762] = ''
all_essays[58866] = ''
all_essays[59945] = ''


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer(stop_words=None, analyzer='word')

def matrix_to_list(matrix):
    matrix = matrix.toarray()
    return matrix.tolist()

def avg_mei_count(essays):
    temp_essays = essays
    avg_wordlist = []
    temp1 = [temp_essays]
    for t in range(1):
        cv_score = cv.fit_transform(temp1)
        cv_score_list = matrix_to_list(cv_score)
        cv_wordlist = cv.get_feature_names()
        for i in range(len(cv_wordlist)):
            wordlen = [len(word) for word in cv_wordlist]
  
            #print(cv_score_list[0][cv.get_feature_names().index('i')] + cv_score_list[0][cv.get_feature_names().index('me')])
        meic = 0
        if 'me' in cv.get_feature_names():
            meic += cv_score_list[0][cv.get_feature_names().index('me')]
        if 'i'  in cv.get_feature_names():
            meic += cv_score_list[0][cv.get_feature_names().index('i')]
        return (meic, np.mean(wordlen)) 


# In[ ]:


avglen = []
meicount = []
for i in range(len(all_essays)):
    print(i, len(all_essays[i].split()))
    if len(all_essays[i].split())>2:
        a, b = avg_mei_count(all_essays[i])
    else:
        a = 0
        b = 0
    meicount.append(a)
    avglen.append(b)


# In[ ]:


all_data.to_csv('new_data_profiles.csv', index=False)


# In[ ]:


df_new = pd.read_csv('new_data_profiles.csv')

