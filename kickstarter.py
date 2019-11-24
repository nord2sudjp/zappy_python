#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[241]:


def prtSep(length=20, marker="-"):
    print( marker * length)

def prtDFBasic(df):
    print(df.head())
    prtSep()
    print(df.describe())
    prtSep()
    print(df.info())

def prtEdaValueCount(df, df_fld):
    for col in df_fld:
        if df[col].dtype != 'float64':
            print(df[col].value_counts())
            prtSep()
            print(df[col].value_counts(normalize=True))
            prtSep()
            
            
def plotEdaCategoral(df, df_fld):
    fig = plt.figure(figsize=(30, 60))
    i = 1
    for col in data1_ctg:
        if d_train[col].dtype != 'float64':
            fig.add_subplot(10,2,i)
            ax = sns.countplot(x=col, data=d_train);
            # seaborn.countplotにはnormalizeオプションはない
            i += 1


# In[ ]:


# Read USER data
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('dev_rsc/ml-100k/u.user', sep='|', names=u_cols)
users.head()


# In[228]:


# read EVALUATION data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('dev_rsc/ml-100k/u.data', sep='\t', names=r_cols)
ratings['date'] = pd.to_datetime(ratings['unix_timestamp'], unit='s')
prtDFBasic(ratings)


# In[229]:


# read MOVIE data
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('dev_rsc/ml-100k/u.item', sep='|', names=m_cols, usecols = range(5), encoding='latin1')
prtDFBasic(movies)


# In[223]:


movies_ratings = pd.merge(movies, ratings)
d_train = pd.merge(movies_ratings, users)
prtDFBasic(d_train)


# In[46]:


data1_base = ['title','rating', 'user_id', 'sex', 'age'] # pretty name/values for charts
data1_ctg = ['rating', 'sex']
data1_qnt = ['age']


# In[236]:


# Categoral - q-@Value Count
prtEdaValueCount(d_train, data1_base)


# In[238]:


# Categoral - Count plot
#  平均値、中央値、最頻値 
#  極端に多いもしくは少ないデータはあるか
#  各層の分布の特徴に傾向があるか
plotEdaCategoral(d_train, data1_ctg)


# In[48]:


# Quantitative Data - Distribution
#     Boxplot - 平均、外れ値の確認
#     Histgram - 分布(平均と分散、正規化しているか?)
i=1
fig = plt.figure(figsize=(20, 30))
for x in data1_qnt:
    ax = fig.add_subplot(4,2,1)
    ax.boxplot(d_train[x], showmeans = True, meanline = True)
    # pyplot.boxplotはhueがつかえないので、定量データの分布および外れ値を見るためだけに利用する
    # hueする場合にはseaborn.boxplotを利用する
    ax.set_xlabel(x)
    
    ax = fig.add_subplot(4,2,i+1)
    ax.hist(d_train[x].dropna(), bins=10, rwidth=0.5)
    ax.set_xlabel(x)    


# In[239]:


# 2変数の共起 - カテゴリ vs カテゴリor定量
#    Sex vs Age, Rating

fig, ax = plt.subplots(1,2,figsize=(14,12))

# Sex vs Age (CTG vs CTG)
sns.countplot(x='sex', hue='rating', data=d_train, ax=ax[0]);
# boxplotはy軸を指定する
# countplotはy軸は件数になる。y軸の指定はできない
ax[1].set_title('Sex vs Ratings Comparison')

# Sex vs Age (CTG vs QNT)
sns.boxplot(x='sex', y = 'age', data=d_train, ax=ax[1]);
ax[0].set_title('Sex vs Age Comparison')


# In[59]:


# 2変数の共起 - 定量 vs カテゴリ/定量
#     Age vs Rating

# Age vs Rating (QNT vs CTG)
a = sns.FacetGrid( d_train, hue='rating', aspect=4 )
a.map(sns.kdeplot, 'age', shade= True )
a.set(xlim=(0 , d_train['age'].max()))
a.add_legend()


# In[250]:


# Title - 件数 vs 評価の傾向
movie_stat = d_train.groupby('title').agg({'rating':[np.size, np.mean]})
movie_stat.columns = movie_stat.columns.droplevel()
x = np.round(movie_stat['mean'])
movie_stat['rating_avg'] = x.astype(np.int64)
prtDFBasic(movie_stat)


# In[251]:


# Value Count
prtEdaValueCount(movie_stat, ['rating_avg'])


# In[307]:


fig = plt.figure(figsize=(20, 30))
fig_col = 4
fig_row = 4

ax = fig.add_subplot(fig_row,fig_col,1)
ax.boxplot(movie_stat['size'], showmeans = True, meanline = True)
ax.set_title('Movie Rating Count - Distribution')
ax.set_xlabel('Movie Rating Count')

ax = fig.add_subplot(fig_row,fig_col,2)
ax.hist(movie_stat['size'].dropna(), bins=10, rwidth=0.5)
ax.set_title('Movie Rating Count - Distribution')
ax.set_xlabel('Movie Rating Count')

ax = fig.add_subplot(fig_row,fig_col,3)
sns.boxplot(x='rating_avg', y = 'size', data=movie_stat, ax=ax);
ax.set_title('Movie Rating Count by Movie Rating')
ax.set_xlabel('Movie Rating')

ax = fig.add_subplot(fig_row,fig_col,4)
splitAsRating = []
for i in [1, 2, 3, 4, 5]:
    splitAsRating.append(movie_stat[movie_stat['rating_avg'] == i])
tmp  = [i['size'].dropna() for i in splitAsRating]
ax.hist(tmp, histtype="barstacked", bins=10, rwidth=0.5)
ax.set_title('Movie Rating Count by Movie Rating')
ax.set_xlabel('Movie Rating Count by Rting Count')


# In[298]:


len(tmp[0])


# In[300]:


a = sns.FacetGrid( movie_stat, hue='rating_avg', aspect=4 )
a.map(sns.kdeplot, 'size', shade= True )
a.set(xlim=(0 , movie_stat['size'].max()/10))
a.add_legend()


# In[301]:


movie_stat_sel = movie_stat['size']>=50

a = sns.FacetGrid( movie_stat[movie_stat_sel], hue='rating_avg', aspect=4 )
a.map(sns.kdeplot, 'size', shade= True )
a.set(xlim=(0 , movie_stat[movie_stat_sel]['size'].max()))
a.add_legend()


# In[302]:


print(movie_stat[movie_stat['rating_avg']==1].sort_values(by=[('size')]))
prtSep()
print(movie_stat[movie_stat['rating_avg']==5].sort_values(by=[('size')]))


# In[304]:


# User - 件数
user_stat = d_train.groupby('user_id').size().reset_index()
user_stat.columns = ['user_id', 'count']
prtDFBasic(user_stat)


# In[206]:


fig = plt.figure(figsize=(20, 30))
ax = fig.add_subplot(4,3,1)
ax.boxplot(user_stat['count'], showmeans = True, meanline = True)
ax.set_title('User Rating Count - Distribution')
ax.set_xlabel('User Rating Count')

ax = fig.add_subplot(4,3,2)
ax.hist(user_stat['count'].dropna(), bins=10, rwidth=0.5)
ax.set_title('User Rating Count - Distribution')
ax.set_xlabel('User Rating Count')


# In[305]:


# User - 件数 vs 評価の傾向
user_stat_r = d_train.groupby('user_id').agg({'rating':[np.size, np.mean]})
user_stat_r.columns = user_stat_r.columns.droplevel()
x = np.round(user_stat_r['mean'])
user_stat_r['rating_avg'] = x.astype(np.int64)
prtBasicInfo(user_stat_r)


# In[306]:


# Value Count
prtEdaValueCount(user_stat_r, ['rating_avg'])


# In[309]:


fig = plt.figure(figsize=(20, 30))
fig_row = 4
fig_col = 4

ax = fig.add_subplot(fig_row, fig_col,1)
ax.boxplot(user_stat_r['size'], showmeans = True, meanline = True)
ax.set_title('User Rating Count - Distribution')
ax.set_xlabel('User Rating Count')

ax = fig.add_subplot(fig_row, fig_col,2)
ax.hist(user_stat_r['size'].dropna(), bins=10, rwidth=0.5)
ax.set_title('User Rating Count - Distribution')
ax.set_xlabel('User Rating Count')

ax = fig.add_subplot(fig_row, fig_col,3)
sns.boxplot(x='rating_avg', y = 'size', data=user_stat_r, ax=ax);
ax.set_title('User Rating Count - Distribution by Rating')
ax.set_xlabel('User Rating')

ax = fig.add_subplot(fig_row,fig_col,4)
splitAsRating = []
for i in [1, 2, 3, 4, 5]:
    splitAsRating.append(user_stat_r[user_stat_r['rating_avg'] == i])
tmp  = [i['size'].dropna() for i in splitAsRating]
ax.hist(tmp, histtype="barstacked", bins=10, rwidth=0.5)
ax.set_title('User Rating Count by Movie Rating')
ax.set_xlabel('User Rating Count by Rating Count')


# In[168]:


a = sns.FacetGrid( user_stat, hue='mean_int', aspect=4 )
a.map(sns.kdeplot, 'size', shade= True )
a.set(xlim=(0 , user_stat['size'].max()))
a.add_legend()

