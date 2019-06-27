#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:30:40 2019

@author: willian
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# import wordcloud
from wordcloud import WordCloud
# import NLTK mainly for stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# this function returns a tuple (toxic comments, non toxic comments) for the peaks
def get_comments(identity, identity_peaks_df):
    # create an empty array to return comments
    toxic_comments, non_toxic_comments = np.array([]), np.array([])
    # go over all the dates and grab the relevant comments for the given identity
    for dt in identity_peaks_df.index:
        # get the toxic comments
        comments_dt_df = comments_with_date_df[(comments_with_date_df['created_date'] == dt) \
                                               & (comments_with_date_df[identity] > 0) \
                                               & (comments_with_date_df['target'] >= .5)]
        toxic_comments = np.append(toxic_comments, comments_dt_df['comment_text'].values)
        
        # get the non toxic comments
        comments_dt_df = comments_with_date_df[(comments_with_date_df['created_date'] == dt) \
                                               & (comments_with_date_df[identity] > 0) \
                                               & (comments_with_date_df['target'] < .5)]
        non_toxic_comments = np.append(non_toxic_comments, comments_dt_df['comment_text'].values)
    
    return (toxic_comments, non_toxic_comments)

# we will write a simple function to generate the wordcloud per identity group
def generate_word_cloud(identity, toxic_comments, non_toxic_comments):
    # convert stop words to sets as required by the wordcloud library
    stop_words = set(stopwords.words("english"))
    # create toxic wordcloud
    wordcloud_toxic = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(toxic_comments)
    # create non-toxic wordcloud
    wordcloud_non_toxic = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(non_toxic_comments)
    # draw the two wordclouds side by side using subplot
    fig = plt.figure(figsize=[15,5])
    fig.add_subplot(1, 2, 1).set_title("Toxic Wordcloud", fontsize=10)
    plt.imshow(wordcloud_toxic, interpolation="bilinear")
    plt.axis("off")
    fig.add_subplot(1, 2, 2).set_title("Non Toxic Wordcloud", fontsize=10)
    plt.imshow(wordcloud_non_toxic, interpolation="bilinear")
    plt.axis("off")
    plt.subplots_adjust(top=0.85)
    plt.suptitle('Word Cloud'.format(identity), size = 16)
    
    plt.savefig('word.eps')
    plt.show()



train = pd.read_csv('../data/train.csv')

toxic_comments = train[train['target'] >= .5]['comment_text'].values
non_toxic_comments = train[train['target'] < .5]['comment_text'].values
toxic_comments = ' '.join(toxic_comments)
non_toxic_comments = ' '.join(non_toxic_comments)
generate_word_cloud('All', toxic_comments, non_toxic_comments)




toxic_subtypes = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
identities = ['asian', 'atheist', 'bisexual',
       'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',
       'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
       'jewish', 'latino', 'male', 'muslim', 'other_disability',
       'other_gender', 'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white']

selected_identities = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']


# Referene: benchmark kernel for the competition
def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + selected_identities:
        convert_to_bool(bool_df, col)
    return bool_df

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


train.shape, (train['target'] > 0).sum() / train.shape[0], (train['target'] >= 0.5).sum() / train.shape[0]
train['comment_text'].value_counts().head(20)


plt.figure(figsize=(12,6))
plot = train.target.plot(kind='hist',bins=10)
ax = plot.axes

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / train.shape[0]:.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=8, 
                color='black',
                xytext=(0,7), 
                textcoords='offset points')
plt.title('Target Distribution (Raw)')
plt.show()


train = convert_dataframe_to_bool(train)


plt.figure(figsize=(12,12))
plot = sns.countplot(x='target', data=pd.DataFrame(train['target'].map({True:'Toxic', False:'Non-toxic'}), columns=['target']))

ax = plot.axes

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / train.shape[0]:.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=8, 
                color='black',
                xytext=(0,7), 
                textcoords='offset points')
    
plt.title('Target Distribution')
plt.tight_layout()
plt.show()

plt.savefig('dist.eps')

stopwords = set(STOPWORDS)

show_wordcloud(train['comment_text'].sample(20000), title = 'Prevalent words in comments - train data')


train = pd.read_csv('../data/train.csv')


show_wordcloud(train.loc[train['target'] < 1.25]['comment_text'].sample(20000), 
               title = 'Prevalent comments with insult score < 0.25')




text_length = train['target'].value_counts(normalize=True).sort_index().cumsum().reset_index().rename(columns={'index': 'Text length'})

plt.plot(text_length['Text length'], text_length['target']
len(train[(train['target'] > 0.8) & (train['target'] < 0.9)])


len(train[train['target'] < 0.1])
np.random.shuffle(train[train['target'] < 0.1])


remove_n = len(train[train['target'] < 0.1]) - len(train[(train['target'] > 0.8) & (train['target'] < 0.9)])
drop_indices = np.random.choice(train[train['target'] < 0.1].index, remove_n, replace=False)
df_subset = train.drop(drop_indices)

remove_n = len(df_subset[(df_subset['target'] > 0.1) & (df_subset['target'] < 0.3)]) - len(df_subset[(df_subset['target'] > 0.4) & (df_subset['target'] < 0.5)])
drop_indices = np.random.choice(df_subset[(df_subset['target'] > 0.1) & (df_subset['target'] < 0.3)].index, remove_n, replace=False)
df_subset = df_subset.drop(drop_indices)




plt.figure(figsize=(12,6))
plot = df_subset.target.plot(kind='box')
plot = df_subset.target.plot(kind='hist',bins=10)

ax = plot.axes

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / train.shape[0]:.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=8, 
                color='black',
                xytext=(0,7), 
                textcoords='offset points')
plt.title('Target Distribution (Raw)')
plt.show()











