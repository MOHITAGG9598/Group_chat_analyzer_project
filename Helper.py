from matplotlib import pyplot as plt
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from textblob import TextBlob
import numpy as np
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
import pandas as pd
import matplotlib.pylab as plt
import PIL.Image
import numpy as np

extract=URLExtract()
def fetch_stats(selected_user,df):

    if selected_user!= "Group analysis":
        df=df[df['users']==selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages=df[df['message'] == '<Media omitted>\n'].shape[0]
    links=[]
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['users'].value_counts().head()
    df=round((df['users'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df
def create_word_cloud(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    wc= WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df_wc=wc.generate(df['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]
    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df=pd.DataFrame(Counter(words).most_common(30))
    return most_common_df

def positive_word_cloud(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    pos_word = df[df['roberta_pos'] > 0.5]
    pos_word = pos_word.pop('message')
    pos_word_df = pd.DataFrame(pos_word)
    stopwords = set(STOPWORDS)
    mask = np.array(PIL.Image.open('wcc.png'))

    # wordcloud
    wordcloud = WordCloud(stopwords=stopwords, mask=mask, background_color="White").generate(
        ''.join(pos_word_df['message']))
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

    return wordcloud


def sentiment_analysis(df):
    x = df['Analysis'].value_counts()
    return x

