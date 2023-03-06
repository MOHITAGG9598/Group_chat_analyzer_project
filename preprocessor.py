import pandas as pd
import re
from wordcloud import WordCloud
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import ipywidgets
import jupyter

def preprocess(data):
    pattern ='\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)

        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])

        else:
            users.append('group_notification')
            messages.append(entry[0])
    df['users'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['Day_name'] = df['date'].dt.day_name()
    df['Month_name'] = df['date'].dt.month_name()

    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp.replace("", np.nan, inplace=True)
    temp = temp.dropna()

    def cleanTxt(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#', '', text)
        text = text.replace('\n', "")
        return text

    temp['message'] = temp['message'].apply(cleanTxt)
    temp['users'] = temp['users'].apply(cleanTxt)




    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    temp['Subjectivity'] = temp['message'].apply(getSubjectivity)
    temp['Polarity'] = temp['message'].apply(getPolarity)

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        if score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    temp['Analysis'] = temp['Polarity'].apply(getAnalysis)


    return temp
