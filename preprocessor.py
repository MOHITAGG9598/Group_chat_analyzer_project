import pandas as pd
import re
from wordcloud import WordCloud
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import ipywidgets
import jupyter
sia=SentimentIntensityAnalyzer()

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

    res = {}
    for i, row in tqdm(temp.iterrows(), total=len(temp)):
        text = row['message']
        myid = row['users']
        res[myid] = sia.polarity_scores(text)

    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'users'})
    vaders = vaders.merge(temp, how="right")

    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    from scipy.special import softmax

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def polarity_scores_roberts(example):
        encoded_text = tokenizer(example, return_tensors="pt")
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]

        }
        return scores_dict

    res = {}
    for i, row in tqdm(vaders.iterrows(), total=len(vaders)):
        try:
            text = row['message']
            myid = row['users']
            vader_result = sia.polarity_scores(text)
            vader_result_rename = {}
            for key, value in vader_result.items():
                vader_result_rename[f"vader_{key}"] = value
            roberta_result = polarity_scores_roberts(text)
            both = {**vader_result, **roberta_result}
            res[myid] = both
        except RuntimeError:
            print(f"Broke for id {myid}")

    results_df = pd.DataFrame(res).T
    results_df = results_df.reset_index().rename(columns={'index': 'users'})
    results_df = results_df.merge(vaders, how="right")



    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    results_df['Subjectivity'] = results_df['message'].apply(getSubjectivity)
    results_df['Polarity'] = results_df['message'].apply(getPolarity)

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        if score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    results_df['Analysis'] = results_df['Polarity'].apply(getAnalysis)


    return results_df
