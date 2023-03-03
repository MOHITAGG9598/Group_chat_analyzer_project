import base64
import os
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import ipywidgets
import jupyter
import nltk
import openpyxl
nltk.download('all')




import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
import Helper
import preprocessor
from textblob import TextBlob
import os
from mtranslate import translate
import pandas as pd
import os
from gtts import gTTS
import base64
st.sidebar.title("Whatsapp Chat analyzer")

uploaded_file= st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    df_new= preprocessor.preprocess(data)
    st.dataframe(df_new)

    user_list= df_new['users'].unique().tolist()
    user_list.sort()
    user_list.insert(0,"Group analysis")
    selected_user=st.sidebar.selectbox("show analysis wrt",user_list)
    if st.sidebar.button("Show Analysis"):
        num_messages,words,num_media_messages,num_links=Helper.fetch_stats(selected_user,df_new)
        col1,col2,col3,col4=st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        if selected_user == "Group analysis":
            st.title("Most busy users")
            x,new_df=Helper.most_busy_users(df_new)
            fig,ax=plt.subplots()
            col1,col2=st.columns(2)

            with col1:
                ax.bar(x.index, x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        st.title("Word Cloud")
        df_wc = Helper.create_word_cloud(selected_user, df_new)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        st.title("Positive Word cloud")
        df_wc = Helper.create_word_cloud(selected_user, df_new)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        plt.axis('off')
        st.pyplot(fig)

        st.title("Most Common Words")
        most_common_df=Helper.most_common_words(selected_user,df_new)
        fig,ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        st.pyplot(fig)
        st.dataframe(most_common_df)

        if selected_user == "Group analysis":
            st.title("Sentiment Analysis")
            x = Helper.sentiment_analysis(df_new)
            fig, ax = plt.subplots()
            ax.bar(x[0],x[1])
            st.pyplot(fig)

st.title("Sentiment Analysis")
@st.cache(allow_output_mutation=True)
def get_model():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer,model


tokenizer, model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

sent_pipeline = pipeline("sentiment-analysis")
if user_input and button:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Prediction: ", sent_pipeline(user_input))
    showWarningOnDirectExecution = False

df = pd.read_excel(os.path.join( 'language.xlsx'),sheet_name='wiki')
df.dropna(inplace=True)
lang = df['name'].to_list()
langlist=tuple(lang)
langcode = df['iso'].to_list()

# create dictionary of language and 2 letter langcode
lang_array = {lang[i]: langcode[i] for i in range(len(langcode))}

# layout
st.title("Language-Translation + Text-To-Speech")
st.markdown("In Python 🐍 with Streamlit ! (https://www.streamlit.io/)")
st.markdown("Languages are pulled from language.xlsx dynamically. If translation is available it will be displayed in TRANSLATED TEXT window.\n In addition if text-to-Speech is supported it will display audio file to play and download." )
inputtext = st.text_area("INPUT",height=200)

choice = st.sidebar.radio('SELECT LANGUAGE',langlist)

speech_langs = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-CN": "Chinese"
}

# function to decode audio file for download
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

c1,c2 = st.columns([4,3])

# I/O
if len(inputtext) > 0 :
    try:
        output = translate(inputtext,lang_array[choice])
        with c1:
            st.text_area("TRANSLATED TEXT",output,height=200)
        # if speech support is available will render autio file
        if choice in speech_langs.values():
            with c2:
                aud_file = gTTS(text=output, lang=lang_array[choice], slow=False)
                aud_file.save("lang.mp3")
                audio_file_read = open('lang.mp3', 'rb')
                audio_bytes = audio_file_read.read()
                bin_str = base64.b64encode(audio_bytes).decode()
                st.audio(audio_bytes, format='audio/mp3')
                st.markdown(get_binary_file_downloader_html("lang.mp3", 'Audio File'), unsafe_allow_html=True)
    except Exception as e:
        st.error(e)



