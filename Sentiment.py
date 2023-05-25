pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers requests beautifulsoup4 pandas numpy 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch
import time
import requests
from bs4 import BeautifulSoup
import regex as re
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sentiment Analyzer", page_icon=":smiley:")

# Add a style tag to the head of the HTML document
with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.title("Sentiment Analysis Using NLP")
st.caption("Unlock the power of sentiment analysis with our app.Simply input your own text, provide a Yelp business URL, or upload a file, and let our app do the rest. ")


#Options for the user to choose what to analyse
ownText = st.text_input("Enter your own text:")
st.write("Or")
url = st.text_input("Enter the Yelp business URL:")
st.write("Or")
with st.expander('Upload a File'):
    uploadedFile = st.file_uploader(" ")

st.markdown("---")

#Converter that converts the strings into numbers. Load the architecture of our model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# Define a function to calculate the sentiment score for a given review
def sentiment_score(ownText):
    tokens = tokenizer.encode(ownText,return_tensors='pt')
    result = model(tokens)
    #result.logits
    return int(torch.argmax(result.logits))+1

# Define a function to scrape a website's reviews
def scrape_reviews(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p',{'class':regex})
    reviews =[result.text for result in results]
    return pd.DataFrame(np.array(reviews), columns=['Reviews'])

# Define a function to perform sentiment analysis on the uploaded file
def sentiment_upload(uploadedFile):
    del df['Unnamed: 0']
    reviews = df['Reviews'].tolist()
    sentiment_scores = [sentiment_score(review[:512]) for review in reviews]
    df['Sentiment Score'] = sentiment_scores
    return df[['Reviews', 'Sentiment Score']]

#Caching the data so it dosent rerun on each refresh
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


with st.container():       
    if ownText:
        st.write("Collecting Review..")

        time.sleep(1)
        st.write("Calculating Sentiment Score...")

        time.sleep(1)
        st.write("Printing Result....")
    
        time.sleep(2)
        st.markdown("---")
        st.write("Result:")
        st.write(f"The Text : {ownText}")

        st.write(f"Sentiment Score : {sentiment_score(ownText)}")

        
with st.container(): 
    if url:
        st.write("Collecting Review..")
        df = scrape_reviews(url)

        time.sleep(1)
        st.write("Calculating Sentiment Score...")
        df['Sentiment Score'] = df['Reviews'].apply(lambda x: sentiment_score(x[:512]))

        time.sleep(1)
        st.write("Printing Result....")

        time.sleep(2)
        st.markdown("---")
        st.write("Result:")
        st.write(df)

        #function that allows the user the option to download the data as a csv file
        csv = convert_df(df)
        st.download_button(
            label= "Download as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv'
        )


with st.container(): 
    if uploadedFile:
        df = pd.read_excel(uploadedFile)
        st.write("Performing Sentiment Analysis...")

        time.sleep(1)
        st.write("Calculating Sentiment Score...")
        result_df = sentiment_upload(df)

        time.sleep(1)
        st.write("Printing Result....")

        time.sleep(2)
        st.markdown("---")
        st.write("Result:")
        st.table(result_df)

        #function that allows the user the option to download the data as a csv file
        csv = convert_df(df)
        st.download_button(
            label= "Download as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv'
        )
