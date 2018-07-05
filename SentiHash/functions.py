import pandas as pd
import numpy as np
import time
import json
import os
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

class StdOutListener(StreamListener):

    def on_data(self, data):
        #print("Writing data")
        with open('fetched_tweets.txt','a') as tf:
            tf.write(data)
        return True

    def on_error(self, status):
        print (status)

def credentials():
    credentials = {}
    with open('Usernames.txt', 'r') as f:
        for line in f:
            user, pwd, url = line.strip().split(';')
            lst=[pwd,url]
            credentials[user] = lst
        return credentials

def model_run(txt,path):
    p=os.path.join(path,'fetched_tweets.txt')
    try:
        os.remove(p)
    except:
        pass
    access_token = "249546463-dfSoFrUfICT7v6G365EMlJIc5hzm3iDVqXwoQjI9"
    access_token_secret = "86A2u1KBgAYy4xmk59pEvolq4An48GrGFsTMJN2lVMZDm"
    consumer_key = "ZUi056QWw4R29pb7HnmaarkL3"
    consumer_secret = "VpId97YGfwYv8aKhrjdlODlVlzKIEqCoFCix1ibfEad6wNdasG"
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    tweets_data = []
    stream.filter(track=[txt],async=True)
    time.sleep(60)
    stream.disconnect()
    #print("File read in progress..")
    for line in open('fetched_tweets.txt','r'):
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue
    tweets = pd.DataFrame()
    tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
    tweets['lang'] = list(map(lambda tweet: tweet['lang'], tweets_data))
    tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data))   
    tweets['Source'] = list(map(lambda tweet: tweet['source'], tweets_data))
    tweets['Location'] = list(map(lambda tweet: tweet['user']['location'], tweets_data))
    tweets['Name'] = list(map(lambda tweet: tweet['user']['name'], tweets_data))
    tweets['Screen_Name'] = list(map(lambda tweet: tweet['user']['screen_name'], tweets_data))
    tweets['URL'] = list(map(lambda tweet: tweet['user']['url'], tweets_data)) 
    tweets['Geo_enabled'] = list(map(lambda tweet: tweet['user']['geo_enabled'], tweets_data))
    tweets.to_csv('spring.csv',index=False)
    lang = pd.read_csv("dictionary of language.csv", encoding='ISO-8859-1')
    full = []
    nulls = []
    for la in tweets['lang']:
        flag = False
        for ind, row in lang.iterrows():
            if la == row['short_form']:
                flag = True
                full.append(row['full_form'])
        if flag == False:
            nulls.append(la)
            full.append("Null")
    tweets['lang'] = full
    sid = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    final=tweets
    neg,neu,pos,comp =[],[],[],[] 
    for text in final['text']:
        tokens = word_tokenize(text)
        cleaned = [x for x in tokens if x not in stop_words]
        cleaned = [x for x in cleaned if x.isalpha()]
        cleaned = " ".join(cleaned)
        ss = sid.polarity_scores(cleaned)
        neg.append(ss['neg'])
        neu.append(ss['neu'])
        pos.append(ss['pos'])
        comp.append(ss['compound'])
    final['Negative'] = neg
    final['Neutral'] = neu
    final['Positive'] = pos
    final['Compound'] = comp
    final.to_csv('final.csv')
    file_names = plot_graph(final)
    return file_names

def plot_graph(data):
    file_names = ['Positive.png','Neutral.png','Negative.png']
    colors = ['g','b','r']
    for file, color in zip(file_names, colors):
        type = file[:len(file)-4]
        print(file, color)
        plt.plot(data[type], color=color)
        plt.title(type+' Sentiment Score')
        plt.savefig('static/images/'+file)
        plt.close()
    return file_names