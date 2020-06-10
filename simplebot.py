from textblob import TextBlob
import twily_classifier as cl
import stop_words as stop_words
import json

with open('twilybot.json', 'r') as f:
    array = json.load(f)

CONVERSATION = array["conversations"]

BOT_NAME = 'Twily'
STOP_WORDS = stopwords.sw_list
neg_distribution = []