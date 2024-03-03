# Import necessary libraries
from wordcloud import STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import emoji
import string
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter  # VSL
import plotly.express as px  # VSL
import pandas as pd
import string  # For some string manipulation tasks
import nltk  # Natural Language Toolkit
import re  # Regular expression
from string import punctuation  # Solving punctuation problems
from nltk.corpus import stopwords  # Stop words in sentences
from nltk.stem import WordNetLemmatizer  # For stemming the sentence
from nltk.stem import SnowballStemmer  # For stemming the sentence
from contractions import contractions_dict  # To solve contractions
from autocorrect import Speller  # Correcting spellings
import seaborn as sns

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Text cleaning

# Libraries for general purpose


# Data preprocessing


# Clean emojis from text
def strip_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def text_cleaning(text):
    # Remove \n and \r, convert to lowercase
    text = text.replace('\r', '').replace('\n', ' ').lower()

    # Remove mentions and links
    text = re.sub(r'(@\S+|https?:\S+|http?:\S+)', ' ', text)

    # Remove non-UTF-8/ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove punctuation
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)

    # Define stop words (assuming you want to use NLTK's English stop words)
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    text = ' '.join(word for word in text.split()
                    if word.lower() not in stop_words)

    # Remove words longer than 14 characters
    text = ' '.join(word for word in text.split() if len(word) < 14)

    return text


STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm',
                  'im', 'll', 'y', 've', 'u', 'ur', 'don',
                  'p', 't', 's', 'aren', 'kp', 'o', 'kat',
                  'de', 're', 'amp', 'will'])


def remove_stopwords(text):
    return " ".join([word for word in
                     str(text).split() if word not in STOPWORDS])

# clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the "#" symbol


def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split(
        '#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))  # remove last hashtags
    # remove hashtags symbol from words in the middle of the sentence
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2

# Filter special characters such as "&" and "$" present in some words


def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

# Remove multiple sequential spaces


def remove_mult_spaces(text):
    return re.sub("\s\s+", " ", text)

# Stemming


def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

# Lemmatization


def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])


# Initialize spell checker
spell = Speller()


def preprocess_text(text):
    # Your text cleaning and preprocessing steps here
    # For example, removing stop words and applying spell checker
    stop_words = set(stopwords.words('english'))
    words = [spell(word)
             for word in text.split() if word.lower() not in stop_words]

    # Additional preprocessing steps can be added

    # Join the words back into a sentence
    cleaned_text = ' '.join(words)

    return cleaned_text

# Then we apply all the defined functions in the following order


def preprocess(text):
    text = strip_emoji(text)
    text = remove_stopwords(text)
    text = text_cleaning(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = stemmer(text)
    return text
