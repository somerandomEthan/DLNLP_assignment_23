import numpy as np
import re
import string
# import pandas as pd
import emoji
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Define preprocessing functions
def preprocess_text(text):
    # Convert null/NaN values to empty strings
    if isinstance(text, float) and np.isnan(text):
        text = ''
    # Convert text to lowercase
    # text = text.lower()

    # Remove URLs, user mentions, and hashtags
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))


    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    text = ' '.join(filtered_tokens)
    
    # Replace emoji with text descriptions
    text = emoji.demojize(text, delimiters=(' ', ' '))
    
    return text

