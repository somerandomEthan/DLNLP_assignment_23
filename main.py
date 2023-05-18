# Importing the libraries needed
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from src.preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Data parameters
basedir = Path.cwd()
train_dir = basedir / "Datasets" / "twitter_training.csv"
test_dir = basedir / "Datasets" / "twitter_validation.csv"
df_test = basedir / "Datasets" / "training.1600000.processed.noemoticon.csv"

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
BATCH_SIZE = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e4  # number of training iterations
WORKERS = 4  # number of workers for loading data in the DataLoader
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate



def kaggle_data():
    test_df = pd.read_csv(test_dir, header=None)
    train_df = pd.read_csv(train_dir, header=None)
    train_df.columns = ["id", "entity", "sentiment", "text"]
    test_df.columns = ["id", "entity", "sentiment", "text"]
    train_df = train_df[['text', 'sentiment']]
    test_df = test_df[['text', 'sentiment']]
    train_df.sentiment = train_df.sentiment.map({"Neutral":0, "Irrelevant":0 ,"Positive":1,"Negative":2})
    test_df.sentiment = test_df.sentiment.map({"Neutral":0, "Irrelevant":0 ,"Positive":1,"Negative":2})
    train_df['text'] = train_df['text'].apply(preprocess_text)
    test_df['text'] = test_df['text'].apply(preprocess_text)
    # Save the dataframes for further use
    train_df.to_pickle('train_data_S.pkl')
    test_df.to_pickle('test_data_S.pkl')
    return train_df, test_df

def sentiment140_data():
    df = pd.read_csv(df_test, header=None, names=['sentiment', 'ids', 'date', 'flag', 'user', 'text'], 
                           encoding='latin-1')
    df = df[['text', 'sentiment']]
    df['text'] = df['text'].apply(preprocess_text)
    df.sentiment = df.sentiment.map({0:0, 4:1})
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    # Split the data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Save the dataframes for further use
    train_df.to_pickle('train_data_B.pkl')
    test_df.to_pickle('test_data_B.pkl')
    return train_df, test_df
    


def logistic_regression(train_df, test_df):
    '''Logistic Regression with feature extraction'''
    tfidf = TfidfVectorizer(max_features=10000)
    X_train = tfidf.fit_transform(train_df['text'])
    X_test = tfidf.transform(test_df['text'])
    y_train = train_df['sentiment']
    y_test = test_df['sentiment']

    # Train a logistic regression model on the training data
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # Predict the sentiment of the validation data using the trained model
    y_pred = lr.predict(X_test)

    # Calculate the accuracy of the model on the validation data
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy score
    print("Accuracy of Logistic Regression:", accuracy)
    return


def random_forest(train_df, test_df):
    '''Random Forest with feature extraction'''
    tfidf = TfidfVectorizer(max_features=10000)
    X_train = tfidf.fit_transform(train_df['text'])
    X_test = tfidf.transform(test_df['text'])
    y_train = train_df['sentiment']
    y_test = test_df['sentiment']

    # Train a logistic regression model on the training data
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    # Predict the sentiment of the validation data using the trained model
    y_pred = rf.predict(X_test)

    # Calculate the accuracy of the model on the validation data
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy score
    print("Accuracy of Random Forest:", accuracy)
    return


def main():
    """
    Main function
    """
    train_df, test_df = kaggle_data()
    # train_df, test_df = sentiment140_data()
    logistic_regression(train_df, test_df)
    random_forest(train_df, test_df)
    
if __name__ == '__main__':
    main()

