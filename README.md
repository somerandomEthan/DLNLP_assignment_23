# README

# DLNLP_23_SN22081179
This is ELEC 0141 Deep Learning for Natural Language Processing 22/23 project which aimed at doing the sentiment analysis on tweets. The datasets are the datasets from [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) and [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140). 

## 1. Prerequisites
To Begin with, it is required to clone this project or download to your computer or server. The structure of the folders are as follows:

### DLNLP_assignment_23

* [Datasets/](.\DLNLP_assignment_23\Datasets)
  * [training.1600000.processed.noemoticon.csv](.\DLNLP_assignment_23\Datasets\training.1600000.processed.noemoticon.csv)
  * [twitter_training.csv](.\DLNLP_assignment_23\Datasets\twitter_training.csv)
  * [twitter_validation.csv](.\DLNLP_assignment_23\Datasets\twitter_validation.csv)

* Datasets/
  * training.1600000.processed.noemoticon.csv
  * twitter_training.csv
  * twitter_validation.csv
* [src/](./DLNLP_assignment_23/src)
  * [datasets.py](./DLNLP_assignment_23/src/datasets.py)
  * [models.py](./DLNLP_assignment_23/src/models.py)
  * [preprocess.py](./DLNLP_assignment_23/src/preprocess.py)
* [environment.yml](./DLNLP_assignment_23/environment.yml)
* [eval_roberta.py](./DLNLP_assignment_23/eval_roberta.py)
* [eval_roberta_S.py](./DLNLP_assignment_23/eval_roberta_S.py)
* [main.py](./DLNLP_assignment_23/main.py)
* [nohup.out](./DLNLP_assignment_23/nohup.out)
* [README.md](./DLNLP_assignment_23/README.md)
* [train_roberta.py](./DLNLP_assignment_23/train_roberta.py)
* [train_roberta_S.py](./DLNLP_assignment_23/train_roberta_S.py)
* checkpoint_roberta_B.pth.tar
* checkpoint_roberta_S.pth.tar

The folder Datasets is not included, you can either download the required Datesets according to the file tree or you can download the folder [here](https://drive.google.com/drive/folders/1-iRiL7OSCDMyGSAuPPsmU7VIUg2gWGig?usp=sharing). The checkpoints are also not included and can also be downloaded [here](https://drive.google.com/drive/folders/1z4PTeLTGj6SmAUFeg4Y5w3Bv_3gR7b6K?usp=sharing) 

### The environment

My advice is to create a new conda environment from the `environment.yml` file in this repo [environment.yml](./environment.yml)
You can simply do it by: 

```bash
conda env create -f environment.yml
```

## 2. How to check the result of this project

### If your server or computer is GPU ready, then you can proceed to check the RoBERTa Results. Or you can also refer to the [nohup.out](./DLNLP_assignment_23/nohup.out) which is the output from the server

(For the project I used turin.ee.ucl.ac.uk, I checked whether it works on this server, I strongly suggest that you test the project on the server as RoBERTa is a huge model)
You can simply check by input:

```python
torch.cuda.is_available()
```

If the output is `True`, then congratulation that you can start from the training by exectuting the corresponding file. You can simply download the checkpoints and the run the [eval_roberta_S.py](./DLNLP_assignment_23/eval_roberta_S.py) to test the results on Kaggle dataset or run the [eval_roberta.py](./DLNLP_assignment_23/eval_roberta.py) to check the result on Sentiment140. Or you can simply fine tuning the model form scratch and genrate your own checkpoints via runing [train_roberta.py](./DLNLP_assignment_23/train_roberta.py) The RoBERTa code is written seperately so you can train and test it on GPU server

### To check the result of the Logistic Regression and Random Forest:

you can simply run [main.py](./DLNLP_assignment_23/main.py)

#### Testing the Kaggle dataset:

You need to comment out the function which obtains the Sentiment140 dataset. The code should be like:

```python
def main():
    """
    Main function
    """
    train_df, test_df = kaggle_data()
    # train_df, test_df = sentiment140_data()
    logistic_regression(train_df, test_df)
    random_forest(train_df, test_df)
```

#### Testing the Sentiment140 dataset:


You need to comment out the function which obtains the Kaggle dataset. The code should be like:

```python
def main():
    """
    Main function
    """
    # train_df, test_df = kaggle_data()
    train_df, test_df = sentiment140_data()
    logistic_regression(train_df, test_df)
    random_forest(train_df, test_df)
```

