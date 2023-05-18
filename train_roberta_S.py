from pathlib import Path
import pandas as pd

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from src.models import RobertaClass_S
from src.datasets import SentimentData


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

basedir = Path.cwd()
train_dir = basedir / "Datasets" / "twitter_training.csv"
test_dir = basedir / "Datasets" / "twitter_validation.csv"



# train_loss_list = []
# Traning_PSNR_list = []
# Validation_PSNR_list = []
# valid_loss_list = []


# Defining some key variables that will be used later on in the training
train_size = 0.8
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# EPOCHS = 1
LEARNING_RATE = 1e-05



def main():

    # Reading the whole dataset from the csv file
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    # whole_df = pd.read_csv(df_test, header=None, names=['sentiments', 'ids', 'date', 'flag', 'user', 'text'], 
    #                        encoding='latin-1')
    # whole_df = whole_df[['text', 'sentiments']]


    train_data=pd.read_csv(train_dir, header=None)
    # test_data=pd.read_csv(train_dir, header=None)
    train_data.columns = ["id", "entity", "sentiments", "text"]
    # test_data.columns = ["id", "entity", "sentiments", "text"]
    train_data = train_data[['text', 'sentiments']]
    # test_data = test_data[['text', 'sentiments']]
    train_data.sentiments = train_data.sentiments.map({"Neutral":0, "Irrelevant":0 ,"Positive":1,"Negative":2})
    # test_data.sentiments = test_data.sentiments.map({"Neutral":0, "Irrelevant":0 ,"Positive":1,"Negative":2})
    # # train_data.to_pickle('train_data.pkl')
    # test_data.to_pickle('test_data.pkl')

    #loading the training data and testing data and do the corresponding tokenization
    training_set = SentimentData(train_data, tokenizer, MAX_LEN)
    # testing_set = SentimentData(test_data, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    # test_params = {'batch_size': VALID_BATCH_SIZE,
    #                 'shuffle': True,
    #                 'num_workers': 0
    #                 }

    training_loader = DataLoader(training_set, **train_params)
    # testing_loader = DataLoader(testing_set, **test_params)

    # Initialize model
    model = RobertaClass_S()
    # Initialize the optimizer
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    # Move to default device
    model = model.to(device)
    # Creating the loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Training loop
    EPOCHS = 5
    for epoch in range(EPOCHS):
        train(training_loader, model, loss_function, optimizer, epoch)



        # Save checkpoint
        torch.save({'epoch': epoch,  
                'model': model,
                'optimizer': optimizer},
                'checkpoint_roberta_S.pth.tar')

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def train(training_loader, model, loss_function, optimizer, epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return


if __name__ == '__main__':
    main()