from pathlib import Path
import pandas as pd

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from src.models import RobertaClass
from src.datasets import SentimentData
               
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

basedir = Path.cwd()
test_dir = basedir / "Datasets" / "twitter_validation.csv"

# Model checkpoints
model_checkpoint = "./checkpoint_roberta_S.pth.tar"

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# EPOCHS = 1
LEARNING_RATE = 1e-05

def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    model = torch.load(model_checkpoint)['model'].to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    test_data=pd.read_csv(test_dir, header=None)
    test_data.columns = ["id", "entity", "sentiments", "text"]
    test_data = test_data[['text', 'sentiments']]
    test_data.sentiments = test_data.sentiments.map({"Neutral":0, "Irrelevant":0 ,"Positive":1,"Negative":2})
    testing_set = SentimentData(test_data, tokenizer, MAX_LEN)
    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    testing_loader = DataLoader(testing_set, **test_params)
    acc = valid(model, testing_loader,loss_function)
    print("Accuracy on test data = %0.2f%%" % acc)


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def valid(model, testing_loader,loss_function):
    model.eval()
    n_correct = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu


if __name__ == '__main__':
    main()