import torch
from torch.utils.data import DataLoader as DL
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
import numpy as np
import argparse
from transformers import AutoTokenizer
import torch

from engine import Engine
import pandas as pd
from utils import *

from create_torch_dataset import CreateTweetsDataset
# import optuna
# from optuna.trial import TrialState

from model import TransformerModel
import json
import random

seed = 0
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def parse():
    parser = argparse.ArgumentParser(description="Train and test the model for training toxicity classifier")
    parser.add_argument('--train_batch_size',help='Set the training batch size',type=int,default=8)
    parser.add_argument('--test_batch_size',help='Set the testing batch size',type=int,default=8)
    parser.add_argument('--model',help='Select the transformer model',default='google/muril-large-cased')
    parser.add_argument('--lr',help = 'Learning rate set',type=float,default=0.001)
    parser.add_argument('--train_data',help='Training Dataset Location',type = str,default="/fs/nexus-scratch/sonalkum/baseline/CONAN/DIALOCONAN/train.csv",required = False)
    parser.add_argument('--test_data',help='Test Dataset Location',type = str,required=False)
    parser.add_argument('--checkpoint',help='Set the location for checkpoint saving',type = str,default='checkpoints')
    parser.add_argument('--starting_epoch',help='Starting epoch for training ',type=int,default=0)
    parser.add_argument('--max_epochs',help='Set the maximum epochs to train the model on ',type = int,default=3)
    global args
    args = parser.parse_args()

def read_data(train_data, valid_data, test_data):
    # train_data = pd.read_csv(train_data)
    # test_data = pd.read_csv(test_data)
    # train_data.dropna(axis='columns', inplace=True)
    # test_data.dropna(axis='columns',inplace = True)
    train_data.tweet = train_data.tweet.apply(text_preprocessing)
    valid_data.tweet = valid_data.tweet.apply(text_preprocessing)
    test_data.tweet = test_data.tweet.apply(text_preprocessing)
    train_data.label = train_data.label.apply(binarise)
    valid_data.label = valid_data.label.apply(binarise)
    test_data.label = test_data.label.apply(binarise)
    train_data = train_data[['tweet','label']]
    valid_data = valid_data[['tweet','label']]
    test_data = test_data[['tweet','label']]
    return train_data,valid_data,test_data


def process_for_transformer(datasets):
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    processed_data = []
    for dataset in datasets:
        batch = tokenizer(list(dataset['tweet']), padding=True, truncation=True, return_tensors="pt",max_length=512)
        data = {key:value.clone() for key,value in batch.items()}
        data['label'] = list(dataset['label'])
        data.pop('token_type_ids', None)
        processed_data.append(data)
    return processed_data


if __name__=='__main__':
    parse()

    # df_emotion = pd.read_csv(args.train_data)

    # for i in range(1,6):

    stats_file = open('muril_stats_conan.txt', 'a', buffering=1)

    train_data_csv = pd.read_csv("./train.csv")
    valid_data_csv = pd.read_csv("./val.csv")
    test_data_csv = pd.read_csv("./test.csv")

    train_data,val_data,test_data = read_data(train_data_csv,valid_data_csv,test_data_csv)
    train_data,val_data,test_data = process_for_transformer([train_data,val_data,test_data])

    train_data = CreateTweetsDataset(train_data)
    val_data = CreateTweetsDataset(val_data)
    test_data = CreateTweetsDataset(test_data)

    global train_loader,val_loader,test_loader

    train_loader = DL(train_data,batch_size=args.train_batch_size,shuffle = True)
    val_loader = DL(val_data,batch_size=args.test_batch_size,shuffle = True)
    test_loader = DL(test_data,batch_size=args.test_batch_size,shuffle = True)

    state = {'max_epochs':args.max_epochs,'start_epoch':args.starting_epoch}
    model = TransformerModel(args.model)
    # optimizer = AdamW(model.parameters(),
    #                 lr = args.lr,
    #                 eps = 1e-8
    #
    #                  )
    #model  = AutoModel.from_pretrained(args.model)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * (args.max_epochs-args.starting_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    engine = Engine(state)

    metrics_dict = engine.learn(train_loader,val_loader,model,optimizer,scheduler,test_loader)

    stats = dict(f1 = float(metrics_dict['f1_score'].detach().cpu().numpy()), accuracy = float(metrics_dict['accuracy'].detach().cpu().numpy()), best_score = float(metrics_dict['best_score_test'].cpu().numpy()))
    print(json.dumps(stats), file=stats_file)

    print("Test f1 score :: ",metrics_dict['f1_score'])
    print("Test accuracy ::",metrics_dict['accuracy'])
