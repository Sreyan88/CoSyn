#run from baseline directory
CUDA_LAUNCH_BLOCKING=1.
path = '.pt' # path to best model ckpt file
import torch
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
import numpy as np
import json
import os
import argparse
from transformers import AutoTokenizer
import torch

#from engine import Engine
import pandas as pd
from utils import *

import pandas as pd
from create_torch_dataset import CreateTweetsDataset
# import optuna
# from optuna.trial import TrialState

from model import TransformerModel

from transformers import *

# from main import read_data, process_for_transformer
from torch.utils.data import DataLoader as DL
from engine import Engine

tokenizer = AutoTokenizer.from_pretrained("./baseline/bert-multi-conversational-hate-sentence-transformer")
config = AutoConfig.from_pretrained("./baseline/bert-multi-conversational-hate-sentence-transformer/sentence_bert_config.json")
model = TransformerModel("./baseline/bert-multi-conversational-hate-sentence-transformer")
model.load_state_dict(torch.load(path,map_location='cuda'))
model = model.cuda()
def read_data(train_data, valid_data, test_data):
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
    tokenizer = AutoTokenizer.from_pretrained("./baseline/bert-multi-conversational-hate-sentence-transformer")
    processed_data = []
    for dataset in datasets:
        batch = tokenizer(list(dataset['tweet']), padding=True, truncation=True, return_tensors="pt")
        data = {key:value.clone() for key,value in batch.items()}
        data['label'] = list(dataset['label'])
        data.pop('token_type_ids', None)
        processed_data.append(data)
    return processed_data

def get_embedding(text):
  #print(tokenizer.convert_ids_to_tokens(tokenizer.encode(text)))

  input_encoded = tokenizer.encode_plus(text,max_length=150, padding='max_length', truncation=True, return_tensors="pt")
  with torch.no_grad():
    states = model(**input_encoded).hidden_states
  output = torch.cat(tuple([states[i] for i in range(len(states))]))
  pooled_output = torch.mean(output,0)
  pooled_ = torch.mean(pooled_output,0)

  return pooled_
df = pd.read_csv("./test.csv")
df2 = pd.read_csv("./train.csv")
_,_,test_data = read_data(df2,df2,df)
'''# if intrinsic is detected 
for i in range(len(df)):
    if df["annotation"][i] == "N":
        test_data["label"][i]= 1
    else:
        test_data["label"][i] =0 
'''
print("Read ")
print(len(test_data))
test_data = process_for_transformer([test_data])[0]
test_data = CreateTweetsDataset(test_data)
test_loader = DL(test_data,batch_size=1,shuffle = True)
state = {'max_epochs':1,'start_epoch':0}
engine = Engine(state)

metrics_dict = engine.test(test_loader,model)
print(metrics_dict)
#stats = dict(f1 = float(metrics_dict['f1_score'].detach().cpu().numpy()), accuracy = float(metrics_dict['accuracy'].detach().cpu().numpy()), best_score = float(metrics_dict['best_score_test'].detach().cpu().numpy()))
# print(json.dumps(stats), file=stats_file)

print("Test f1 score :: ",metrics_dict['f1_score'])
print("Test accuracy ::",metrics_dict['accuracy'])
