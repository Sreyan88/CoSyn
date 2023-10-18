
from tqdm import tqdm 
#run from baseline directory
path = '/fs/nexus-scratch/sonalkum/baseline/Test_muril_test_cad_model.pt'
import torch
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
import numpy as np
import json
import os 
import argparse
from transformers import AutoTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch

#from engine import Engine
import pandas as pd
from utils import *

import pandas as pd 
from create_torch_dataset import CreateTweetsDataset
# import optuna
# from optuna.trial import TrialState

from model import Transformer

from transformers import AutoTokenizer , AutoConfig

tokenizer = AutoTokenizer.from_pretrained("google/muril-large-cased")
config = AutoConfig.from_pretrained("google/muril-large-cased")
model = Transformer("google/muril-large-cased")
a,b= model.load_state_dict(torch.load(path,map_location='cpu'))
model =model.cuda()



def get_embedding(text):
  #print(tokenizer.convert_ids_to_tokens(tokenizer.encode(text)))
    
  input_encoded = tokenizer.encode_plus(text,max_length=150, padding='max_length', truncation=True, return_tensors="pt")
  with torch.no_grad():
    states = model(input_ids= input_encoded["input_ids"].cuda(), attention_mask = input_encoded["attention_mask"].cuda() ).hidden_states
  output = torch.stack([states[i] for i in range(len(states)-4, len(states))])
  output = states[-1]
  output = torch.mean(output, dim=0)
  output = torch.mean(output, dim =0)
  print("Output shape is {}".format(output.shape))
  return output




out_path = "./tweet_embed/"
import csv
import pandas as pd

df = pd.read_csv("") # path to csv 
import pandas as pd

for i in tqdm(range(len(df))):
    text = df['tweet'][i]
    file = df['id'][i]
    if pd.isnull(text):
      text=""

    x = get_embedding(text)
    outfile=out_path+str(file)+".pt"
    # print(outfile)
    torch.save(x,outfile)
    print(x.size())

