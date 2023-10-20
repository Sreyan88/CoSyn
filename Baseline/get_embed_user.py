from tqdm import tqdm 
#run from baseline directory
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
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer , AutoConfig
import re
from sentence_transformers import SentenceTransformer, util
import csv
import pickle
model = SentenceTransformer('.pt')# model checkpoint path
model.cuda()

def get_embedding(text):
    
  text = re.sub(r'(@.*?)[\s]', '', text)
  embedding = model.encode(text)
  return torch.from_numpy(embedding)


out_path = "./user_embed/"
in_path = "./last_tweets/"
A =os.listdir(in_path)

if __name__ == "__main__":
    for i in tqdm(range(len(A))) :
        
        file = A[i]
        testfile = in_path+"/"+ file
        outfile=out_path+"/"+os.path.splitext(file)[0]+".pt"
        lines_=[]
        # print(testfile)
        with open(testfile, mode ='r')as f:
            csvFile = csv.reader(f)

            for lines in csvFile:
                lines_+=lines

        embedding=[]
        for val in lines_[1:]:#to remove "tweet"
          embedding.append(get_embedding(val))
          # print(outfile)
        torch.save(embedding,outfile)
        # print(embedding[0].shape)
        print(len(embedding))

