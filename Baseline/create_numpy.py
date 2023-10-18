import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import BertModel, RobertaModel
import numpy as np
from tqdm import tqdm
import re
import os


def load_pretrain(path, model):
    checkpoint = torch.load(path)
    mod_missing_keys,mod_unexpected_keys = model.load_state_dict(checkpoint['state_dict'],strict=False)
    print(mod_missing_keys)
    print(mod_unexpected_keys)
    return model

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub("[\(\[].*?[\)\]]", '', text)

    # Replace '&amp;' with '&'
    text = re.sub(" +",' ', text).strip()

    return text

path = ""

csv = pd.read_csv("./Train.csv")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base").cuda()

model = load_pretrain(model,path)

for i, row in tqdm(csv.iterrows()):

    tokenized_sentence = tokenizer(text_preprocessing(row["tweet"])).input_ids
    tokenized_sentence = torch.tensor(tokenized_sentence).unsqueeze(0).cuda()

    output = model(tokenized_sentence)[0]

    output = output.cpu().detach().numpy()

    np.save("./baseline/" + str(row["FileName"]), output)

    



