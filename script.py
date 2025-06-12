import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataloader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os


s3_path = 's3://ism-multiclass-textclassification-bucket/training-data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep = '\t', names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[['TITLE','CATEGORY']]

my_dict = {
    'e' : 'Entertainment',
    'b' : 'Business',
    't' : 'Science',
    'm' : 'Health'
}

def update_cat(x):
    return my_dict[x]

df['CATEGORY'] = df['CATEGORY'].apply(lambda x:update_cat(x))

print(df)

# *Making a fraction of the dataset

df = df.sample(frac = 0.05, random_state = 1)
df = df.reset_index(drop = True)

# Categorical Encoding using Integer Encoding

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x : encode_cat(x))

encode_dict.keys()

df





