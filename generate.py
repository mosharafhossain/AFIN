# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import random
import numpy as np
import torch
import argparse
import json
from torch.utils.data import DataLoader
from transformers import (
    T5Tokenizer
)

from model import PInterp
from data import PIDataset
from config import Config
import utils


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
 


argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True)      
args        = argParser.parse_args()
config_path = args.config_path



# Step 1: Read parameters and set a seed ----------------------------------------------------
with open(config_path) as json_file_obj: 
	params = json.load(json_file_obj)

params = Config(params)
set_seed(params.seed)


# Step 2: Get appropriate device ------------------------------------------------------------
if torch.cuda.is_available()==True and params.use_gpu: 
    device = torch.device("cuda:"+str(params.device))
else: 
    device = torch.device("cpu") 



# Step 3: Read test dataset for evaluation--------------------------------------------------
test_data = utils.read_data(params.data_path["test"])


# Step 4: Get tokenizer --------------------------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained(params.T5_path[params.T5_type])



# Step 5: Initialize model and load with the trained weights--------------------------------
map_location = 'cuda:{}'.format(params.device) if params.use_gpu else 'cpu'
state = torch.load(params.best_model_path, map_location=map_location)

model = PInterp(params, device, tokenizer) 
model.load_state_dict(state['model'])
model.to(device)  
model.eval() 
print(f"Model is loaded from {params.best_model_path}")



# Step 5: Prepare the datasets for data loader -------------------------------------------------
test_dataset = PIDataset(tokenizer, test_data, params)

test_size = len(test_dataset)
batch_num = int(np.ceil(test_size/params.batch_size))
print(f"test_size: {test_size}, batch_num: {batch_num}")



# Step 6: Generate text for the test dataset-------------------------------------------------
test_output = []
test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
print(f"Test loader: {len(test_loader)}")

for batch_idx, batch in enumerate(test_loader):                     

    if params.use_multi_gpus:
        test_batch_output = model.module.predict(batch)
    else:
        test_batch_output = model.predict(batch)
    test_output.extend(test_batch_output)  



 
utils.save_prediction_pi_formatted(test_output, params.num_return_sequences, params.output_path["test"])

    