# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import random
import numpy as np
import torch
import tqdm
import argparse
import json
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    Adafactor,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from model import PInterp
from data import PIDataset
from config import Config
import utils
torch.cuda.empty_cache()

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
    
def get_optimizer(params, model):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": params.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if params.optimizer == "AdamW":
        optimizer = AdamW(grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon)
    elif params.optimizer == "Adafactor":
        optimizer = Adafactor(grouped_parameters, lr=params.learning_rate, scale_parameter=False,
                         relative_step=False)
    return optimizer




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



# Step 3: Read train and dev data for the training process-------------------------------------
train_data = utils.read_data(params.data_path["train"])
dev_data   = utils.read_data(params.data_path["dev"])




# Step 4: Get tokenizer and initialize the model------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained(params.T5_path[params.T5_type])
model = PInterp(params, device, tokenizer) 
model.to(device)  
print("Model is successfully loaded!")
state = dict(model=model.state_dict())




# Step 5: Prepare the datasets for data loader -------------------------------------------------
train_dataset = PIDataset(tokenizer, train_data, params)
dev_dataset   = PIDataset(tokenizer, dev_data, params)

train_size = len(train_dataset)
dev_size   = len(dev_dataset)
print("dev_size: {}, batch_num: {}".format(dev_size, int(np.ceil(dev_size/params.batch_size))))


# Step 6: Get optimizer and Scheduler ----------------------------------------------------------
optimizer = get_optimizer(params, model)
batch_num = int(np.ceil(train_size/params.batch_size))
print(f"train_size: {train_size}, batch_num: {batch_num}")
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num * params.warmup_epoch,
                                           num_training_steps=batch_num * params.max_epoch)



# Step 7: Train and save the best model--------------------------------------------------------
min_loss = float('inf')
# Training the model
for epoch in range(params.max_epoch):
    # training set
    progress = tqdm.tqdm(total=batch_num, ncols=75, desc='Train Epoch {}/{}'.format(epoch+1, params.max_epoch ))
    optimizer.zero_grad()
    
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=params.train_shuffle) 
    print(f"\nTrain loader: {len(train_loader)}")
    for batch_idx, batch in enumerate(train_loader):    
        #print(f"train batch: {len(batch)}")            
        loss = model(batch)  
        loss = loss * (1 / params.accumulate_step)
        if params.use_multi_gpus:
            loss.mean().backward()
        else:
            loss.backward()
        
        
        if (batch_idx + 1) % params.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        print(f"batch_idx: {batch_idx}, loss: {loss}")
    
    
    if params.is_model_tuning and (epoch + 1) > params.epochs_no_eval:                  
        dev_output = []
        dev_gold   = []
        
        dev_loader = DataLoader(dev_dataset, batch_size=params.batch_size, shuffle=False) 
        print(f"Dev loader: {len(dev_loader)}")
        dev_loss = 0
        for batch_idx, batch in enumerate(dev_loader):                     
            #model.eval()
            
            temp_loss = model(batch)
            
            dev_loss += temp_loss.item()
            #print(f"batch_idx: {batch_idx}, dev batch: {len(batch)}, dev_loss: {dev_loss/(batch_idx+1)}")   
            
            if params.use_multi_gpus:
                dev_batch_output = model.module.predict(batch)
            else:
                dev_batch_output = model.predict(batch)
            dev_output.extend(dev_batch_output)  
            
        utils.save_prediction(params.num_return_sequences, dev_output, params.output_path["dev"])
        
        current_loss = dev_loss/(batch_idx+1)
        print(f"\nCurrent loss: {current_loss}, min loss: {min_loss}, dev_output: {len(dev_output)}")        
        if current_loss < min_loss:
            min_loss = current_loss            
            print('Saving the best model at {}'.format(params.best_model_path))
            torch.save(state, params.best_model_path)
            patience = 0
        else:
            patience += 1
            if patience > params.patience:
                break
        
            
    progress.close()  

if not params.is_model_tuning: 
    print('Saving the current model at {}'.format(params.best_model_path))
    torch.save(state, params.best_model_path)
    
