# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""
from torch.utils.data import Dataset

class PIDataset(Dataset):
    def __init__(self, tokenizer, data_list, params):
        self.input_len = params.input_len[params.problem_name]
        self.target_len = params.target_len[params.problem_name]
        self.tokenizer = tokenizer
        self.verb_included = params.verb_included
    
        self.inputs = []
        self.targets = []

        self._process(data_list)
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  
        target_mask = self.targets[index]["attention_mask"].squeeze()  

        return {"source_ids": source_ids, 
                "source_mask": src_mask, 
                "target_ids": target_ids, 
                "target_mask": target_mask}

    def _process(self, data_list):
        for data_dict in data_list:                    
            if self.verb_included:                             
                input_ = "verb: {} sentence: {}".format(data_dict["verb"].strip(), data_dict["sentence"].strip())                
            else:
                input_ = "sentence: {}".format(data_dict["sentence"].strip())                
            input_ = input_ + ' </s>'
    
            target_ = "afin: {}".format( data_dict["affirmative interpretation"].strip() )
            target_ = target_ + " </s>"
    
            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
              [input_], max_length=self.input_len, padding='max_length', truncation=True, return_tensors="pt"
            ) # or, padding='longest' or max_length
            
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
              [target_], max_length=self.target_len, padding='max_length', truncation=True, return_tensors="pt"
            ) # or, padding='longest' or max_length
    
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            
        
            
            
            
            
            