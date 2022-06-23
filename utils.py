# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import json

def read_data(file_path):
    """
    Read a file that contains json data as dictionary in each line.
    :param file_path: path to the input file
    """
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            data_list.append(json.loads(line))
            
    return data_list


def write_data(data_list, file_path):
    """
    Write json data into a file. Each json object (dict) in a each line.
    :param data_list: list of data containing json objects
    :param file_path: file path to write the data
    """
    
    with open(file_path, "w", encoding="utf-8") as file_obj:
        for json_obj in data_list:
            file_obj.write(json.dumps(json_obj))
            file_obj.write("\n")


def save_prediction(num_return_sequences, outputs, file_path):
    """
    Save predicted outputs by the model.
    :param num_return_sequences: number of generated text per input.
    :param outputs: list of predicted output
    :param file_path: path to the output file
    """
    

    with open(file_path, "w", encoding="utf-8") as file_obj:
        count = 0
        gen_texts = {}
        for text in outputs:
            gen_texts[count] = text.replace("<pad>", "").replace("</s>", "").strip()
            count += 1
                
            if count == num_return_sequences:
                file_obj.write(json.dumps(gen_texts))
                file_obj.write("\n")
                count = 0
                gen_texts = {}
                

def save_prediction_qa_formatted(data_list, outputs, num_ret_seq, file_path):
    """
    Save predicted outputs for question-answering problem.
    :param data_list: original list of data. Each element is a dictionary containing annotation per instance. 
    :param outputs: list of predicted output. Each element is a dictionary containg n number of generated sequences for each input
    :param num_ret_seq: number of return sequences from the model while predicting
    :param file_path: path to the output file
    """
    
    with open(file_path, "w", encoding="utf-8") as file_obj:
        curr_indx = 0
        for idx, data_dict in enumerate(data_list):
            num_qa = data_dict["num_qa"]
            gen_texts = {}
            for i in range(num_qa):
                output = outputs[curr_indx]
                gen_texts["a"+str(i+1)] = output.replace("<pad>", "").replace("</s>", "").replace("answer:", "").strip()  # taking the fist predicted sequence by output_dict[0]
                curr_indx = curr_indx + num_ret_seq
                
            file_obj.write(json.dumps(gen_texts))
            file_obj.write("\n")
            
            
def save_prediction_pi_formatted(outputs, num_ret_seq, file_path):
    """
    Save predicted outputs from the positive interpretation generation problem.    
    :param outputs: list of predicted output. Each element is a dictionary containg n number of generated sequences for each input
    :param num_ret_seq: number of return sequences from the model while predicting
    :param file_path: path to the output file
    """
    
    
    with open(file_path, "w", encoding="utf-8") as file_obj:
        gen_texts = {}
        for i, text in enumerate(outputs):            
            if i% num_ret_seq == 0:  # system generates num_ret_seq (e.g.,3) texts for each input.
                gen_texts["affirmative interpretation"] = text.replace("<pad>", "").replace("</s>", "").replace("afin:", "").strip()
                file_obj.write(json.dumps(gen_texts))
                file_obj.write("\n")
                gen_texts = {}
            

def show_few_processed_data(tokenizer, train_dataset, dev_dataset, train_points=[0,1], dev_points=[0,1] ):
    """
    Show few examples from the datasets after tokenization.
    """
    for i in train_points:
        print(f"Training Data - index {i}")
        data = train_dataset[i]
        print(tokenizer.decode(data['source_ids']))
        print(tokenizer.decode(data['target_ids']))
        
    for i in dev_points:
        print(f"Dev Data - index {i}")
        data = dev_dataset[i]
        print(tokenizer.decode(data['source_ids']))
        print(tokenizer.decode(data['target_ids']))                
    

if __name__ == "__main__":
    print("") 
   