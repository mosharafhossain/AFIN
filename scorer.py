# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

import numpy as np
import argparse
import utils
import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import single_meteor_score
import os

class Scorer():
    def get_sentence_blue_smoothing(self, ref_list, sys_list, is_smoothing=True):
        """
        Calculate sentence-level blue score with smoothing
        :param ref_list: list of reference texts. The texts are not tokenized.
        :param sys_list: list of generated texts. The texts are not tokenized.
        # smoothing references: 
            http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf 
            https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.SmoothingFunction.method3
        """
        size = len(ref_list)
        assert size == len(sys_list)
        
        score_list = []
        for i in range(size):
            ref = ref_list[i].split()
            sys = sys_list[i].split()
            if is_smoothing:
                sf = SmoothingFunction()
                score_list.append(sentence_bleu([ref], sys, smoothing_function=sf.method2) )
            else:
                score_list.append( sentence_bleu([ref], sys) )
        print(f"bleu score_list: {score_list[0:20]}")
        return np.mean(score_list)
    
    def get_sentence_blue(self, ref_list, sys_list):
        """
        https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
        """
        size = len(ref_list)
        assert size == len(sys_list)
        
        bleu1 = []
        bleu2 = []
        bleu3 = []
        bleu4 = []
        bleu  = {}
        for i in range(size):
            ref = ref_list[i].split()
            sys = sys_list[i].split()
            bleu1.append( sentence_bleu([ref], sys, weights=(1, 0, 0, 0)) )
            bleu2.append( sentence_bleu([ref], sys, weights=(0.5, 0.5, 0, 0)) )
            bleu3.append( sentence_bleu([ref], sys, weights=(0.33, 0.33, 0.33, 0)) )
            bleu4.append( sentence_bleu([ref], sys, weights=(0.25, 0.25, 0.25, 0.25)) )
            
        bleu = {"bleu1": round(np.mean(bleu1),4), "bleu2": round(np.mean(bleu2),4),
                "bleu3": round(np.mean(bleu3),4), "bleu4": round(np.mean(bleu4),4)}    
        return bleu
        
    def get_sentence_chrfpp(self, ref_list, sys_list):
        """
        Calculate sentence-level CHRF++ score
        Ref: https://github.com/m-popovic/chrF
             https://aclanthology.org/W15-3049.pdf
        :param ref_list: list of reference texts. The texts are not tokenized.
        :param sys_list: list of generated texts. The texts are not tokenized.
        """
        size = len(ref_list)
        assert size == len(sys_list)
        
        score_list = []
        for i in range(size):
            ref = ref_list[i]
            sys = sys_list[i]
            score_list.append( sentence_chrf(ref, sys) )
        
        return round(np.mean(score_list), 4)
    
    
    def get_sentence_meteor(self, ref_list, sys_list):
        """
        Calculate sentence-level meteor score
        :param ref_list: list of reference texts. The texts are not tokenized.
        :param sys_list: list of generated texts. The texts are not tokenized.
        """
        size = len(ref_list)
        assert size == len(sys_list)
        
        score_list = []
        for i in range(size):
            ref = ref_list[i]
            sys = sys_list[i]
            score_list.append( single_meteor_score(ref, sys) )
        
        return round(np.mean(score_list), 4)
    

def prepare_ref_sys_qa_list(gold_json_list, sys_json_list):
    """
    :param gold_json_list: list of json data for the gold texts
    :param sys_json_list: list of json data for the system generated texts
    """
    ref_list = []
    sys_list = []
    for i in range(len(gold_json_list)):
        ref = gold_json_list[i]
        sys = sys_json_list[i]
        
        for idx in range(1, 9): # 1-8: number of possible questions/answers
         if "a"+str(idx) in ref  and "a"+str(idx) in sys:
             ref_list.append(ref["a"+str(idx)])
             sys_list.append(sys["a"+str(idx)])
             
    return ref_list, sys_list


def prepare_ref_sys_pi_list(gold_json_list, sys_json_list):
    """
    :param gold_json_list: list of json data for the gold texts
    :param sys_json_list: list of json data for the system generated texts
    """
    ref_list = []
    sys_list = []
    for i in range(len(gold_json_list)):
        ref_dict = gold_json_list[i]
        sys_dict = sys_json_list[i]
        
        ref_list.append(ref_dict["affirmative interpretation"])
        sys_list.append(sys_dict["affirmative interpretation"])
        
    assert len(ref_list) == len(sys_list)         
    return ref_list, sys_list
             
def create_files_bertscore(ref_list, sys_list, dir_path):
    
    assert len(ref_list) == len(sys_list)
    ref_obj = open( os.path.join(dir_path, "ref.txt"), "w", encoding="utf-8")
    sys_obj = open( os.path.join(dir_path, "sys.txt"), "w", encoding="utf-8")
    for ref_sent, sys_sent in zip(ref_list, sys_list):
        ref_obj.write(ref_sent)
        ref_obj.write("\n")
        
        sys_obj.write(sys_sent)
        sys_obj.write("\n")
        
    ref_obj.close()
    sys_obj.close()
    
    
if __name__ == "__main__":
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-g", "--gold_file", help="path of the ground truth file", required=True)
    argParser.add_argument("-s", "--system_file", help="path of the file generated by the system", required=True)
    argParser.add_argument("-t", "--task_name", help="name of the task", required=True)
    args        = argParser.parse_args()
    gold_file   = args.gold_file
    system_file = args.system_file
    task_name   = args.task_name
    
    gold_data = utils.read_data(gold_file)
    system_data = utils.read_data(system_file)
    

    ref_list, sys_list = prepare_ref_sys_pi_list(gold_data, system_data)        
    bleu_score = Scorer().get_sentence_blue(ref_list, sys_list)
    chrfpp_score = Scorer().get_sentence_chrfpp(ref_list, sys_list)
    meteor_score = Scorer().get_sentence_meteor(ref_list, sys_list)
    print(f"\nbleu_score: {bleu_score}, \nCHRF++: {chrfpp_score}, \nmeteor_score: {meteor_score}")
    
    
    
            