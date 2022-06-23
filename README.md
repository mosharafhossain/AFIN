A Question-Answer Driven Approach to Reveal Affirmative Interpretations from Verbal Negations
============================================================================================================

This repository contains the code and corpus of the NAACL 2022 (Findings) paper "A Question-Answer Driven Approach to Reveal Affirmative Interpretations from Verbal Negations."  
Authors: Md Mosharaf Hossain, Luke Holman, Anusha Kakileti, Tiffany Iris Kao, Nathan Raul Brito, Aaron Abraham Mathews, and Eduardo Blanco  
Arxiv Link: https://arxiv.org/pdf/2205.11467.pdf  


## AFIN Corpus
We share AFIN, a corpus of verbal negations and their affirmative interpretations in the **data** directory.  


## Requirements (Generation System)
Python 3.7 (recommended)  
Python packages: list of packages are provided in ./env-setup/requirements.txt file.  

```bash
# Create Virtual Environment (optional)
python3 -m venv your_location/afin-generation
source your_location/afin-generation/bin/activate

# Install required packages (required)
pip install pip -U
pip install -r ./env-setup/requirements.txt  

# Download T5-Large from https://huggingface.co/t5-large:  
cd ./model/pre-trained/T5-large
wget https://huggingface.co/t5-large/resolve/main/config.json
wget https://huggingface.co/t5-large/resolve/main/pytorch_model.bin
wget https://huggingface.co/t5-large/resolve/main/spiece.model
wget https://huggingface.co/t5-large/resolve/main/tokenizer.json
cd ../../../
```


## How to Run (Generation System)

- Train the affirmative interpretation generation system: 
```bash
  # Setup: without verb
  python train.py --config_path ./config/config_without_verb.json
  
  # Setup: with verb
  python train.py --config_path ./config/config_with_verb.json
  
  + Arguments:
	config_path: path to the configuration file (required)
```
  
- Generate affirmative interpretations from negations:  
```bash 
  # Setup: without verb
  python generate.py --config_path ./config/config_without_verb.json
  
  # Setup: with verb
  python generate.py --config_path ./config/config_with_verb.json
  
  + Arguments:
	config_path: path to the configuration file (required)
```


## How to Run (Natural Language Inference System)  
We utilize the code for the NLI experiments (Section 5.1 in the paper) from https://github.com/mosharafhossain/negation-and-nli


## An Example from AFIN:
An example instance from AFIN is explained below:    
{"sentence": "It was not made by living organisms .", "unique_id": 10, "verb": "made", "neg_cue": "not", "q1": "What was made by something?", "a1": "It", "s1": 4, "q2": "What was something made by?", "a2": "inanimate organisms", "s2": 4, "num_qa": 2, "q0": "Was something made by something?", "a0": "yes", "affirmative interpretation": "It was made by inanimate organisms."}  

Attributes:  
sentence: Sentence containing negation  
verb: (Target) negated verb  
neg_cue: Negation cue  
q0: Predicate question (with yes/no answer) relating to the target verb     
q1-q8: Argument questions  relating to the target verb  
a0: Answer to the predicate question (i.e., yes/no)  
a1-a8: Answers to the argument questions of q1-q8  
s1-s8: Confidence scores of the answers of a1-a8, and   
affirmative interpretation: affirmative interpretation of the sentence containing negation  

