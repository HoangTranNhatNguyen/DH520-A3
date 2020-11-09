# Marking criteria:
# - Speed
# - Accuracy

import os
import re
import numpy as np
import editdistance
from tqdm import tqdm

# Tokenize long strings into tokens 
def tokenize(text):
    tok_list = []
    for tok in text.lower().split():
        tok_list.append(tok)
    return tok_list

# Filter punctuations from strings
def filter(text, pattern='[:;.,!?\"]', repl=' '):
    text = re.sub(pattern,repl,text)
    return text

def find_best_match(input, word_list): # O(N)
    min_score = 1e9
    min_index = -1
    for i, word in enumerate(word_list):
        score = editdistance.eval(input, word)
        if score < min_score:
            min_score = score
            min_index = i
    return word_list[min_index]
    
# O(N) operations
words_file = open('words.txt', 'r')
lexis_file = open('lexicon.txt', 'r')
bad_ocr_file = open('raven_ocr.txt', 'r')
ref_ocr_file = open('Poe_Raven_clean.txt', 'r')

word_list = words_file.read().splitlines()
lexi_list = lexis_file.read().splitlines()
comb_list = word_list + lexi_list

bad_document = bad_ocr_file.readlines()
ref_document = ref_ocr_file.readlines()

word_set = set(comb_list)
logs = {}
outputs = []

for line in tqdm(bad_document):
    filtered_line = filter(line)
    tokenized_line = tokenize(filtered_line)
    
    output = []
    for token in tokenized_line:
        if token not in comb_list:
            best_word = find_best_match(token, comb_list)
        else:
            best_word = token
        
        output.append(best_word)
        logs[token] = best_word

    outputs.append(output)

for line in outputs:
    print(' '.join(line))