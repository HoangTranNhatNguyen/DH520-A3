# Authors:
# - Tran Nhat Hoang Nguyen
# - Cheryl Walker

# import re
# -> re is RegularExpression library
# -> it helps preprocessing texts
# import tqdm
# -> tqdm is an external library similar to a Progress Bar
# -> we can use it to measure how long the code run
# -> Installation: pip install tqdm

# Automated Language Scores:
# Metric: Before correcting --> After correcting
# Bleu_1: 0.604098 --> 0.618107
# Bleu_2: 0.519436 --> 0.536700
# Bleu_3: 0.450275 --> 0.470299
# Bleu_4: 0.389150 --> 0.409721
# METEOR: 0.420514 --> 0.429363
# ROUGE_L: 0.581968 --> 0.595932
# CIDEr: 3.560423 --> 3.725965

import re
import numpy as np
import editdistance
from tqdm import tqdm

# Tokenize long strings into tokens
# Example: 'I am doing my home work' --> ['i', 'am', 'doing', 'my', 'home', 'work']
def tokenize(text):
    tok_list = []
    for tok in text.lower().split():
        tok_list.append(tok)
    return tok_list

# Filter punctuations from strings
# Example: 'Good morning! You too.' --> 'Good morning You too'
def filter(text, pattern='[:;.,!?\"]', repl=' '):
    text = re.sub(pattern,repl,text)
    return text

# Find the best candidate in dictionary for a given input token
# Example: wofd --> word (edit_distance = 1)
# Time complexity: O(N) worst case where N is dictionary size.
def find_best_match(input, word_list):
    min_score = 1e9 # Infinity
    min_index = -1 # Save the best word
    for i, word in enumerate(word_list):
        score = editdistance.eval(input, word)
        if score < min_score:
            min_score = score
            min_index = i
    return word_list[min_index]
    
if __name__ == "__main__":
    # Input vocabulary
    file_words = open('words.txt', 'r')
    w_list = file_words.read().lower().splitlines()
    file_lexicon = open('lexicon.txt', 'r')
    l_list = file_lexicon.read().lower().splitlines()
    # Top 20k high-frequency words sorted by frequency
    file_freqwords = open('20k.txt', 'r') 
    h_list = file_freqwords.read().lower().splitlines()
    # Handle number cases
    n_list = [str(i) for i in range(100)] 

    # Quickly search if one token is in dictionary
    # Time complexity of combining two dictionaries: O(max(N,M)) 
    vocab_set = set(h_list + w_list + l_list + n_list) 
    # Control the word list of dictionary
    vocab_list = list(h_list + w_list + l_list + n_list)

    # Read the OCR file
    ocr_file = open('raven_ocr.txt', 'r')
    ocr_document = ocr_file.readlines()
    
    # Keep a record of corrected words
    ocr_to_word = {}
    outputs = []

    for line in tqdm(ocr_document):
        filtered_line = filter(line)
        token_list = tokenize(filtered_line)
        
        output = []
        for token in token_list:
            if token not in vocab_set: # Time complexity: O(1)
                best_word = find_best_match(token, vocab_list)
            else:
                best_word = token

            ocr_to_word[token] = best_word
            output.append(best_word)
        outputs.append(output)

    ocr_file = open('raven_out.txt', 'w')
    for line in outputs:
        ocr_file.writelines(' '.join(line) + '\n')
    ocr_file.close()
    