import torch
import torch.nn as nn
import sys
import os
LASER = os.environ['LASER']
sys.path.append(LASER + '/source/')
sys.path.append(LASER + '/source/lib')
# from the LASER library
from embed import Encoder, SentenceEncoder
from text_processing import Token, BPEfastApply
from models import *


############################################################
# Parse text file
############################################################

# load and preprocess input file
BPE_CODES = LASER+'/models/93langs.fcodes'
LANGUAGE_CODE = 'en'
VERBOSE = True

input_file = 'data/test_sentences.txt'
tokenized_f = 'data/test_tokenized.txt'
bpe_f = 'data/test_bpe.txt'
# tokenize
Token(input_file,
      tokenized_f,
      lang=LANGUAGE_CODE,
      romanize=False, #kept static for simplicity
      lower_case=True, gzip=False,
      verbose=VERBOSE, over_write=False)

# BPE
BPEfastApply(tokenized_f,
             bpe_f,
             BPE_CODES,
             verbose=VERBOSE, over_write=False)

############################################################
# Load + infer model
############################################################

model = LASEREmbedderIV(LASER+'/models/bilstm.93langs.2018-12-26.pt', LASERHiddenExtractor, 300,100, 10)

tokens = torch.LongTensor([[1,2,3],[4,5,6],[6,7,8],[7,8,9]])
embeddings = model(tokens)
print(embeddings.size())