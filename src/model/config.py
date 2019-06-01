import os
import torch

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word


class Config():
    def __init__(self):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

    def set_model_name(self, name):
        self.model_name = name
        subfolder = 'POS' if self.pos_target else 'NER'
        self.ner_model_path = os.path.join('saves_lc', self.langfolder, subfolder, self.model_name)

    def set_pos_target(self, task):
        self.pos_target = True if task == 'POS' else False
        if self.pos_target:
            self.label_to_idx = {
                'ADJ': 0,
                'ADP': 1,
                'ADV': 2,
                'AUX': 3,  # no difference between aux and verb
                'CONJ': 4,
                'DET': 5,
                'INTJ': 6,
                'NOUN': 7,
                'NUM': 8,
                'PART': 9,
                'PRON': 10,
                'PROPN': 11,
                'PUNCT': 12,
                'SCONJ': 13,
                'VERB': 3,  # no difference between aux and verb
                'X': 15,
                'SYM': 15,
            }
        else:
            self.label_to_idx = {'O': 0, 'I-PER': 1, 'I-ORG': 2, 'I-LOC': 3, 'I-MISC': 4,
                            'B-PER': 5, 'B-ORG': 6, 'B-LOC': 7, 'B-MISC': 8}

        self.ntags = len(self.label_to_idx)


    # general config
    dir_output = "results_lc/test/"
    dir_model = dir_output
    path_log = dir_output + "log.txt"
    pos_target = False # flag to indicate whether to perform NER or POS tagging


    # FILES TO TRAIN AND EVALUATE ON
    filename_dev = "parsed_data_lowercased/ger_valid_bio_bpe.txt"
    filename_test = "data/test_bio_bpe.txt"
    filename_train = "parsed_data_lowercased/ger_train_bio_bpe.txt"

    # training
    train_embeddings = False
    nepochs = 25
    dropout = 0.5
    batch_size = 128
    lr_method = "rmsprop"
    lr = 0.002
    weight_decay = 0.01
    lr_decay = 0.5
    epoch_drop = 3  # Step Decay: per # epochs to apply lr_decay
    clip = 5  # if negative, no clipping
    nepoch_no_imprv = 2
    use_transformer = True
    learning_rate_warmup_steps = 1
    # model hyperparameters
    hidden_size_lstm = 300  # lstm on word embeddings

    model_name = 'LASEREmbedderIII.pt'
    model_folder = 'saves_lc'
    subfolder = 'POS' if pos_target else 'NER'
    langfolder = 'ger'
    ner_model_path = os.path.join(model_folder,langfolder ,subfolder, model_name) #'"saves/ner_{}e_glove".format(nepochs)
    results_folder = 'results_lc'

    use_laser = True
    # use_muse = not use_laser -> not in use yet
    if pos_target:
        label_to_idx = {
                 'ADJ':0,
                 'ADP':1,
                 'ADV':2,
                 'AUX':3, # no difference between aux and verb
                 'CONJ':4,
                 'DET':5,
                 'INTJ':6,
                 'NOUN':7,
                 'NUM':8,
                 'PART':9,
                 'PRON':10,
                 'PROPN':11,
                 'PUNCT':12,
                 'SCONJ':13,
                 'VERB':3,  # no difference between aux and verb
                 'X':15,
                'SYM':15,
        }
    else:
        label_to_idx = {'O':0,'I-PER': 1, 'I-ORG': 2, 'I-LOC': 3, 'I-MISC': 4,
                        'B-PER': 5, 'B-ORG': 6, 'B-LOC': 7, 'B-MISC': 8}
    ntags = len(label_to_idx)

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU

    # if use_laser:
    model_path = os.path.join('..','LASER','models', 'bilstm.93langs.2018-12-26.pt')
    word_to_idx = torch.load(model_path)['dictionary']

        #TODO: load MUSE word_to_idx


