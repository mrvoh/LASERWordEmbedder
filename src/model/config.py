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

    def set_manual_params(self, dropout_before_laser, dropout_in_laser, transformer_drop, dropout, hidden_size_lstm, weight_decay, learning_rate_warmup_steps, num_heads, filter_size):
        self.drop_before_laser   = dropout_before_laser
        self.drop_in_laser   = dropout_in_laser
        self.transformer_drop   = transformer_drop
        self.dropout    = dropout
        self.hidden_size_lstm   = hidden_size_lstm
        self.weight_decay   = weight_decay
        self.learning_rate_warmup_steps = learning_rate_warmup_steps
        self.filter_size    = filter_size
        self.num_heads = num_heads

    def set_n_epoch_no_imprv(self, n_epoch):
        self.nepoch_no_imprv = n_epoch

    def set_params(self, use_laser):
        if use_laser:
            self.nepoch_no_imprv = 1
            self.weight_decay = 0.01
            self.transformer_drop = 0.4
            self.drop_before_laser = 0.1
            self.drop_after_laser = 0.0
        else:
            self.nepoch_no_imprv = 2
            self.weight_decay = 0.01
            self.transformer_drop = 0.1
            self.drop_before_laser = 0.0
            self.drop_after_laser = 0.1
        # less regularization when doing code-switching
        if 'mixed' in self.filename_train:
            if self.use_laser:
                self.transformer_drop = 0.3
            else:
                self.transformer_drop = 0.15
            self.nepoch_no_imprv = 2
            self.drop_before_laser = 0.1
            self.drop_after_laser = 0.0

        if self.pos_target:
            self.drop_before_laser = 0.1
            self.drop_after_laser = 0.0
            self.nepoch_no_imprv = 2
            self.transformer_drop = 0.5
        elif self.static_lstm:
            self.drop_before_laser = 0
            self.drop_after_laser = 0.3



    # general config
    dir_output = "results_lc/test/"
    dir_model = dir_output
    path_log = dir_output + "log.txt"
    pos_target = False # flag to indicate whether to perform NER or POS tagging


    # FILES TO TRAIN AND EVALUATE ON
    filename_dev = "parsed_data_lowercased/eng_valid_bio_bpe.txt"
    filename_test = "data/test_bio_bpe.txt"
    filename_train = "parsed_data_lowercased/eng_train_bio_bpe.txt"

    # training
    static_lstm = False
    nepochs = 25
    dropout = 0.4102888917548943
    transformer_drop = 0.011957334642877793
    batch_size = 64
    lr_method = "rmsprop"
    lr = 0.0035
    weight_decay = 0.00861998251543089
    lr_decay = 0.5
    epoch_drop = 3  # Step Decay: per # epochs to apply lr_decay
    clip = 5  # if negative, no clipping
    nepoch_no_imprv = 1
    use_transformer = True
    learning_rate_warmup_steps = 3
    # model hyperparameters
    hidden_size_lstm = 300  # lstm on word embeddings
    drop_before_laser = 0.39470291836505855
    drop_after_laser = 0.0
    drop_in_laser = 0.024818696534753992
    num_heads = 2
    filter_size = 25
    num_layers = 1


    model_name = 'LASEREmbedderIII.pt'
    model_folder = 'saves_lc'
    subfolder = 'POS' if pos_target else 'NER'
    langfolder = 'eng'
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


