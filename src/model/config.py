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
        self.ner_model_path = os.path.join('saves', self.model_name)


    # general config
    dir_output = "results/test/"
    dir_model = dir_output
    path_log = dir_output + "log.txt"


    # FILES TO TRAIN AND EVALUATE ON
    filename_dev = "data/valid_bio_bpe.txt"
    filename_test = "data/test_bio_bpe.txt"
    filename_train = "data/train_bio_bpe.txt"

    # training
    train_embeddings = False
    nepochs = 20
    dropout = 0.5
    batch_size = 64
    lr_method = "adam"
    lr = 0.001
    weight_decay = 0.01
    lr_decay = 0.5
    epoch_drop = 1  # Step Decay: per # epochs to apply lr_decay
    clip = 5  # if negative, no clipping
    nepoch_no_imprv = 3

    # model hyperparameters
    hidden_size_lstm = 300  # lstm on word embeddings

    model_name = 'LASEREmbedderIII.pt'
    ner_model_path = os.path.join('saves', model_name) #'"saves/ner_{}e_glove".format(nepochs)

    use_laser = True
    # use_muse = not use_laser -> not in use yet
    label_to_idx = {'O':0,'I-PER': 1, 'I-ORG': 2, 'I-LOC': 3, 'I-MISC': 4,
     'B-PER': 5, 'B-ORG': 6, 'B-LOC': 7, 'B-MISC': 8}
    ntags = len(label_to_idx)

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU

    # if use_laser:
    model_path = os.path.join('..','LASER','models', 'bilstm.93langs.2018-12-26.pt')
    word_to_idx = torch.load(model_path)['dictionary']

        #TODO: load MUSE word_to_idx


