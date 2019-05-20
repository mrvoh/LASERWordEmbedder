
from torchnlp.word_to_vector import FastText
from torchnlp.datasets import Dataset
from collections import OrderedDict
import numpy as np
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
from utils import parse_dataset_muse, get_embedding
from models import *


def main(config = None):
    # create instance of config
    if config is None:
        config = Config()
    train, word_to_idx = parse_dataset_muse(config.filename_train, config.label_to_idx, pos_target = config.pos_target)
    dev, word_to_idx = parse_dataset_muse(config.filename_dev, config.label_to_idx, word_to_idx, pos_target = config.pos_target)

    vectors = FastText(aligned = True, cache='.word_vectors_cache')

    embed_table = get_embedding(vectors, word_to_idx)

    embedder_muse = MUSEEmbedder(word_to_idx, embed_table)
    fit(config, embedder_muse, train, dev)





def fit(config, embedder, train, dev):
    #set output filename
    config.set_model_name(embedder.__class__.__name__)
    config.use_laser = False
    pad_len = 0 # no BPE fragments used
    # Initiate model
    model = NERModel(config, embedder,
                     pad_len)
    # train
    learn = NERLearner(config, model, pad_len, pad_len)
    learn.fit(train, dev)


if __name__ == "__main__":
    main()
