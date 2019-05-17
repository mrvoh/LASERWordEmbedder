""" Command Line Usage
Args:
    eval: Evaluate F1 Score and Accuracy on test set
    pred: Predict sentence.
    (optional): Sentence to predict on. If none given, predicts on "Peter Johnson lives in Los Angeles"

Example:
    > python test.py eval pred "Obama is from Hawaii"
"""

from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
import sys
from utils import parse_dataset, parse_dataset_laser
from models import *
import os

def eval_model_dataset(config, embedder, data, pad_len, model_path, use_laser):


    #build model
    model = NERModel(config, embedder, pad_len)

    learn = NERLearner(config, model, pad_len, pad_len)
    learn.load(model_path)
    n_batches, dataloader = learn.batch_iter(data, config.batch_size, use_laser= use_laser)
    if use_laser:
        learn.test_laser(n_batches, dataloader)
    else:
        learn.test_base(n_batches, dataloader)


def main():
    data_filepath = os.path.join('parsed_data', 'ned_test_bio_bpe')
    # create instance of config
    config = Config()
    # get dataset

    data_laser, pad_len = parse_dataset_laser(config.filename_train, config.label_to_idx, config.word_to_idx)
    data, pad_len = parse_dataset(config.filename_train, config.label_to_idx, config.word_to_idx)

    #####################################################################
    # SETUP
    #####################################################################

    base_path = os.path.join('saves', 'LASERNERBase.pt')
    base_gru_path = os.path.join('saves', 'LASERNERBaseGRUNoPadding.pt')
    i_path = os.path.join('saves', 'LASEREmbedderIStaticLSTM.pt')
    # iii_path = os.path.join('saves', 'LASEREmbedderIIINonStatic.pt')

    paths = [
        base_path,
        base_gru_path,
        i_path,
        # iii_path
    ]


    embedder_base = LASEREmbedderBase(config.model_path, pad_len)
    embedder_base_gru = LASEREmbedderBaseGRU(config.model_path, pad_len)
    embedderI = LASEREmbedderI(config.model_path)
    # embedderIII = LASEREmbedderIII(config.model_path)

    embedders = [
        embedder_base,
        embedder_base_gru,
        embedderI,
        # embedderIII,
    ]

    use_laser = [
        False,
        False,
        True,
        # True
    ]

    for embedder, path, d in zip(embedders, paths, use_laser):
        dset = data_laser if d else data
        eval_model_dataset(config, embedder, dset, pad_len, path, d)

if __name__ == "__main__":
    main()
