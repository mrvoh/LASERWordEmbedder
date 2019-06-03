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
import json
import time

def eval_model_dataset(config, embedder, data, pad_len, model_path, use_laser):


    #build model
    model = NERModel(config, embedder, pad_len)

    learn = NERLearner(config, model, pad_len, pad_len)
    learn.load(model_path)
    learn.model.set_bpe_pad_len(pad_len)
    n_batches, dataloader = learn.batch_iter(data, config.batch_size, use_laser= use_laser)
    if use_laser:
        return learn.test_laser(n_batches, dataloader)
    else:
        return learn.test_base(n_batches, dataloader)


def main(config=None):

    results = {}
    if config is None:
        # create instance of config
        config = Config()
    eng_path = os.path.join('parsed_data_lowercased', 'eng_test_bio_bpe{}.txt'.format('1' if config.pos_target else ''))
    ger_path = os.path.join('parsed_data_lowercased', 'ger_test_bio_bpe{}.txt'.format('1' if config.pos_target else ''))
    ned_path = os.path.join('parsed_data_lowercased', 'ned_test_bio_bpe{}.txt'.format('1' if config.pos_target else ''))
    spa_path = os.path.join('parsed_data_lowercased', 'esp_pos_test_bio_bpe.txt') if config.pos_target else os.path.join('parsed_data_lowercased', 'esp_test_bio_bpe.txt')
    data_filepaths = [
        eng_path,
        ned_path,
        spa_path,
        ger_path,
    ]
    for data_filepath in data_filepaths:

        # get dataset
        encoding = 'latin-1' if (data_filepath == ger_path) else 'utf-8'
        print(encoding)
        data_laser, pad_len = parse_dataset_laser(data_filepath, config.label_to_idx, config.word_to_idx,  pos_target = config.pos_target, encoding = encoding)
        data, pad_len = parse_dataset(data_filepath, config.label_to_idx, config.word_to_idx,  pos_target = config.pos_target, encoding = encoding)

        #####################################################################
        # SETUP
        #####################################################################
        subfolder = 'POS' if config.pos_target else 'NER'
        langfolder = config.langfolder
        base_path = os.path.join('saves_lc',langfolder, subfolder, 'LASEREmbedderBase.pt')
        base_gru_path = os.path.join('saves_lc',langfolder,subfolder, 'LASEREmbedderBaseGRU.pt')
        i_path = os.path.join('saves_lc',langfolder,subfolder, 'LASEREmbedderI.pt')
        iii_path = os.path.join('saves_lc',langfolder,subfolder, 'LASEREmbedderIII.pt')
        elmo_path = os.path.join('saves',langfolder,subfolder, 'LASEREmbedderIIIELMo.pt')

        paths = [
            base_path,
            base_gru_path,
            i_path,
            iii_path,
            # elmo_path
        ]


        embedder_base = LASEREmbedderBase(config.model_path, pad_len)
        embedder_base_gru = LASEREmbedderBaseGRU(config.model_path, pad_len)
        embedderI = LASEREmbedderI(config.model_path)
        embedderIII = LASEREmbedderIII(config.model_path)
        # elmo_embedderIII = LASEREmbedderIIIELMo(config.model_path)

        embedders = [
            embedder_base,
            embedder_base_gru,
            embedderI,
            embedderIII,
            # elmo_embedderIII
        ]

        use_laser = [
            False,
            False,
            True,
            True,
            # True
        ]
        lang_results = {}

        for embedder, path, d in zip(embedders, paths, use_laser):
            print(path)
            dset = data_laser if d else data
            f1 = eval_model_dataset(config, embedder, dset, pad_len, path, d)
            lang_results[embedder.__class__.__name__] = f1

        results[data_filepath] = lang_results
    print(results)
    return results, config

if __name__ == "__main__":
    res, config = main()
    out_path = os.path.join(config.results_folder, config.langfolder, config.subfolder, 'results_laser.json')
    with open(out_path, 'w') as f:
        json.dump(res, f)
        time.sleep(60)
