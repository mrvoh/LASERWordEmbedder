from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
from utils import parse_dataset, parse_dataset_laser

#from model.ent_model import EntModel
#from model.ent_learner import EntLearner
from models import *
from subprocess import run

def main():
    # create instance of config
    config = Config()



    # if config.use_laser:
    train_laser, tr_pad_len = parse_dataset_laser(config.filename_train, config.label_to_idx,  config.word_to_idx)
    dev_laser, dev_pad_len = parse_dataset_laser(config.filename_dev, config.label_to_idx, config.word_to_idx)
    # else:
    train_base, tr_pad_len = parse_dataset(config.filename_train, config.label_to_idx, config.word_to_idx)
    dev_base, dev_pad_len = parse_dataset(config.filename_dev, config.label_to_idx, config.word_to_idx) #TODO: try without pad len of train
    # # build model
    # model = NERModel(config, LASEREmbedderIII(config.model_path, bpe_pad_len=tr_pad_len), tr_pad_len) #TODO: check longest pad len test, train, dev
    # learn = NERLearner(config, model, tr_pad_len, dev_pad_len)
    # learn.fit(train, dev)
    #
    embedder_base = LASEREmbedderBase(config.model_path, tr_pad_len)
    embedder_base_gru = LASEREmbedderBaseGRU(config.model_path, tr_pad_len)
    embedderI = LASEREmbedderI(config.model_path)
    embedderIII = LASEREmbedderIII(config.model_path)

    embedders = [
        embedder_base,
        embedder_base_gru,
        embedderI,
        embedderIII,
    ]

    use_laser = [
        False,
        False,
        True,
        True
    ]

    for embedder, laser in zip(embedders, use_laser):
        train = train_laser if laser else train_base
        dev = dev_laser if laser else dev_base

        fit(config, embedder, tr_pad_len, dev_pad_len, train, dev, laser)


def fit(config, embedder, tr_pad_len, dev_pad_len, train, dev, laser):
    #set output filename
    config.set_model_name(embedder.__class__.__name__)
    config.use_laser = laser
    # Initiate model
    model = NERModel(config, embedder,
                     tr_pad_len)
    # train
    learn = NERLearner(config, model, tr_pad_len, dev_pad_len)
    learn.fit(train, dev)


if __name__ == "__main__":
    main()
