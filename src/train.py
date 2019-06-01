from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
from utils import parse_dataset, parse_dataset_laser
import time

#from model.ent_model import EntModel
#from model.ent_learner import EntLearner
from models import *
from subprocess import run

def main(config = None):
    # create instance of config
    if config is None:
        config = Config()

    encoding = 'utf-8'
    static_lstm = False

    # parse datasets
    train_laser, tr_pad_len = parse_dataset_laser(config.filename_train, config.label_to_idx,  config.word_to_idx, pos_target = config.pos_target, encoding=encoding)
    dev_laser, dev_pad_len = parse_dataset_laser(config.filename_dev, config.label_to_idx, config.word_to_idx, pos_target = config.pos_target, encoding=encoding)
    # else:
    train_base, tr_pad_len = parse_dataset(config.filename_train, config.label_to_idx, config.word_to_idx, pos_target = config.pos_target, encoding=encoding)
    dev_base, dev_pad_len = parse_dataset(config.filename_dev, config.label_to_idx, config.word_to_idx, pos_target = config.pos_target, encoding=encoding)
    # # build model
    embedder_base = LASEREmbedderBase #(config.model_path, tr_pad_len)
    embedder_base_gru = LASEREmbedderBaseGRU#(config.model_path, tr_pad_len)
    embedderI = LASEREmbedderI#(config.model_path, static_lstm = False)
    embedderIII = LASEREmbedderIII#(config.model_path, static_lstm = False)
    # embedderIIIElmo = LASEREmbedderIIIELMo(config.model_path)

    embedders = [
        # embedder_base,
        # embedder_base_gru,
        # embedderI,
        embedderIII,
        # embedder9 IIIElmo
    ]

    use_laser = [
        # False,
        # False,
        # True,
        True,
        # True
    ]



    for embedder, laser in zip(embedders, use_laser):
        if config.pos_target:
            drop_before = drop_after = 0
        elif static_lstm:
            drop_before = 0
            drop_after = 0.35
        else:
            drop_before = 0.4 if isinstance(embedder,embedderIII) else 0.1
            drop_after = 0.3
        train = train_laser if laser else train_base
        dev = dev_laser if laser else dev_base
        model = embedder(config.model_path, bpe_pad_len=tr_pad_len, static_lstm = static_lstm,
                         drop_before = drop_before, drop_after = drop_after)


        fit(config, model, tr_pad_len, dev_pad_len, train, dev, laser)
        del model
        time.sleep(60) # free up CUDA memory


def fit(config, embedder, tr_pad_len, dev_pad_len, train, dev, laser):
    #set output filename
    config.set_model_name(embedder.__class__.__name__)
    config.use_laser = laser
    # Initiate model
    model = NERModel(config, embedder,
                     tr_pad_len)
    # print(model)
    # train
    learn = NERLearner(config, model, tr_pad_len, dev_pad_len)
    learn.fit(train, dev)


if __name__ == "__main__":
    main()
