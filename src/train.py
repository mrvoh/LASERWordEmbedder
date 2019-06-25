from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
from utils import parse_dataset, parse_dataset_laser
import time
from torch.cuda import empty_cache

#from model.ent_model import EntModel
#from model.ent_learner import EntLearner
from models import *
from subprocess import run

def main(config = None, embedders_to_train=None):
    # create instance of config
    if config is None:
        config = Config()
    if embedders_to_train is None:
        embedders_to_train = [
            # 'LASEREmbedderBase',
            #             # 'LASEREmbedderBaseGRU',
            'LASEREmbedderI',
            # 'LASEREmbedderIII',
         # 'LASEREmbedderIIIELMo',
        ]


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
    embedderIIIElmo = LASEREmbedderIIIELMo

    embedders = {
        'LASEREmbedderBase':embedder_base,
        'LASEREmbedderBaseGRU':embedder_base_gru,
        'LASEREmbedderI':embedderI,
        'LASEREmbedderIII':embedderIII,
        'LASEREmbedderIIIELMo':embedderIIIElmo
    }
    # model_name = {
    #     embedder_base:'LASEREmbedderBase',
    #     embedder_base_gru:'LASEREmbedderBaseGRU',
    #     embedderI:'LASEREmbedderI',
    #     embedderIII:'LASEREmbedderIII',
    #     embedderIIIElmo:'LASEREmbedderIIIELMo',
    # }

    use_laser = {
        'LASEREmbedderBase': False,
        'LASEREmbedderBaseGRU': False,
        'LASEREmbedderI': True,
        'LASEREmbedderIII': True,
        'LASEREmbedderIIIELMo': True
    }

    for embedder in embedders_to_train:

        # set output filename
        laser = use_laser[embedder]
        config.set_model_name(embedder)
        config.use_laser = laser
        # config.set_params(laser)
        train = train_laser if laser else train_base
        dev = dev_laser if laser else dev_base
        model = embedders[embedder](config.model_path, bpe_pad_len=tr_pad_len, static_lstm = static_lstm,
                         drop_before = config.drop_before_laser, drop_after = config.drop_after_laser, drop_within=config.drop_in_laser)

        # try:
        fit(config, model, tr_pad_len, dev_pad_len, train, dev)
        del model
        empty_cache()
        time.sleep(60) # free up CUDA memory
        # except:
        #     time.sleep(60)
        #     with open('log.txt', 'a') as f:
        #         f.write(str(embedder)+config.filename_train)


def fit(config, embedder, tr_pad_len, dev_pad_len, train, dev):

    # Initiate model
    model = NERModel(config, embedder,
                     tr_pad_len, dropout = config.transformer_drop,
                     num_heads=config.num_heads, num_layers = config.num_layers,
                     filter_size = config.filter_size)
    # train
    learn = NERLearner(config, model, tr_pad_len, dev_pad_len)
    learn.fit(train, dev)


if __name__ == "__main__":
    main()
