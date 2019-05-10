from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
from utils import parse_dataset

#from model.ent_model import EntModel
#from model.ent_learner import EntLearner
from models import *
from subprocess import run

def main():
    # create instance of config
    config = Config()


    # create datasets
    # dev = CoNLLDataset(config.filename_dev, config.processing_word,
    #                    config.processing_tag, config.max_iter, config.use_crf)
    # train = CoNLLDataset(config.filename_train, config.processing_word,
    #                      config.processing_tag, config.max_iter, config.use_crf)

    train, tr_pad_len = parse_dataset(config.filename_train, config.label_to_idx, config.word_to_idx)
    dev, dev_pad_len = parse_dataset(config.filename_dev, config.label_to_idx, config.word_to_idx)
    # build model
    model = NERModel(config, LASEREmbedderBase(config.model_path, bpe_pad_len=tr_pad_len)) #TODO: check longest pad len test, train, dev
    learn = NERLearner(config, model)
    learn.fit(train, dev)


if __name__ == "__main__":
    main()
