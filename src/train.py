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
    run('export LASER="/home/developer/Desktop/LASERWordEmbedder/LASER/"', shell=True)
    # build model
    model = NERModel(config, LASEREmbedderI(config.model_path))

    # create datasets
    # dev = CoNLLDataset(config.filename_dev, config.processing_word,
    #                    config.processing_tag, config.max_iter, config.use_crf)
    # train = CoNLLDataset(config.filename_train, config.processing_word,
    #                      config.processing_tag, config.max_iter, config.use_crf)

    train = parse_dataset(config.filename_train, config.label_to_idx, config.word_to_idx)
    dev = parse_dataset(config.filename_train, config.label_to_idx, config.word_to_idx) #TODO: change back to dev set

    learn = NERLearner(config, model)
    learn.fit(train, dev)


if __name__ == "__main__":
    main()
