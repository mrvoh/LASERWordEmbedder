import torch
from utils import load_data, dataset2sentences, dataset2vocab, initialise_bpe, bpe_apply, map_encoded_sentences_to_dataset

TRAIN_FILE_PATH = "./data/train_bio.txt"
TEST_FILE_PATH = "./data/test_bio.txt"
VALID_FILE_PATH = "./data/valid_bio.txt"

initialise_bpe()
data = load_data(VALID_FILE_PATH)

sentences = dataset2sentences(data)
encoded_sentences = bpe_apply(sentences)
m = mapped_encoded_sentences = map_encoded_sentences_to_dataset(data, encoded_sentences)


with open("data/valid_bio_bpe.txt", "w") as f:
    for s in m:
        for word in s:
            f.write(str(" ".join(word) + "\n"))

        f.write("\n")

