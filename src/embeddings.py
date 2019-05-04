from utils import load_data, dataset2sentences, dataset2vocab, pbe_generate_tokens, pbe_initialise_encoder, pbe_sentence2ids, pbe_sentence2tokens, pbe_ids2sentence

TRAIN_FILE_PATH = "./data/train.txt"
TEST_FILE_PATH = "./data/test.txt"
VALID_FILE_PATH = "./data/valid.txt"

data = load_data(TRAIN_FILE_PATH)
vocab = dataset2vocab(data)
sentences = dataset2sentences(data)

pbe_initialise_encoder()
pbe_generate_tokens(vocab)



s1 = sentences[1]
print(s1)
print("--")
s2 = pbe_sentence2tokens(s1)
print(s2)
print("--")
s3 = pbe_sentence2ids(s1)
print(s3)
print("--")
s4 = pbe_ids2sentence(s3)
print(s4)
print("--")
