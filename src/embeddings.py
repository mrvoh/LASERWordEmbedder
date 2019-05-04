from bpe import Encoder
from utils import load_data, dataset2sentences, dataset2vocab

TRAIN_FILE_PATH = "./data/train.txt"
TEST_FILE_PATH = "./data/test.txt"
VALID_FILE_PATH = "./data/valid.txt"

data = load_data(TRAIN_FILE_PATH)
vocab = dataset2vocab(data)
sentences = dataset2sentences(data)

encoder = Encoder()
encoder.fit(vocab)

example = "Vizzini: He didn't fall? INCONCEIVABLE!"
print(encoder.tokenize(example))
# ['__sow', 'vi', 'z', 'zi', 'ni', '__eow', '__sow', ':', '__eow', 'he', 'didn', "'", 't', 'fall', '__sow', '?', '__eow', '__sow', 'in', 'co', 'n', 'ce', 'iv', 'ab', 'le', '__eow', '__sow', '!', '__eow']
print(next(encoder.transform([example])))
# [26, 108, 79, 104, 72, 24, 26, 117, 24, 9, 11, 8, 12, 10, 26, 90, 24, 26, 154, 56, 37, 149, 80, 169, 84, 24, 26, 156, 24]
print(next(encoder.inverse_transform(encoder.transform([example]))))
# vizzini : he didn ' t fall ? inconceivable !




