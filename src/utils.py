from bpe import Encoder

pbe_encoder = None

def load_data(path):
    sentences = []
    sentence = []

    with open(path) as f:
        for line in f:
            if line in ['\n', '\r\n']:
                sentences.append(sentence)
                sentence = []
                continue

            else:
                sentence.append(line.split())

        f.close()

    return sentences

def dataset2sentences(dataset):
    sentences = []

    for sentence in dataset:
        st = ""

        s = [s[0] for s in sentence]

        for w in s: 
            st += " " + w if w not in [".", ",", "!", "?"] else w

        st = st[1:]

        sentences.append(st)
    
    return sentences


def dataset2vocab(dataset):
    vocab = []

    for sentence in dataset:
        for word in sentence:
            w = word[0]

            vocab.append(w)

    return vocab


def vocab2str(vocab):
    return ' '.join(vocab)


def pbe_initialise_encoder():
    global pbe_encoder

    pbe_encoder = Encoder(1000)

def pbe_generate_tokens(vocab):
    global pbe_encoder

    pbe_encoder.fit(vocab)


def pbe_sentence2tokens(sentence):
    global pbe_encoder

    encoded_sentence = pbe_encoder.tokenize(sentence)

    return encoded_sentence


def pbe_sentence2ids(sentence):
    global pbe_encoder

    if type(sentence) == str:
        sentence = [sentence]

    encoded_sentence = pbe_encoder.transform(sentence)

    return encoded_sentence

def pbe_ids2sentence(encoded_sentence):
    global pbe_encoder
    
    decoded_sentence = pbe_encoder.inverse_transform(encoded_sentence)

    return decoded_sentence

'''
print(encoder.tokenize(example))
# ['__sow', 'vi', 'z', 'zi', 'ni', '__eow', '__sow', ':', '__eow', 'he', 'didn', "'", 't', 'fall', '__sow', '?', '__eow', '__sow', 'in', 'co', 'n', 'ce', 'iv', 'ab', 'le', '__eow', '__sow', '!', '__eow']
print(next(encoder.transform([example])))
# [26, 108, 79, 104, 72, 24, 26, 117, 24, 9, 11, 8, 12, 10, 26, 90, 24, 26, 154, 56, 37, 149, 80, 169, 84, 24, 26, 156, 24]
print(next(encoder.inverse_transform(encoder.transform([example]))))
# vizzini : he didn ' t fall ? inconceivable !
'''