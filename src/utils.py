import fastBPE
from torchnlp.word_to_vector import FastText

bpe = None

def load_data(path):
    sentences = []
    sentence = []
    newline = False

    with open(path) as f:
        for line in f:
            if line in ['\n', '\r\n']:
                if not newline:
                    sentences.append(sentence)

                sentence = []

                newline = True
                continue



            else:
                newline = False
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


def initialise_bpe():
    global bpe

    FCODES_PATH = "/home/vm/Documents/LASERWordEmbedder/LASER/models/93langs.fcodes"
    FVOCAB_PATH = "/home/vm/Documents/LASERWordEmbedder/LASER/models/93langs.fvocab"

    bpe = fastBPE.fastBPE(FCODES_PATH, FVOCAB_PATH)

def bpe_apply(sentences):
    global bpe
    
    return bpe.apply(sentences)


def map_encoded_sentences_to_dataset(dataset, encoded_sentences):
    mapping = []
    
    for i in range(len(dataset)):
        d = dataset[i]
        es = encoded_sentences[i].split()
        es_len = len(es)

        try:
            d_counter = 0
            word_info = d[d_counter]
            sentence_mapping = []

            for e in range(es_len):
                fragment = es[e]

                sentence_mapping.append((fragment, word_info[1], word_info[2], word_info[3]))

                if "@" in fragment:
                    continue


                if word_info[0][-len(fragment):] == fragment and e != (es_len-1):
                    d_counter += 1
                    word_info = d[d_counter]

        except Exception as e:
            print(i)
            print(es)
            print(es_len)
            print(d_counter)
            print(word_info)

        mapping.append(sentence_mapping)

    return mapping



