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
    vocab = {}

    for sentence in dataset:
        for word in sentence:
            w = word[0]

            if w not in vocab:
                vocab[w] = 0

    return list(vocab.keys())


def vocab2str(vocab):
    return ' '.join(vocab)
