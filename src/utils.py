from urllib.request import urlopen
from torchnlp.datasets import Dataset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text import stack_and_pad_tensors, pad_tensor
from torch.utils.data import DataLoader
import torch
import os
import nltk
import numpy as np
import json

if os.name == 'posix': import fastBPE
LASER = os.environ['LASER']

bpe = None


def load_data(path):
    sentences = []
    sentence = []
    newline = False

    with open(path) as f:
        for line in f:
            if 'DOCSTART' in line:
                continue
            if line in ['\n', '\r\n']:
                if not newline:
                    sentences.append(sentence)
                sentence = []

                newline = True
                continue



            else:
                newline = False
                l= line.split()
                word = l[0].replace('--', '')
                if len(word) == 0: word = '-'
                l[0] = word
                sentence.append(l)

        f.close()

    return sentences


def dataset2sentences(dataset):
    sentences = []

    for sentence in dataset:
        st = ""

        s = [s[0] for s in sentence]

        for w in s:
            st += " " + w #if w not in [".", ",", "!", "?"] else w

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

    FCODES_PATH = LASER + "/models/93langs.fcodes"
    FVOCAB_PATH = LASER + "models/93langs.fvocab"

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

        d_counter = 0
        word_info = d[d_counter]
        sentence_mapping = []

        fragments = []
        for e in range(es_len):
            fragment = es[e]

            fragments.append(fragment)

            if "@" in fragment:
                continue
            else:
                sentence_mapping.append((word_info[0], word_info[1], word_info[2], word_info[3], ' '.join(fragments)))
                fragments = []

            if word_info[0][-len(fragment):] == fragment and e != (es_len - 1):
                d_counter += 1
                word_info = d[d_counter]

        mapping.append(sentence_mapping)

    return mapping


def get_conll_vocab(case_insensitive=False):
    TRAIN_FILE_PATH = "./data/train.txt"
    TEST_FILE_PATH = "./data/test.txt"
    VALID_FILE_PATH = "./data/valid.txt"

    dataset_paths = [TRAIN_FILE_PATH, TEST_FILE_PATH, VALID_FILE_PATH]

    vocab = []

    for dp in dataset_paths:
        with open(dp) as f:
            for line in f:
                word = line[:line.find(" ")]

                if case_insensitive:
                    word = word.lower()

                vocab.append(word)

            f.close()

    vocab = list(set(vocab))

    return vocab


def get_muse_vectors(case_insensitive=False):
    embeddings_url = "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec"
    vectors = {}

    with urlopen(embeddings_url) as f:
        f.readline()

        for line in f:
            w_vec = line.decode("utf-8").split()
            v = w_vec[0]

            if case_insensitive:
                v = str(v).lower()

            vectors[v] = w_vec[1:]

        f.close()

    return vectors


def get_conll_muse_vectors(case_insensitive=True):
    conll_muse_vectors = {}
    unknown_word_vector = load_unknown_muse_vector()


    conll_vocab = get_conll_vocab(case_insensitive)
    muse_vectors = get_muse_vectors(case_insensitive)

    for word in conll_vocab:
        if word in muse_vectors:
            conll_muse_vectors[word] = muse_vectors[word]

        else:
            conll_muse_vectors[word] = unknown_word_vector

    return conll_muse_vectors

def parse_dataset_laser(path, label_to_idx, word_to_idx):
    sentences = []
    UNK = 3
    PAD = 1

    with open(path) as f:

        sample = {'word_ids': [], 'labels': [], 'word_len': []}
        max_len_token = 0
        for line in f.read().splitlines():
            if line in ['\n', '\r\n', '']:  # end of sequence
                if len(sample['labels']) > 0:
                    sample['labels'] = torch.LongTensor(sample['labels'])
                    sample['word_ids'] = torch.LongTensor(sample['word_ids'])
                    sample['word_len'] = torch.LongTensor(sample['word_len'])
                    sentences.append(sample)
                sample = {'word_ids': [], 'labels': [], 'word_len': []}
                continue
            else:
                ls = line.split()
                max_len_token = max(max_len_token, len(ls[4:]))
                word = ls[4:]
                label = ls[3]
                if len(word) > 0:
                    word_ids = [word_to_idx[w.lower()] if w.lower() in word_to_idx.keys() else UNK for w in word]
                    sample['word_ids'].extend(
                        word_ids
                    )  # 3 -> <unk>
                    sample['word_len'].append(len(word_ids))
                    sample['labels'].append(label_to_idx[label])
                    if len(word_ids) > 20:
                        print(line)
    return Dataset(sentences), max_len_token

def parse_dataset(path, label_to_idx, word_to_idx, pad_len=None):
    sentences = []
    UNK = 3
    PAD = 1

    with open(path) as f:

        sample = {'word_ids': [], 'labels': []}
        max_len_token = 0
        for line in f.read().splitlines():
            if line in ['\n', '\r\n', '']:  # end of sequence
                if len(sample['labels']) > 0:
                    sample['labels'] = torch.LongTensor(sample['labels'])
                    sentences.append(sample)
                sample = {'word_ids': [], 'labels': []}
                continue
            else:
                ls = line.split()
                max_len_token = max(max_len_token, len(ls[4:]))
                word = ls[4:]
                label = ls[3]
                if len(word) > 0:
                    word_ids = [word_to_idx[w.lower()] if w.lower() in word_to_idx.keys() else UNK for w in word]
                    sample['word_ids'].append(
                        torch.LongTensor(word_ids)
                    )  # 3 -> <unk>
                    sample['labels'].append(label_to_idx[label])
                    if len(word_ids) > 20:
                        print(line)

    # padd all BPE encodings to max length in dataset
    if pad_len is not None:
        max_len_token = max(pad_len, max_len_token)
    for s in range(len(sentences)):
        sen = sentences[s]
        for i in range(len(sen['word_ids'])):
            sen['word_ids'][i] = pad_tensor(sen['word_ids'][i], length=max_len_token, padding_index=PAD)

        # stack word ids back together
        sen['word_ids'] = torch.stack(sen['word_ids'], dim=0).view(-1)

    return Dataset(sentences), max_len_token


def collate_fn_infer(batch):
    """ list of tensors to a batch tensors """
    batch, _ = stack_and_pad_tensors([row['word_ids'] for row in batch])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch = batch.to(device)

    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    return transpose(batch)

def collate_fn_eval_laser(batch):
    word_ids_batch, _ = stack_and_pad_tensors([seq['word_ids'] for seq in batch])
    label_batch, _ = stack_and_pad_tensors([seq['labels'] for seq in batch])
    seq_len_batch = torch.LongTensor([len(seq['word_ids']) for seq in batch])
    word_len_batch, _ = stack_and_pad_tensors([seq['word_len'] for seq in batch])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    word_ids_batch = word_ids_batch.to(device)
    word_len_batch = word_len_batch.to(device)
    seq_len_batch = seq_len_batch.to(device)
    label_batch = label_batch.to(device)

    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    # return (word_ids_batch, seq_len_batch, label_batch)
    return (transpose(word_ids_batch), transpose(word_len_batch), seq_len_batch, transpose(label_batch))

def collate_fn_eval_base(batch):
    """ list of tensors to a batch tensors """

    word_ids_batch, _ = stack_and_pad_tensors([seq['word_ids'] for seq in batch])
    label_batch, _ = stack_and_pad_tensors([seq['labels'] for seq in batch])
    seq_len_batch = torch.LongTensor([len(seq['word_ids']) for seq in batch])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    word_ids_batch = word_ids_batch.to(device)
    seq_len_batch = seq_len_batch.to(device)
    label_batch = label_batch.to(device)

    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    # return (word_ids_batch, seq_len_batch, label_batch)
    return (transpose(word_ids_batch), seq_len_batch, transpose(label_batch))


def get_data_loader(data, batch_size, drop_last, collate_fn=collate_fn_eval_base):
    sampler = BucketBatchSampler(data,
                                 batch_size,
                                 drop_last=drop_last,
                                 sort_key=lambda row: -len(row['word_ids']))

    loader = DataLoader(data,
                        batch_sampler=sampler,
                        collate_fn=collate_fn)

    return loader


def words2fragments(dataset, encoded_sentences):
    mapping = []

    for sentence_index in range(len(dataset)):
        sentence_mapping = []
        fragment_counter = 0

        for word in dataset[sentence_index]:
            word_mapping = word
            w = word_mapping[0]
            fragments = encoded_sentences[sentence_index].split()
            checked_fragments = []

            for fr in range(len(fragments)):
                for sc in [".", ",", "!", "?"]:
                    if sc in fragments[fr]:
                        checked_fragments.append(fragments[fr][:fragments[fr].find(sc)])
                        checked_fragments.append(fragments[fr][fragments[fr].find(sc):])

                    else:
                        checked_fragments.append(fragments[fr])

                    break

            fragments = checked_fragments

            fragments_for_word = []

            for f in range(fragment_counter, len(fragments)):
                fragments_for_word.append(fragments[f])
                fragment_counter += 1

                if w[-len(fragments[f]):] == fragments[f]:
                    break

            word_mapping += fragments_for_word
            sentence_mapping.append(word_mapping)

        mapping.append(sentence_mapping)

    return mapping


def i2b(i_dataset):
    b_format = []

    for dp in i_dataset:
        previous_named_entity_tag_i = False
        sentence_b_format = []

        for word in dp:
            word_b_format = list(word)

            named_entity_tag_i = True if word[3][:1] in ['I', 'B'] else False

            if named_entity_tag_i:
                if not previous_named_entity_tag_i:
                    word_b_format[3] = 'B' + word_b_format[3][1:]

            previous_named_entity_tag_i = named_entity_tag_i

            sentence_b_format.append(word_b_format)

        b_format.append(sentence_b_format)

    return b_format


def generate_conll2002_datasets():
    nltk.download('conll2002')

    sets = [
        ("esp_test", "https://www.clips.uantwerpen.be/conll2002/ner/data/esp.testa"),
        ("esp_valid", "https://www.clips.uantwerpen.be/conll2002/ner/data/esp.testb"),
        ("esp_train", "https://www.clips.uantwerpen.be/conll2002/ner/data/esp.train"),
        ("ned_test", "https://www.clips.uantwerpen.be/conll2002/ner/data/ned.testa"),
        ("ned_valid", "https://www.clips.uantwerpen.be/conll2002/ner/data/ned.testb"),
        ("ned_train", "https://www.clips.uantwerpen.be/conll2002/ner/data/ned.train")
    ]

    #esp_pos_tags = get_esp_pos_tags()
    counter = 0

    #trying nltk instead
    tags = nltk.corpus.conll2002.tagged_words()


    for set in sets:
        path = set[0]
        url = set[1]
        sentences = []
        sentence = []
        esp = True if path[:3] == 'esp' else False

        with urlopen(url) as f:
            if not esp:
                f.readline()

            for line in f:
                line = line.decode("windows-1252")

                line = line.replace('\n', '')

                if line in ['\n', '\r\n', '']:
                    sentences.append(sentence)
                    sentence = []

                else:
                    info = line.split()

                    if esp and info[0] != tags[counter][0]:
                        print(info[0])
                        print(tags[counter])
                        return

                    if esp:
                        t = tags[counter]

                        if t[0] != info[0]:
                            print("error with tags")
                            return

                        else:
                            sentence.append([info[0], tags[counter][1], 'DUMMY', info[1]])
                            counter += 1

                    else:
                        sentence.append([info[0], info[1], 'DUMMY', info[2]])


        with open("./data/" + path + ".txt", "w") as f:
            for s in sentences:
                for w in s:
                    f.write(' '.join(w) + '\n')

                f.write('\n')
        


            f.close()


def generate_conll2003_german_datasets():
    sets = [
        ("ger_test", "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deuutf.testa"),
        ("ger_valid", "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testb"),
        ("ger_train", "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deuutf.train")
    ]


    for set in sets:
        path = set[0]
        url = set[1]
        sentences = []
        sentence = []

        with urlopen(url) as f:
            f.readline()
            f.readline()

            for line in f:
                line = line.decode("latin-1")

                line = line.replace('\n', '')

                if line in ['\n', '\r\n', '']:
                    sentences.append(sentence)
                    sentence = []

                else:
                    info = line.split()
                    info.pop(1)

                    sentence.append(info)

        sentences = i2b(sentences)

        with open("./data/" + path + ".txt", "w", encoding="latin-1") as f:
            for s in sentences:
                for w in s:
                    f.write(' '.join(w) + '\n')

                f.write('\n')


            f.close()



def get_esp_pos_tags():
    pos_tags = {}

    paths = ("test", "train", "valid")

    for p in paths:
        with open ("./data/esp_" + p + "_words.tag") as f:
            for line in f:
                if line != "\n":
                    line = line.split("\t")
                    word = line[0]
                    tag = line[1].replace("\n", "")

                    if word not in pos_tags:
                        pos_tags[word] = tag


    return pos_tags


'''
def generate_unknown_muse_vector():
    a = np.random.uniform(low=-0.15, high=0.15, size=(300,))
    a = list(a)

    with open("./data/muse_unknown_vector.json", "w") as f:
        json.dump(a, f)
        f.close()
'''


'''a, b = get_conll_muse_vectors()

print(len(a))
print(len(b))'''

def load_unknown_muse_vector():
    with open("./data/muse_unknown_vector.json") as f:
        return json.load(f)



