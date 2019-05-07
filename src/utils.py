# import fastBPE
from torchnlp.word_to_vector import FastText
from urllib.request import  urlopen
from torchnlp.datasets import Dataset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text import stack_and_pad_tensors
from torch.utils.data import DataLoader
import torch

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

    FCODES_PATH = "/home/developer/Desktop/LASERWordEmbedder/LASER/models/93langs.fcodes"
    FVOCAB_PATH = "/home/developer/Desktop/LASERWordEmbedder/LASER/models/93langs.fvocab"

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

        for e in range(es_len):
            fragment = es[e]

            sentence_mapping.append((fragment, word_info[1], word_info[2], word_info[3]))

            if "@" in fragment:
                continue


            if word_info[0][-len(fragment):] == fragment and e != (es_len-1):
                d_counter += 1
                word_info = d[d_counter]


        mapping.append(sentence_mapping)

    return mapping

def get_conll_vocab():
    TRAIN_FILE_PATH = "./data/train.txt"
    TEST_FILE_PATH = "./data/test.txt"
    VALID_FILE_PATH = "./data/valid.txt"

    dataset_paths = [TRAIN_FILE_PATH, TEST_FILE_PATH, VALID_FILE_PATH]

    vocab = []

    for dp in dataset_paths:
        with open(dp) as f:
            for line in f:
                word = line[:line.find(" ")]

                vocab.append(word.lower())

            f.close()

    vocab = list(set(vocab))

    return vocab

def get_muse_vectors():
    embeddings_url = "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec"
    vectors = {}

    with urlopen(embeddings_url) as f:
        f.readline()

        for line in f:
            w_vec = line.decode("utf-8").split()

            vectors[str(w_vec[0]).lower()] = w_vec[1:]

        f.close()

    return vectors


def get_conll_muse_vectors():
    conll_muse_vectors = {}
    conll_words_not_in_muse_vectors = []

    conll_vocab = get_conll_vocab()
    muse_vectors = get_muse_vectors()

    for word in conll_vocab:
        if word in muse_vectors:
            conll_muse_vectors[word] = muse_vectors[word]

        else:
            conll_words_not_in_muse_vectors.append(word)

    return conll_muse_vectors, conll_words_not_in_muse_vectors








def parse_dataset(path, label_to_idx, word_to_idx):
    sentences = []
    sentence = []

    with open(path) as f:

        sample = {'word_ids': [], 'labels': []}
        for line in f:

            if line in ['\n', '\r\n']:
                sample['word_ids'] = torch.LongTensor(sample['word_ids'])
                sample['labels'] = torch.LongTensor(sample['labels'])
                if len(sample['word_ids']) > 0:
                    sentences.append(sample)
                sample = {'word_ids': [], 'labels': []}
                continue
            else:
                word = line.split()[0]
                label = line.split()[-1]
                sample['word_ids'].append(word_to_idx[word] if word in word_to_idx.keys() else 3)  # 3 -> <unk>
                sample['labels'].append(label_to_idx[label])

    return Dataset(sentences)


def collate_fn_infer(batch):
    """ list of tensors to a batch tensors """
    batch, _ = stack_and_pad_tensors([row['word_ids'] for row in batch])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch = batch.to(device)

    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    return transpose(batch)


def collate_fn_eval(batch):
    """ list of tensors to a batch tensors """
    word_ids_batch, _ = stack_and_pad_tensors([seq['word_ids'] for seq in batch])
    label_batch, _ = stack_and_pad_tensors([seq['labels'] for seq in batch])
    seq_len_batch = torch.LongTensor([len(seq['word_ids']) for seq in batch])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    word_ids_batch = word_ids_batch.to(device)
    seq_len_batch =  seq_len_batch.to(device)
    label_batch = label_batch.to(device)

    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    # return (word_ids_batch, seq_len_batch, label_batch)
    return (transpose(word_ids_batch), transpose(seq_len_batch), transpose(label_batch))


def get_data_loader(data, batch_size, drop_last, collate_fn=collate_fn_eval):
    sampler = BucketBatchSampler(data,
                                 batch_size,
                                 drop_last=drop_last,
                                 sort_key=lambda row: -len(row['word_ids']))

    loader = DataLoader(data,
                        batch_sampler=sampler,
                        collate_fn=collate_fn)

    return loader

def get_padded_accuracy(logits, targets, seq_lengths):

    predictions = logits.argmax(dim=-1)
    B = targets.size(0)


