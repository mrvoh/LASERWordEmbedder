import fastBPE
from torchnlp.word_to_vector import FastText
from urllib.request import  urlopen
from torchnlp.datasets import Dataset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text import stack_and_pad_tensors, pad_tensor
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

            if word_info[0][-len(fragment):] == fragment and e != (es_len-1):
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
    conll_words_not_in_muse_vectors = []

    conll_vocab = get_conll_vocab(case_insensitive)
    muse_vectors = get_muse_vectors(case_insensitive)

    for word in conll_vocab:
        if word in muse_vectors:
            conll_muse_vectors[word] = muse_vectors[word]

        else:
            conll_words_not_in_muse_vectors.append(word)

    return conll_muse_vectors, conll_words_not_in_muse_vectors


def parse_dataset(path, label_to_idx, word_to_idx):
    sentences = []
    UNK = 3
    PAD = 1

    with open(path) as f:

        sample = {'word_ids': [], 'labels': []}
        max_len_token = 0
        for line in f:

            if line in ['\n', '\r\n']:  # end of sequence
                sample['labels'] = torch.LongTensor(sample['labels'])
                if len(sample['word_ids']) > 0:
                    sentences.append(sample)
                sample = {'word_ids': [], 'labels': []}
                continue
            else:
                ls = line.split()
                max_len_token = max(max_len_token, len(ls[4:]))
                word = ls[4:]
                label = ls[3]
                sample['word_ids'].append(
                    torch.LongTensor([word_to_idx[w] if w in word_to_idx.keys() else UNK for w in word])
                )  # 3 -> <unk>
                sample['labels'].append(label_to_idx[label])

    # padd all BPE encodings to max length in dataset
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


def collate_fn_eval(batch):
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


def get_data_loader(data, batch_size, drop_last, collate_fn=collate_fn_eval):
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
    print(dataset[0])
    print(encoded_sentences[0])
    print("--")

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



