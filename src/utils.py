import fastBPE
from torchnlp.datasets import Dataset
from torchnlp.samplers import BucketBatchSampler
from torchnlp.encoders.text import stack_and_pad_tensors
from torch.utils.data import DataLoader
import torch

bpe = None

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


def parse_dataset(path, label_to_idx, word_to_idx):
    sentences = []
    sentence = []

    with open(path) as f:

        sample = {'word_ids': [], 'labels': []}
        for line in f:

            if line in ['\n', '\r\n']:
                sample['word_ids'] = torch.LongTensor(sample['word_ids'])
                sample['labels'] = torch.LongTensor(sample['labels'])
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

    return (transpose(word_ids_batch), transpose(seq_len_batch), transpose(label_batch))


def get_data_loader(data, batch_size, drop_last, collate_fn=collate_fn_eval):
    sampler = BucketBatchSampler(data,
                                 batch_size,
                                 drop_last=drop_last,
                                 sort_key=lambda row: len(row['word_ids']))

    loader = DataLoader(data,
                        batch_sampler=sampler,
                        collate_fn=collate_fn)

    return loader


