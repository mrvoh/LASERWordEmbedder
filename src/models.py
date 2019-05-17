import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.encoders.text import stack_and_pad_tensors
import sys
import os
import math
from numpy import count_nonzero
LASER = os.environ['LASER']
sys.path.append(os.path.join(LASER, 'source'))
sys.path.append(os.path.join(LASER, 'source', 'lib'))

# from the LASER library
from embed import Encoder, SentenceEncoder

RNNS = ['LSTM', 'GRU']


class LASERHiddenExtractor(Encoder):
    """
        Class to extract hidden states per time step from pretrained LASER LSTM encoder.
        Parameters as in LASER/source/embed.py -> Encoder except for take_diff.
        take_diff takes temporal difference of hidden states if set to True.
    """
    def __init__(
            self, num_embeddings, padding_idx, embed_dim=320, hidden_size=512, num_layers=1, bidirectional=False,
            left_pad=True, padding_value=0., store_hidden=False
    ):
        super().__init__(num_embeddings, padding_idx, embed_dim, hidden_size, num_layers, bidirectional,
                         left_pad, padding_value)  # initializes original encoder
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # self.embed_tokens.requires_grad = False
        # self.lstm.requires_grad = False

        # static variables
        self.num_layers = 5
        self.num_directions = 2
        self.store_hidden = store_hidden

    def forward(self, src_tokens):
        # Adjusted version of original LASER forward pass to store cell states

        # B, seq_len = src_tokens.size()
        seq_len, B = src_tokens.size()

        # embed tokens
        token_embeddings = self.embed_tokens(src_tokens)

        if not self.store_hidden:
            output, _ = self.lstm(token_embeddings)
            return output
        else:
            hidden_states = []

            prev_h = torch.zeros(10, B, self.hidden_size).to(self.device)  # 10 = 2*5 = num_layers * directions
            prev_c = torch.zeros(10, B, self.hidden_size).to(self.device)
            for i in range(seq_len):
                s, (prev_h, prev_c) = self.lstm(token_embeddings[i, :, :].unsqueeze(0), (prev_h, prev_c))
                hidden_states.append(prev_c.view(5, 2, B, self.hidden_size)) #TODO: change back4

            hidden_states = torch.stack(hidden_states)
            return hidden_states
###########################################################################
# Classes to extract/learn embeddings from LASER encoder
###########################################################################
class LASEREmbedderBase(nn.Module):

    def __init__(self, encoder_path, bpe_pad_len = 43, encoder = LASERHiddenExtractor):
        super().__init__()
        self.ENCODER_SIZE = self.embedding_dim = 320  # LASER encoder encodes to 512 dims
        self.NUM_LAYERS = 5
        self.NUM_DIRECTIONS = 2
        self.bpe_pad_len = bpe_pad_len
        state_dict = torch.load(encoder_path)
        encoder = encoder(take_diff=False, **state_dict['params'])
        encoder.load_state_dict(state_dict['model'])

        self.bpe_emb = encoder.embed_tokens
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']

    def forward(self, tokens):

        # Split token sequences into words
        tok = torch.split(tokens, split_size_or_sections=self.bpe_pad_len, dim=0)
        # Reconstruct tensor as [seq_len, B, bpe_pad_len]
        tok = torch.stack(tok).permute(0,2,1)
        bpe_embeddings = self.bpe_emb(tok)
        # average to get embeddings to in order to aggregate over BPE sequence to word
        embeddings = torch.mean(bpe_embeddings, dim=2)

        return embeddings


class LASEREmbedderBaseGRU(nn.Module):

    def __init__(self, encoder_path, bpe_pad_len = 43, encoder = LASERHiddenExtractor):
        super().__init__()
        self.ENCODER_SIZE = self.embedding_dim = 320  # LASER encoder encodes to 512 dims
        self.NUM_LAYERS = 5
        self.NUM_DIRECTIONS = 2
        self.bpe_pad_len = bpe_pad_len
        state_dict = torch.load(encoder_path)
        encoder = encoder(take_diff=False, **state_dict['params'])
        encoder.load_state_dict(state_dict['model'])

        self.bpe_emb = encoder.embed_tokens
        for param in self.bpe_emb.parameters():
            param.requires_grad = False
        gru = RNNEncoder(self.ENCODER_SIZE, int(self.embedding_dim/2))
        att = Attention(self.embedding_dim)
        self.token_embedder = TokenEncoder(gru, att, self.embedding_dim)
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']

    def forward(self, tokens):

        seq_len, B = tokens.size()
        token_seq_len = int(seq_len /self.bpe_pad_len)
        # get BPE embeddings
        # Split token sequences into words
        tok = torch.split(tokens, split_size_or_sections=self.bpe_pad_len, dim=0)
        # Reconstruct tensor as [bpe_pad_len, seq_len*B]
        tok = torch.stack(tok).permute(1,2,0).contiguous().view(self.bpe_pad_len, -1)
        bpe_embeddings = self.bpe_emb(tok)

        # apply GRU with self-attention
        embeddings = self.token_embedder(bpe_embeddings)

        # Resize to sequence length level
        embeddings = torch.split(embeddings, split_size_or_sections=token_seq_len, dim=0)
        embeddings = torch.stack(embeddings).permute(1, 0, 2)


        return embeddings

class LASEREmbedderI(nn.Module):

    def __init__(self, encoder_path, bpe_pad_len=43, embedding_dim=320, encoder = LASERHiddenExtractor):
        super().__init__()
        self.ENCODER_SIZE = 512  # LASER encoder encodes to 512 dims
        self.NUM_LAYERS = 5
        self.NUM_DIRECTIONS = 2

        self.bpe_pad_len = bpe_pad_len
        self.embedding_dim = embedding_dim
        gru = RNNEncoder(self.ENCODER_SIZE, int(self.embedding_dim / 2))
        # att = Attention(self.embedding_dim)
        self.token_embedder = TokenEncoder(gru, None, self.embedding_dim)

        state_dict = torch.load(encoder_path)
        self.encoder = encoder(store_hidden=False,**state_dict['params'])
        self.encoder.load_state_dict(state_dict['model'])
        #TODO: change back
        for param in self.encoder.embed_tokens.parameters():
            param.requires_grad = False

        self.scaling_param = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(self.ENCODER_SIZE)

        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']

    def forward(self, input):
        tokens, token_lens = input
        seq_len, B = tokens.size()

       # get metrics of input for splitting
        token_lens = list(token_lens.permute(1,0).cpu().numpy())
        token_lens = [list(l) for l in token_lens]
        sequence_lengths_token = [count_nonzero(token_len) for token_len in token_lens]
        for token_len in token_lens:
            token_len.append(seq_len-sum(token_len))


        token_lens = [l for token_len in token_lens for l in token_len]
        # get final hidden states from LASER encoder
        # hidden_states = self.encoder(tokens)[:, -1, :, :, :] #tok
        encoding = self.encoder(tokens).view(seq_len, B, 2, self.ENCODER_SIZE)
        hidden_states, _ = encoding.max(dim=2)
            # max pool the forward and backward hidden states
        # hidden_states, _ = hidden_states.max(dim=1)
        # split over all words
        hidden_states = self.bn(hidden_states.permute(0,2,1)).permute(0,2,1)
        hidden_states = torch.split(hidden_states.permute(1,0,2).contiguous().view(-1,self.ENCODER_SIZE), split_size_or_sections=token_lens, dim=0)
        s = sequence_lengths_token
        token_pad = max(s)
        # chunk the hidden states per word (bpe fragments per word)
        h0 = [hidden_states[i+i * token_pad:i+i * token_pad + s[i]] for i in range(len(s))]
        h0 = [l for sublist in h0 for l in sublist]
        # pad for token encoding
        h1, _ = stack_and_pad_tensors(h0)
        h1 = self.scaling_param * h1 #nspired by ELMO

        # encode the embeddings
        embeddings = self.token_embedder(h1.permute(1,0,2))
        embeddings = embeddings.split(split_size=s, dim=0)
        embeddings, _ = stack_and_pad_tensors(embeddings)
        embeddings = embeddings.permute(1,0,2)

        return embeddings





class LASEREmbedderIII(nn.Module):
    #TODO: use ELMO LSTM
    def __init__(self, encoder_path, bpe_pad_len = 43, embedding_dim = 320, encoder = LASERHiddenExtractor):
        super().__init__()
        self.ENCODER_SIZE = 512  # LASER encoder encodes to 512 dims
        self.NUM_LAYERS = 5
        self.NUM_DIRECTIONS = 2

        self.bpe_pad_len = bpe_pad_len
        self.embedding_dim = embedding_dim
        gru = RNNEncoder(self.ENCODER_SIZE, int(self.embedding_dim / 2))
        self.token_embedder = TokenEncoder(gru)

        state_dict = torch.load(encoder_path)
        self.encoder = encoder(**state_dict['params'], store_hidden=True)
        self.encoder.load_state_dict(state_dict['model'])
        # freeze params of LASER encoder:
        for param in self.encoder.embed_tokens.params():
            param.requires_grad = False

        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']
        self.embedding_dim = embedding_dim

        self.bn = nn.BatchNorm1d(self.ENCODER_SIZE)
        self.scaling_param = nn.Parameter(torch.ones(1))
        self.layer_weights = nn.Parameter(torch.ones(self.NUM_LAYERS))

        self.hidden_decoder = nn.Linear(self.NUM_LAYERS * self.ENCODER_SIZE, embedding_dim)

    def forward(self, inp):
        (tokens, token_lens) = inp
        seq_len, B = tokens.size()

        # get metrics of input for splitting
        token_lens = list(token_lens.permute(1, 0).cpu().numpy())
        token_lens = [list(l) for l in token_lens]
        sequence_lengths_token = [count_nonzero(token_len) for token_len in token_lens]
        for token_len in token_lens:
            token_len.append(seq_len - sum(token_len))

        token_lens = [l for token_len in token_lens for l in token_len]
        # token_seq_len = int(seq_len / self.bpe_pad_len)
        # assume encoder to return token embeddings
        hidden_states = self.encoder(tokens)
        max_pooled, _ = hidden_states.max(dim=2)

        # Take softmax-normalized weighted average of embedding
        pooled_per_layer = max_pooled.split(dim=1, split_size=1)
        layer_weights = nn.Softmax()(self.layer_weights)
        weighted = torch.cat([pooled_per_layer[i]*layer_weights[i] for i in range(len(pooled_per_layer))],dim=1)
        embeddings = torch.sum(weighted, dim=1)

        # split over all words
        embeddings = self.bn(embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        embeddings = torch.split(embeddings.permute(1, 0, 2).contiguous().view(-1, self.ENCODER_SIZE),
                                    split_size_or_sections=token_lens, dim=0)
        s = sequence_lengths_token
        token_pad = max(s)
        # chunk the hidden states per word (bpe fragments per word)
        h0 = [embeddings[i + i * token_pad:i + i * token_pad + s[i]] for i in range(len(s))]
        h0 = [l for sublist in h0 for l in sublist]
        # pad for token encoding
        h1, _ = stack_and_pad_tensors(h0)
        h1 = self.scaling_param * h1  # inspired by ELMO

        # encode the embeddings
        embeddings = self.token_embedder(h1.permute(1, 0, 2))
        embeddings = embeddings.split(split_size=s, dim=0)
        embeddings, _ = stack_and_pad_tensors(embeddings)
        embeddings = embeddings.permute(1, 0, 2)
        return embeddings

        # Get embeddings
        # embeddings = []
        # for b in range(B):
        #     # apply fc per sequence
        #     m = max_pooled[:, :, b, :].contiguous().view(seq_len, -1)
        #
        #     et = self.hidden_decoder(m)
        #     embeddings.append(et)

        # # stack embeddings back together
        # bpe_embeddings = torch.stack(embeddings).permute(1,0,2)
        # # reshape to in order to aggregate over BPE sequence to word
        # bpe_embeddings = bpe_embeddings.contiguous().view(self.bpe_pad_len, token_seq_len * B, self.embedding_dim)
        #
        # # max pool the forward and backward hidden states
        # embeddings = self.token_embedder(bpe_embeddings)
        # # resize to token-level embeddings
        # embeddings = embeddings.view(token_seq_len, B, self.embedding_dim)
        #
        # return embeddings


class LASEREmbedderIV(nn.Module):

    def __init__(self, encoder_path,  embedding_dim, encoder = LASERHiddenExtractor):
        super().__init__()
        self.ENCODER_SIZE = 512  # LASER encoder encodes to 512 dims
        self.NUM_LAYERS = 5
        self.NUM_DIRECTIONS = 2
        state_dict = torch.load(encoder_path)
        self.encoder = encoder(take_diff=True, **state_dict['params'])
        self.encoder.load_state_dict(state_dict['model'])
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']
        self.embedding_dim = embedding_dim

        self.hidden_decoder = nn.Linear(self.NUM_LAYERS * self.ENCODER_SIZE, embedding_dim)

    def forward(self, tokens):
        B = tokens.size(0)
        seq_len = tokens.size(1)
        # assume encoder to return token embeddings
        hidden_states = self.encoder(tokens)
        max_pooled, _ = hidden_states.max(dim=2)
        # Get embeddings
        embeddings = []
        for b in range(B):
            # apply fc per sequence
            embeddings.append(self.hidden_decoder(max_pooled[:, :, b, :].contiguous().view(seq_len, -1)))

        embeddings = torch.stack(embeddings)

        return embeddings


class RNNEncoder(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
               bidirectional=True, rnn_type='GRU'):
    super(RNNEncoder, self).__init__()
    self.bidirectional = bidirectional
    assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
    rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
    self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers,
                        dropout=dropout, bidirectional=bidirectional)

  def forward(self, input, hidden=None):
    return self.rnn(input, hidden)


class Attention(nn.Module):
  def __init__(self, query_dim):
    super(Attention, self).__init__()
    self.scale = 1. / math.sqrt(query_dim)

  def forward(self, query, keys, values):
    # Query = [BxQ]
    # Keys = [TxBxK]
    # Values = [TxBxV]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)

    query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
    keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
    energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
    energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

    values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
    linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
    return energy, linear_combination

class TokenEncoder(nn.Module):
  def __init__(self, encoder):
    super(TokenEncoder, self).__init__()
    self.encoder = encoder
    # self.attention = attention

    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))


  def forward(self, input):
    outputs, hidden = self.encoder(input)

    max_pool, _ = outputs.max(dim=0)
    return max_pool
    # if isinstance(hidden, tuple): # LSTM
    #   hidden = hidden[1] # take the cell state
    #
    # if self.encoder.bidirectional: # need to concat the last 2 hidden layers
    #   hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
    # else:
    #   hidden = hidden[-1]
    #
    # energy, linear_combination = self.attention(hidden, outputs, outputs)
    # return linear_combination


