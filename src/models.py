import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math
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
            left_pad=True, padding_value=0., take_diff=False
    ):
        super().__init__(num_embeddings, padding_idx, embed_dim, hidden_size, num_layers, bidirectional,
                         left_pad, padding_value)  # initializes original encoder
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.embed_tokens.requires_grad = False
        self.lstm.requires_grad = False

        # static variables
        self.num_layers = 5
        self.num_directions = 2
        self.take_diff = take_diff

    def forward(self, src_tokens):
        # Adjusted version of original LASER forward pass to store cell states

        # B, seq_len = src_tokens.size()
        seq_len, B = src_tokens.size()

        # embed tokens
        token_embeddings = self.embed_tokens(src_tokens)

        hidden_states = []

        prev_h = torch.zeros(10, B, self.hidden_size).to(self.device)  # 10 = 2*5 = num_layers * directions
        prev_c = torch.zeros(10, B, self.hidden_size).to(self.device)
        for i in range(seq_len):
            s, (prev_h, prev_c) = self.lstm(token_embeddings[i, :, :].unsqueeze(0), (prev_h, prev_c))
            hidden_states.append(prev_h.view(5, 2, B, self.hidden_size))

        hidden_states = torch.stack(hidden_states)

        if self.take_diff:
            h = hidden_states.view(seq_len, self.num_layers, self.num_directions, B, -1)
            # take difference of hidden states per time step in forward and backward direction
            forward_diff = [h[t, l, 0, :] - h[t - 1, l, 0, :] for t in range(1, seq_len) for l in
                            range(self.num_layers)]
            backward_diff = [h[t - 1, l, 1, :] - h[t, l, 1, :] for t in range(seq_len - 1, 0, -1) for l in
                             range(self.num_layers)][::-1]
            # add boundaries to list as no difference is taken there
            forw_start = [h[0, l, 0, :] for l in range(self.num_layers)]
            forward_diff = forw_start + forward_diff  # insert the first element of the forward layer in the front
            backward_diff.extend(
                [h[-1, l, 1, :] for l in range(self.num_layers)])  # append the last element of the backward layer

            # reconstruct hidden states per layer
            hidden_states = [torch.stack([h_f, h_b], dim=2) for (h_f, h_b) in zip(forward_diff, backward_diff)]
            # stack hidden states per time stap back into one tensor
            hidden_states = torch.stack(hidden_states).permute(0,3,1,2).view(seq_len, self.num_layers,self.num_directions, B, self.hidden_size)
        #
        #.permute(1,0,2)
        return hidden_states
###########################################################################
# Classes to extract/learn embeddings from LASER encoder
###########################################################################

class LASEREmbedderI(nn.Module):

    def __init__(self, encoder_path, encoder = LASERHiddenExtractor):
        super().__init__()
        self.ENCODER_SIZE = self.embedding_dim = 512  # LASER encoder encodes to 512 dims
        self.NUM_LAYERS = 5
        self.NUM_DIRECTIONS = 2
        state_dict = torch.load(encoder_path)
        self.encoder = encoder(**state_dict['params'])
        self.encoder.load_state_dict(state_dict['model'])
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']

    def forward(self, tokens):
        # Take hidden states of final layer
        hidden_states = self.encoder(tokens)[:, -1, :, :, :]

        # max pool the forward and backward hidden states
        max_pooled, _ = hidden_states.max(dim=1)

        embeddings = max_pooled

        return embeddings


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
        gru = RNNEncoder(self.ENCODER_SIZE, int(self.embedding_dim/2))
        att = Attention(self.embedding_dim)
        self.token_embedder = TokenEncoder(gru, att, self.embedding_dim, 1)
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']

    def forward(self, tokens):

        seq_len, B = tokens.size()
        token_seq_len = int(seq_len /self.bpe_pad_len)
        # Encode embeddings as emb_size x (B * seq_len)
        bpe_embeddings = self.bpe_emb(tokens) #.permute(2,1,0) #.view(self.ENCODER_SIZE,-1)

        # reshape to in order to aggregate over BPE sequence to word
        bpe_embeddings = bpe_embeddings.view(self.bpe_pad_len, token_seq_len*B, self.ENCODER_SIZE)

        # max pool the forward and backward hidden states
        embeddings = self.token_embedder(bpe_embeddings)

        # resize to token-level embeddings

        embeddings = embeddings.view(token_seq_len, B, self.ENCODER_SIZE)

        return embeddings


class LASEREmbedderIII(nn.Module):
    #TODO: use ELMO LSTM
    def __init__(self, encoder_path, embedding_dim, encoder = LASERHiddenExtractor):
        super().__init__()
        self.ENCODER_SIZE = 512  # LASER encoder encodes to 512 dims
        self.NUM_LAYERS = 5
        self.NUM_DIRECTIONS = 2
        state_dict = torch.load(encoder_path)
        self.encoder = encoder(**state_dict['params'])
        self.encoder.load_state_dict(state_dict['model'])
        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']
        self.embedding_dim = embedding_dim

        self.hidden_decoder = nn.Linear(self.NUM_LAYERS * self.ENCODER_SIZE, embedding_dim)

    def forward(self, tokens):
        B = tokens.size(1)
        seq_len = tokens.size(0)
        # assume encoder to return token embeddings
        hidden_states = self.encoder(tokens)
        max_pooled, _ = hidden_states.max(dim=2)
        # Get embeddings
        embeddings = []
        for b in range(B):
            # apply fc per sequence
            m = max_pooled[:, :, b, :].contiguous().view(seq_len, -1)
            et = self.hidden_decoder(m)
            embeddings.append(et)

        # stack embeddings back together
        embeddings = torch.stack(embeddings).permute(1,0,2) #.view(seq_len, B, -1)

        return torch.tanh(embeddings)


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
  def __init__(self, encoder, attention, hidden_dim, num_classes):
    super(TokenEncoder, self).__init__()
    self.encoder = encoder
    self.attention = attention
    self.decoder = nn.Linear(hidden_dim, num_classes)

    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))


  def forward(self, input):
    outputs, hidden = self.encoder(input)
    if isinstance(hidden, tuple): # LSTM
      hidden = hidden[1] # take the cell state

    if self.encoder.bidirectional: # need to concat the last 2 hidden layers
      hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
    else:
      hidden = hidden.view(-1,320)

    energy, linear_combination = self.attention(hidden, outputs, outputs)
    return linear_combination


