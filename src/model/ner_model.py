#from fastai.text import *
from .core import *
from .transformer import Encoder as TransformerEncoder

class NERModel(nn.Module):

    def __init__(self, config, embedder, bpe_pad_len, use_transformer=True, dropout=0.1):
        super().__init__()
        self.config = config
        self.bpe_pad_len = bpe_pad_len
        self.embedder = embedder # LASERWordEmbedder

        self.dropout = nn.Dropout(p=config.dropout)
        self.use_transformer = use_transformer
        if use_transformer:
            self.word_transformer = TransformerEncoder(
                embedding_size= 320,
                hidden_size = self.config.hidden_size_lstm,
                num_layers= 2,
                num_heads = 2,
                total_key_depth = 300, #todo: check
                total_value_depth = 4, #check as well
                filter_size = 200,
                max_length = 150,
                attention_dropout=dropout,
                input_dropout=dropout,
                relu_dropout=dropout,
                layer_dropout=dropout
            )
            self.linear = LinearClassifier(self.config, layers=[self.config.hidden_size_lstm, self.config.ntags],
                                           drops=[config.dropout], use_transformer=use_transformer)
        else:
            self.word_lstm = nn.LSTM(embedder.embedding_dim,
                                 config.hidden_size_lstm, bidirectional=True)
            self.linear = LinearClassifier(self.config, layers=[2*self.config.hidden_size_lstm, self.config.ntags],
                                           drops=[config.dropout], use_transformer=use_transformer)





    def forward(self, input):
        # Word_dim = (batch_size x sent_length)

        word_emb = self.embedder(input)
        if self.use_transformer:
            output = self.word_transformer(word_emb.permute(1,0,2))
            output = output.permute(1,0,2)
        else:
            output, _ = self.word_lstm(word_emb) #shape = S*B*hidden_size_lstm
        output = self.dropout(output)

        output = self.linear(output)
        return output #shape = S*B*ntags

    def set_bpe_pad_len(self, l):
        self.embedder.bpe_pad_len = l

class LinearBlock(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))


class LinearClassifier(nn.Module):
    def __init__(self, config, layers, drops, use_transformer=False):
        self.config = config
        self.use_transformer = use_transformer
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, input):
        output = input
        sl,bs,_ = output.size()
        if self.use_transformer:
            x = output.contiguous().view(-1, self.config.hidden_size_lstm)
        else:
            x = output.view(-1, 2 * self.config.hidden_size_lstm)

        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x.view(sl, bs, self.config.ntags)
