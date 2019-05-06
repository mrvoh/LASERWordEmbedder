#from fastai.text import *
from .core import *

class NERModel(nn.Module):

    def __init__(self, config, embedder):
        super().__init__()
        self.config = config

        self.embedder = embedder # LASERWordEmbedder

        self.dropout = nn.Dropout(p=config.dropout)
        self.word_lstm = nn.LSTM(embedder.embedding_dim,
                                 config.hidden_size_lstm, bidirectional=True)

        self.linear = LinearClassifier(self.config, layers=[self.config.hidden_size_lstm*2, self.config.ntags], drops=[0.5])


    def forward(self, input):
        # Word_dim = (batch_size x sent_length)

        word_emb = self.embedder(input)

        output, _ = self.word_lstm(word_emb) #shape = S*B*hidden_size_lstm
        output = self.dropout(output)

        output = self.linear(output)
        return output #shape = S*B*ntags

class LinearBlock(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))


class LinearClassifier(nn.Module):
    def __init__(self, config, layers, drops):
        self.config = config
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, input):
        output = input
        sl,bs,_ = output.size()
        x = output.view(-1, 2*self.config.hidden_size_lstm)

        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x.view(sl, bs, self.config.ntags)
