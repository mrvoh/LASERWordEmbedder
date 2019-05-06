import torch
import torch.nn as nn
import sys
import os
LASER = os.environ['LASER']
sys.path.append(LASER + '/source/')
sys.path.append(LASER + '/source/lib')
# from the LASER library
from embed import Encoder, SentenceEncoder
from text_processing import Token, BPEfastApply


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

		B, seq_len = src_tokens.size()
		# embed tokens
		token_embeddings = self.embed_tokens(src_tokens)

		hidden_states = []

		prev_h = torch.zeros(10, B, self.hidden_size).to(self.device)  # 10 = 2*5 = num_layers * directions
		prev_c = torch.zeros(10, B, self.hidden_size).to(self.device)
		for i in range(seq_len):
			s, (prev_h, prev_c) = self.lstm(token_embeddings[:, i, :].unsqueeze(0), (prev_h, prev_c))
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
			hidden_states = torch.stack(hidden_states).view(seq_len, self.num_layers,
															self.num_directions, B, self.hidden_size)

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


class LASEREmbedderII(nn.Module):

	def __init__(self, encoder_path, encoder = LASERHiddenExtractor):
		super().__init__()
		self.ENCODER_SIZE = self.embedding_dim = 512  # LASER encoder encodes to 512 dims
		self.NUM_LAYERS = 5
		self.NUM_DIRECTIONS = 2
		state_dict = torch.load(encoder_path)
		self.encoder = encoder(take_diff=True, **state_dict['params'])
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


class LASEREmbedderIII(nn.Module):

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

		# stack embeddings back together
		embeddings = torch.stack(embeddings)

		return embeddings


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



