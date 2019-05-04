import operator
from string import ascii_lowercase, ascii_uppercase
import itertools

class BPE:
    def __init__(self):
        self.bp2token = {}
        self.token2bp = {}
        self.frequency_dict = {}
        self.sorted_frequency_dict = None

        self.__generate_tokens()

    def generate_tokens(self, sentences):
        self.__generate_frequency_dict(sentences)
        self.__generate_tokens()


    # assume first letter of sentence isn't space
    def __generate_frequency_dict(sentences):
        for sentence in sentences:
            cont = False

            for i in range(len(sentence) - 1):
                if cont:
                    cont = False
                    continue
                    
                if sentence[i+1] == "\n":
                    cont = True
                    continue

                bp = sentence[i:i+2]

                if bp not in self.frequency_dict:
                    self.frequency_dict[bp] = 0

                self.frequency_dict[bp] += 1

        self.sorted_frequency_dict = sorted(
            self.frequency_dict.items(), key=operator.itemgetter(1)
        )

    #iterating over all lowercase/uppercase/number/special character combinations to generate tokens
    def __generate_tokens(self):
        candidate_token_vocab = ascii_lowercase + ascii_uppercase + "1234567890" + "_!#?"
        candidate_token_vocab = list(candidate_token_vocab)

        candidate_tokens = [a+b for a in candidate_token_vocab for b in candidate_token_vocab]
        
        for ct in candidate_tokens:
            if ct not in self.frequency_dict:


bpe = BPE()