""" Works with pytorch 0.4.0 """

from .core import *
from .data_utils import pad_sequences, minibatches, get_chunks
from .crf import CRF
from .general_utils import Progbar
from torch.optim.lr_scheduler import StepLR
if os.name == "posix": from allennlp.modules.elmo import batch_to_ids # AllenNLP is currently only supported on linux

import sys
from utils import *
from torchnlp.metrics import get_accuracy
# LASER = os.environ['LASER']
# sys.path.append(LASER + '/source/lib')
# from text_processing import Token, BPEfastApply

class NERLearner(object):
    """
    NERLearner class that encapsulates a pytorch nn.Module model and ModelData class
    Contains methods for training a testing the model
    """
    def __init__(self, config, model, tr_pad_len, dev_pad_len):
        super().__init__()
        self.config = config
        self.logger = self.config.logger
        self.model = model
        self.model_path = config.dir_model
        self.tr_pad_len = tr_pad_len
        self.dev_pad_len = dev_pad_len


        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.label_to_idx.items()}

        self.criterion = CRF(self.config.ntags)
        if config.use_transformer:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        else:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        if USE_GPU:
            self.use_cuda = True
            self.logger.info("GPU found.")
            self.model = model.cuda()
            self.criterion = self.criterion.cuda()

        else:
            self.model = model.cpu()
            self.use_cuda = False
            self.logger.info("No GPU found.")

    def lr_decay_noam(self, config):
        return lambda t: (
                10.0 * config.hidden_size_lstm ** -0.5 * min(
            (t + 1) * config.learning_rate_warmup_steps ** -1.5, (t + 1) ** -0.5))

    def get_model_path(self, name):
        return os.path.join(self.model_path,name)

    def get_layer_groups(self, do_fc=False):
        return children(self.model)

    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:
            set_trainable(l, False)
        for l in c[n:]:
            set_trainable(l, True)

    def unfreeze(self):
        self.freeze_to(0)

    def save(self, name=None):
        if not name:
            name = self.config.ner_model_path

        torch.save(self.model.state_dict(), name+'.pt')
        # save_model(self.model, name)
        self.logger.info(f"Saved model at {name}")


    def load(self, fn=None):
        if not fn: fn = self.config.ner_model_path
        # fn = self.get_model_path(fn)
        self.model.load_state_dict(torch.load(fn))

    def load_muse(self, fn=None):
        if not fn: fn = self.config.ner_model_path
        state_dict  = torch.load(fn)
        state_dict = {k: v for k, v in state_dict.items() if k != 'embedder.embed.weight'}

        # s = self.model.state_dict()
        # s = {k: v for k, v in s.items() if k != 'embedder.embed.weight'}

        self.model.load_state_dict(state_dict, strict = False)

    def batch_iter(self, train, batch_size, return_lengths=False, shuffle=False, sorter=False, drop_last =True, use_laser=None):
        """
        Builds a generator from the given dataloader to be fed into the model

        Args:
            train: Dataset
            batch_size: size of each batch
            return_lengths: if True, generator returns a list of sequence lengths for each
                            sample in the batch
                            ie. sequence_lengths = [8,7,4,3]
            shuffle: if True, shuffles the data for each epoch
            sorter: if True, uses a sorter to shuffle the data

        Returns:
            nbatches: (int) number of batches
            data_generator: batch generator yielding
                                dict inputs:{'word_ids' : np.array([[padded word_ids in sent1], ...])
                                             'char_ids': np.array([[[padded char_ids in word1_sent1], ...],
                                                                    [padded char_ids in word1_sent2], ...],
                                                                    ...])}
                                labels: np.array([[padded label_ids in sent1], ...])
                                sequence_lengths: list([len(sent1), len(sent2), ...])

        """
        if use_laser is None: use_laser = self.config.use_laser
        nbatches = (len(train) + batch_size - 1) // batch_size

        if use_laser:
            dataloader = get_data_loader(train, batch_size, drop_last, collate_fn=collate_fn_eval_laser)
        else:
            dataloader = get_data_loader(train, batch_size, drop_last)

        return (nbatches, dataloader)


    def fine_tune(self, train, dev=None):
        """
        Fine tune the NER model by freezing the pre-trained encoder and training the newly
        instantiated layers for 1 epochs
        """
        self.logger.info("Fine Tuning Model")
        self.fit(train, dev, epochs=1, fine_tune=True)


    def fit(self, train, dev=None, epochs=None, fine_tune=False):
        """
        Fits the model to the training dataset and evaluates on the validation set.
        Saves the model to disk
        """
        n_epoch_no_improv = 0
        prev_best = 0
        if not epochs:
            epochs = self.config.nepochs
        batch_size = self.config.batch_size

        nbatches_train, train_generator = self.batch_iter(train, batch_size, drop_last=True)
        if dev:
            nbatches_dev, dev_generator = self.batch_iter(dev, batch_size, drop_last=True)

        if self.config.use_transformer:
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                self.lr_decay_noam(self.config)
            )
        else:
            self.scheduler = StepLR(self.optimizer, step_size=self.config.epoch_drop, gamma=self.config.lr_decay)

        if not fine_tune: self.logger.info("Training Model")

        for epoch in range(epochs):
            self.model.set_bpe_pad_len(self.tr_pad_len)
            # if not self.config.use_transformer:
            self.scheduler.step()
            if self.config.use_laser:
                self.train_laser(epoch, nbatches_train, train_generator, fine_tune=fine_tune)
            else:
                self.train_base(epoch, nbatches_train, train_generator, fine_tune=fine_tune)

            if dev:
                self.model.set_bpe_pad_len(self.dev_pad_len)
                if self.config.use_laser:
                    f1 = self.test_laser(nbatches_dev, dev_generator, fine_tune=fine_tune)
                else:
                    f1 = self.test_base(nbatches_dev, dev_generator, fine_tune=fine_tune)

            # Early stopping
            if f1 > prev_best:
                self.save(self.config.ner_model_path)
                n_epoch_no_improv = 0
                prev_best = f1
            else:
                n_epoch_no_improv += 1
                if n_epoch_no_improv >= self.config.nepoch_no_imprv:
                    print("No improvement in the last {} epochs. Stopping training".format(self.config.nepoch_no_imprv))
                    break

        # if fine_tune:
        #     self.save(self.config.ner_ft_path)
        # else:
        #     self.save(self.config.ner_model_path)

    def train_laser(self, epoch, nbatches_train, train_generator, fine_tune=False):
        self.logger.info('\nEpoch: %d' % epoch)
        self.model.train()

        train_loss = 0
        correct = 0
        total = 0
        total_step = None

        prog = Progbar(target=nbatches_train)

        for batch_idx, (inputs, word_lens, sequence_lengths, targets) in enumerate(train_generator):

            if batch_idx == nbatches_train: break
            if targets.shape[0] == self.config.batch_size:
                # self.logger.info('Skipping batch of size=1')
                continue

            total_step = batch_idx
            self.optimizer.zero_grad()
            outputs = self.model((inputs, word_lens))

            # Create mask
            mask = create_mask(sequence_lengths, targets, cuda=self.use_cuda)

            # Get CRF Loss
            loss = -1 * self.criterion(outputs, targets, mask=mask)
            loss.backward()
            self.optimizer.step()
            # if self.config.use_transformer:
            #     self.scheduler.step()

            # Callbacks
            train_loss += loss.item()
            predictions = self.criterion.decode(outputs, mask=mask)
            masked_targets = mask_targets(targets, sequence_lengths)

            t_ = mask.type(torch.LongTensor).sum().item()
            total += t_
            c_ = sum([1 if p[i] == mt[i] else 0 for p, mt in zip(predictions, masked_targets) for i in range(len(p))])
            correct += c_

            prog.update(batch_idx + 1, values=[("train loss", loss.item())], exact=[("Accuracy", 100 * c_ / t_)])

        self.logger.info("Train Loss: %.3f, Train Accuracy: %.3f%% (%d/%d)" % (
        train_loss / (total_step + 1), 100. * correct / total, correct, total))
    def train_base(self, epoch, nbatches_train, train_generator, fine_tune=False):
        self.logger.info('\nEpoch: %d' % epoch)
        self.model.train()

        train_loss = 0
        correct = 0
        total = 0
        total_step = None

        prog = Progbar(target=nbatches_train)

        for batch_idx, (inputs, sequence_lengths, targets) in enumerate(train_generator):

            if batch_idx == nbatches_train: break
            if targets.shape[0] == self.config.batch_size:
                # self.logger.info('Skipping batch of size=1')
                continue

            total_step = batch_idx
            # targets = T(targets, cuda=self.use_cuda).transpose(0,1).contiguous()
            self.optimizer.zero_grad()



            # inputs = T(inputs, cuda=self.use_cuda)
            # inputs, targets = Variable(inputs, requires_grad=False), \
            #                                   Variable(targets)

            # seq_len = inputs.size(0)
            outputs = self.model(inputs) #.view(seq_len,-1,9)

            # Create mask
            mask = create_mask(sequence_lengths, targets, cuda=self.use_cuda)

            # Get CRF Loss
            loss = -1*self.criterion(outputs, targets, mask=mask)
            loss.backward()
            self.optimizer.step()
            # if self.config.use_transformer:
            #     self.scheduler.step()

            # Callbacks
            train_loss += loss.item()
            predictions = self.criterion.decode(outputs, mask=mask)
            masked_targets = mask_targets(targets, sequence_lengths)

            t_ = mask.type(torch.LongTensor).sum().item()
            total += t_
            c_ = sum([1 if p[i] == mt[i] else 0 for p, mt in zip(predictions, masked_targets) for i in range(len(p))])
            correct += c_

            prog.update(batch_idx + 1, values=[("train loss", loss.item())], exact=[("Accuracy", 100*c_/t_)])

        self.logger.info("Train Loss: %.3f, Train Accuracy: %.3f%% (%d/%d)" %(train_loss/(total_step+1), 100.*correct/total, correct, total) )

    def test_laser(self, nbatches_val, val_generator, fine_tune=False):
        self.model.eval()
        accs = []
        test_loss = 0
        correct_preds = 0
        total_correct = 0
        total_preds = 0
        # total_step = None

        for batch_idx, (inputs, word_lens, sequence_lengths, targets) in enumerate(val_generator):
            if batch_idx == nbatches_val: break
            if targets.shape[0] == self.config.batch_size:
                # self.logger.info('Skipping batch of size=1')
                continue

            total_step = batch_idx
            #targets = T(targets, cuda=self.use_cuda).transpose(0,1).contiguous()


            # inputs = T(inputs, cuda=self.use_cuda)
            # inputs, targets = Variable(inputs, requires_grad=False), \
            #                                   Variable(targets)
            # seq_len = inputs.size(0)
            outputs = self.model((inputs,word_lens)) #.view(seq_len,-1,9)

            # Create mask
            mask = create_mask(sequence_lengths, targets, cuda=self.use_cuda)

            # Get CRF Loss
            loss = -1*self.criterion(outputs, targets, mask=mask)

            # Callbacks
            test_loss += loss.item()
            predictions = self.criterion.decode(outputs, mask=mask)
            masked_targets = mask_targets(targets, sequence_lengths)

            for lab, lab_pred in zip(masked_targets, predictions):

                accs += [1 if a==b else 0 for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.label_to_idx))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.label_to_idx))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        self.logger.info("Val Loss : %.3f, Val Accuracy: %.3f%%, Val F1: %.3f%%" %(test_loss/(total_step+1), 100*acc, 100*f1))
        return 100*f1
    def test_base(self, nbatches_val, val_generator, fine_tune=False):
        self.model.eval()
        accs = []
        test_loss = 0
        correct_preds = 0
        total_correct = 0
        total_preds = 0
        total_step = None

        for batch_idx, (inputs, sequence_lengths, targets) in enumerate(val_generator):
            if batch_idx == nbatches_val: break
            if targets.shape[0] == self.config.batch_size:
                # self.logger.info('Skipping batch of size=1')
                continue

            total_step = batch_idx
            outputs = self.model(inputs) #.view(seq_len,-1,9)

            # Create mask
            mask = create_mask(sequence_lengths, targets, cuda=self.use_cuda)

            # Get CRF Loss
            loss = -1*self.criterion(outputs, targets, mask=mask)

            # Callbacks
            test_loss += loss.item()
            predictions = self.criterion.decode(outputs, mask=mask)
            masked_targets = mask_targets(targets, sequence_lengths)

            for lab, lab_pred in zip(masked_targets, predictions):

                accs += [1 if a==b else 0 for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.label_to_idx))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.label_to_idx))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        self.logger.info("Val Loss : %.3f, Val Accuracy: %.3f%%, Val F1: %.3f%%" %(test_loss/(total_step+1), 100*acc, 100*f1))
        return 100*f1

    def evaluate(self,test):
        batch_size = self.config.batch_size
        nbatches_test, test_generator = self.batch_iter(test, batch_size,
                                                        return_lengths=True)
        self.logger.info('Evaluating on test set')
        self.test(nbatches_test, test_generator)

    def predict_batch(self, words):
        self.model.eval()
        if len(words) == 1:
            mult = np.ones(2).reshape(2, 1).astype(int)





        # word_ids, sequence_lengths = pad_sequences(words, 1)
        #
        # word_ids = np.asarray(word_ids)
        #
        # if len(words) == 1:
        #     word_ids = mult*word_ids

        # word_input = T(word_ids, cuda=self.use_cuda)
        #
        # inputs = Variable(word_input, requires_grad=False)

        outputs = self.model(words)

        predictions = self.criterion.decode(outputs)

        # predictions = [p[:i] for p, i in zip(predictions, sequence_lengths)]

        return predictions

    def predict(self, sentences):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        nlp = spacy.load('en') #TODO: replace spacy with LASER tokenizer -- discuss
        doc = nlp(sentences)
        words_raw = [[token.text for token in sent] for sent in doc.sents]

        words = [[self.config.processing_word(w) for w in s] for s in words_raw]
            # print(words)
            # raise NameError('testing')
            # if type(words[0]) == tuple:
            #     words = zip(*words)

        pred_ids = self.predict_batch(words)
        preds = [[self.idx_to_tag[idx.item() if isinstance(idx, torch.Tensor) else idx]  for idx in s] for s in pred_ids]

        return preds


def create_mask(sequence_lengths, targets, cuda, batch_first=False):
    """ Creates binary mask """
    mask = Variable(torch.ones(targets.size()).type(torch.ByteTensor))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mask = mask.to(device)

    for i,l in enumerate(sequence_lengths):
        if batch_first:
            if l < targets.size(1):
                mask.data[i, l:] = 0
        else:
            if l < targets.size(0):
                # print((l, targets.size(0)))
                mask.data[l:, i] = 0

    return mask


def mask_targets(targets, sequence_lengths, batch_first=False):
    """ Masks the targets """
    if not batch_first:
         targets = targets.transpose(0,1)
    t = []
    for l, p in zip(targets,sequence_lengths):
        t.append(l[:p].data.tolist())
    return t




