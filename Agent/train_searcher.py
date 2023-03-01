
import torch
from transformers import get_cosine_schedule_with_warmup

from searcher import Searcher

import json
import random
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("../utils")
from train_utils import Logger, train

"""
Training script for Searcher.
May need need to updated as other files change.
"""


# path to json with training data in it
TRAIN_FILE = "../local_data/hotpot_data/train.json"
# path to encodings of training corpuses
TRAIN_ENCODINGS = "../local_data/corpus_encodings/train.pt"

# path to file with val data in it
VAL_FILE = "../local_data/hotpot_data/val.json"
# path to encodings of training corpuses
VAL_ENCODINGS = "../local_data/corpus_encodings/val.pt"

# file location to save checkpoint (make sure folder exists)
CHECKPOINT = "./checkpoints/searcher-p_scaled"
# file location to save metric logs (make sure folder exists)
LOG = "./logs/searcher-p_scaled.csv"
# file location to save metric graph (make sure folder exists)
GRAFF = "./logs/searcher-p_scaled.png"

# base learning rate
LR = 1e-6
# batch size
BATCH_SIZE = 24

# number of noise corpuses to add for each question
N_FRENS = 1
# amount of noise evidence to add to states, see use
NOISE_DECAY = 2
# If using TopKLoss, this is k
TOP_K = 5

# when training, can divide the sets by this number for debugging
SKIP = 1
# only the the first TRUNC elements from each dataset (for seperation/debugging)
TRUNC = 20000

class SearchDataset:

    def __init__(self, file, corpus_encodings, n_frens=None, noise_decay=None, device=torch.device("cpu")):
        """ Create a Dataset that can automatically shuffle, noise, and set targets for searcher on hotpotqa.

        Args:
            file (_type_): path to data json
            corpus_encodings (_type_): path to corpus encodings .pt
            n_frens (_type_, optional): number of noise corpuses per question. Defaults to None.
            noise_decay (_type_, optional): Amount of evidence to add. 'rate' of rounded exponential distribution. Defaults to None.
            device (_type_, optional): device to store stuff on. Defaults to torch.device("cpu").
        """

        # save device
        self.device = device

        # json of all this data
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)

        # truncate it
        self.data = self.data[:TRUNC]

        # how big is it?
        self.size = len(self.data)

        # load all of the embeddings
        self.corpus = torch.load(corpus_encodings)
        for i in range(len(self.corpus)):
            self.corpus[i] = self.corpus[i].to(self.device)
            self.corpus[i].requires_grad = False

        # truncate that too
        self.corpus = self.corpus[:TRUNC]

        # check that things match up
        assert len(self.corpus) == self.size

        # get targets with 1 as evidence, zero else (corresponding to emeddings)
        self.targets = []
        for i in range(len(self.data)):
            targ = torch.zeros((self.corpus[i].shape[0],), dtype=torch.float32, device=self.device)
            targ[self.data[i]["evidence_raw_ids"]] = 1 # list as index
            self.targets.append(targ)
        
        # number of extra corpuses per question
        self.n_frens = n_frens
        # amount of noise evidence that is added (rate of exponential distribution)
        self.noise_decay = noise_decay

        # hold the actual data that we send out
        self.x = []
        self.y = []

        # each question will get a random subset of its evidence
        self.temp_evidence = []

        # states that will be used to prompt
        self.states = []

        # hold the fren index list
        self.frens = []

        # init out how much noise will go with each question
        self.noise_generator = torch.distributions.exponential.Exponential(rate=self.noise_decay) if self.noise_decay is not None else None
        self.noise_amounts = None if self.noise_decay is None else []
        self.noise = None if self.noise_decay is None else []

        # map indices to a different random index
        self.shuffler = []

        # init all this empty stuff
        self.reset()
    

    def cpu(self):
        # send to cpu
        self.device = torch.device("cpu")
        self._update_device(self)

    def cuda(self):
        # send to gpu
        self.device = torch.device("cuda")
        self._update_device(self)

    def _update_device(self):
        # move all tensors
        for i in range(self.size):
            self.corpus[i] = self.corpus[i].to(self.device)
            self.targets[i] = self.targets[i].to(self.device)
    

    def shuffle(self):
        """ Randomly rearrange the data, add random noise, evidence and frens.
        """

        self._generateEvidence()
        self._generateNoise()
        self._generateStates()

        random.shuffle(self.frens)
        random.shuffle(self.shuffler)

        self._updateData()


    def reset(self):
        """ Reset data to deterministic state. 'Random' things are made deterministic with seeding.
        """

        # these can just be identities
        self.frens = list(range(self.size))
        self.frens.append(self.frens.pop(0)) # barrel shift to avoid collisions
        self.shuffler = list(range(self.size))

        # noise uses torch random
        torch.manual_seed(0)
        self._generateNoise()
        torch.manual_seed(random.randrange(0xfffe))
        
        # these use random random
        random.seed(0)
        self._generateEvidence()
        self._generateStates()
        random.seed(torch.randint(0xfffe, size=(1,)).item())

        self._updateData()


    def _generateEvidence(self):
        """ Randomly select some evidence subset to provide to each question.
        """
        self.temp_evidence = []

        for i in range(len(self)):
            # size of question's evidence set
            n_evidence = len(self.data[i]["evidence_sentences"])

            # choose random subset, |subset| ~ uniform
            self.temp_evidence.append(
                random.choices(
                    range(n_evidence),
                    k = random.randrange(n_evidence) # can be none, cannot be all
                )
            )


    def _generateNoise(self):
        """ Add some random noise to temp_evidence that doesn't belong there.
        """

        # if none, we don't use noise
        if self.noise_decay is None:
            self.noise_amounts = None
            self.noise = None

        else:
            # sample from exponential distribution, round (0 is most commmon)
            self.noise_amounts = torch.round(self.noise_generator.sample((self.size,))).tolist()
            
            self.noise = []
            for i in range(self.size):
                noise_i = []

                for n in range(int(self.noise_amounts[i])):
                    # select random corpus
                    article = torch.randint(len(self.data[i]["corpus_titles"]), size=(1,)).item()
                    # select random sentence in that corpus
                    sentence = torch.randint(len(self.data[i]["corpus"][article]), size=(1,)).item()
                    # save both title and sentence, for preprocessing
                    noise_i.append((article, sentence))
                self.noise.append(noise_i)


    def _generateStates(self):
        """ Combine the questions, evidence, and noise into the model's input.
        """
        self.states = []

        for i in range(len(self)):
            p = self.data[i]

            # a list of all the things (evidence and noise) that come after the question
            parts = []
            for e in self.temp_evidence[i]:
                # THIS IS INPUT FORMAT FOR MODEL
                parts.append(" " + p["evidence_titles"][e] + ": " + p["evidence_sentences"][e])

            for n_title, n_ind in self.noise[i]:
                parts.append(" " + p["corpus_titles"][n_title] + ": " + p["corpus"][n_title][n_ind])

            # put evidence and noise in random order
            random.shuffle(parts)
            
            # cat the evidence/noise onto the question to get full input
            state = p["question"]
            for part in parts:
                state += part
            self.states.append(state)


    def _updateData(self):
        """ Generate x and y based on the other current variables.
        """

        # this is to help with a 'memory leak'
        self.prev_x_1 = None

        self.x = []
        self.y = []

        for i in tqdm(range(len(self)), desc="Loading", leave=False):

            # get the target for this question, make sure to clone
            target = self.targets[i].clone()
            target[self.temp_evidence[i]] = 0
            target = [target]

            # get corus, not changed so don't need to clone
            x_corpus = [self.corpus[i]]

            for c in range(self.n_frens):
                # get fren as local indices from fren list
                fren = self.corpus[self.frens[(i+c) % self.size]]

                # add fren corpus, and extend target with zeros for each of the fren's sentences
                x_corpus.append(fren)
                target.append(torch.zeros([fren.shape[0]], dtype=torch.float32, device=self.device))

            assert len(x_corpus) == len(target)

            # x format is input to model
            self.x.append((self.states[i], x_corpus))
            self.y.append(target)


    def __len__(self):
        # get number of data points
        return self.size

    def __getitem__(self, getter):
        """ Get a batch at (index, batch_size)

        Args:
            getter (_type_): Either int for index, or (index: int, batchsize: int)

        Returns:
            _type_: ((state, corpus), y) tuple, each is list of inputs, corpuses, and targete
        """

        # split up input
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # get the indices of the data points we will use
        indices = self.shuffler[index:index+batchsize]

        # states
        x_0 = []
        # corpuses
        x_1 = []
        # targets
        y = []
        for ind in indices:
            # split into seperate lists
            state, enc = self.x[ind]
            x_0.append(state)
            x_1.append(torch.cat(enc).to(torch.float32)) # we cat/float here to save memory

            y.append(torch.cat(self.y[ind]))

        # because fuck me, that's why
        # (if this isn't here, then memory use constantly increases)
        if self.prev_x_1 is not None:
            old_x, old_y = self.prev_x_1
            for k in range(len(old_x)):
                old_x[k].detach_()
                old_y[k].detach_()
        self.prev_x_1 = (x_1, y)
        torch.cuda.empty_cache() # this may be the thing that's actually fixing the problem

        return (x_0, x_1), y


class TopKCrossEntropy(torch.nn.Module):

    def __init__(self, k):
        """ Get a module that computes corss entropy, but only among the
        top k predictions (could help with semi-supervization)

        Args:
            k (_type_): number of elements to calc loss on
        """
        super().__init__()

        self.k = k

    def forward(self, pred, target):
        """ Calculate the loss

        Args:
            pred (_type_): List of 1d prediction tensors
            target (_type_): List of 1d target tensors

        Returns:
            _type_: Loss to minimize
        """
        assert len(pred) == len(target)

        # list of top-k for each
        pred_stack = []
        target_stack = []

        for i in range(len(pred)):
            # get top k indexes
            vals, inds = torch.topk(pred[i], self.k)

            # pull them out of tensors
            pred_stack.append(pred[i][inds])
            target_stack.append(target[i][inds])

        # create batched tensors
        pred_batch = torch.stack(pred_stack)
        target_batch = torch.stack(target_stack)

        return torch.nn.functional.cross_entropy(pred_batch, target_batch)


def MaxPLoss(pred, target):
    """ Loss function to maximize the sum of the of probabilities of choosing 
    the correct element (see latent search paper)

    Args:
        pred (_type_): list of 1d prediction tensors
        target (_type_): list of 1d target tensors

    Returns:
        _type_: Loss to minimize
    """
    assert len(pred) == len(target)

    # similar batching process to TopKCrossEntropy
    pred_stack = []
    target_stack = []

    # we need to truncate to some same size, this should be fine
    b_size = min([p.numel() for p in pred])

    # truncate tensors
    for i in range(len(pred)):
        vals, inds = torch.topk(pred[i], b_size)
        pred_stack.append(pred[i][inds])
        target_stack.append(target[i][inds])

    # stack into batch
    pred_batch = torch.stack(pred_stack)
    target_batch = torch.stack(target_stack)

    # get the -> LOG <- probabilities to priotitize removing false negatives
    log_p = torch.nn.functional.log_softmax(pred_batch, dim=-1)

    # sum and norm by number of batches. Make negative for min goal instead of max
    loss = torch.sum(torch.where(target_batch == 1, log_p, torch.zeros_like(log_p)))
    return -loss / len(pred)


class SearchLogger(Logger):
    def __init__(self, log_loc=LOG, graff=GRAFF):
        """ Keeps track of metrics throughout training to log, checkpoint and save

        Args:
            log_loc (_type_, optional): Location to save metric csv. Defaults to LOG.
            graff (_type_, optional): Location to save metric graph. Defaults to GRAFF.
        """

        # whole bunch of stuff to track
        self.train_accs = []
        self.val_accs = []

        self.train_percs = []
        self.val_percs = []

        self.train_probs = []
        self.val_probs = []

        # this is the checkpoint metric
        self.best_val_prob = 0

        # save locations
        self.log_loc = log_loc
        self.graff = graff

        # create metic file and write header
        with open(self.log_loc, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(["epoch", "train_prob", "val_prob", "train_perc", "val_perc", "train_acc", "val_acc"])


    def initialize(self, model):
        """ Get the models components as references so we can save checkpoints

        Args:
            model (_type_): Searcher model that is training
        """
        # these are the only components that we need to save
        self.tokenizer = model.search_tokenizer
        self.model = model.search_encoder
    

    def log(self, train_log, val_log):
        """ Called during trining, executes class functionality

        Args:
            train_log (_type_): (pred, target ) tuple of list of list of tensors from training
            val_log (_type_): (pred, target ) tuple of list of of list tensors from validation
        """

        # poorly decomposed way of storing metrics
        this_train_acc = 0
        this_train_perc = 0
        this_train_prob = 0
        train_seen = 0 # to average

        # mirror of above
        this_val_acc = 0
        this_val_perc = 0
        this_val_prob = 0
        val_seen = 0

        # cat batches into one long list
        train_pred_batched, train_y_batched = train_log
        train_pred = []
        train_y = []
        for i in range(len(train_pred_batched)):
            train_pred += train_pred_batched[i]
            train_y += train_y_batched[i]
        assert len(train_pred) == len(train_y)
        
        for t in range(len(train_pred)):
            # avoid div by zero
            if torch.all(torch.sum(train_y[t]) == 0):
                continue

            # for average at end
            train_seen += 1

            # the highest prediction
            highest_ev = torch.max(train_pred[t][train_y[t] == 1])

            if highest_ev == torch.max(train_pred[t]).item():
                # prediction was correct
                this_train_acc += 1
                this_train_perc += 1
            else:
                # prediction was incorrrect, calc number that beat it
                beat_by = torch.sum(torch.where(train_pred[t] > torch.max(train_pred[t][train_y[t] == 1]), 1, 0)).item()
                this_train_perc += (1+beat_by) / train_pred[t].numel()

            # calc total non-log prob
            this_train_prob += torch.sum(torch.nn.functional.softmax(train_pred[t], dim=-1)[train_y[t] == 1]).item()

        # mirror above for val
        val_pred_batched, val_y_batched = val_log
        val_pred = []
        val_y = []
        for i in range(len(val_pred_batched)):
            val_pred += val_pred_batched[i]
            val_y += val_y_batched[i]
        assert len(val_pred) == len(val_y)
        
        for t in range(len(val_pred)):
            if torch.all(torch.sum(val_y[t]) == 0):
                continue
            val_seen += 1

            highest_ev = torch.max(val_pred[t][val_y[t] == 1])
            if highest_ev == torch.max(val_pred[t]).item():
                this_val_acc += 1
                this_val_perc += 1
            else:
                beat_by = torch.sum(torch.where(val_pred[t] > torch.max(val_pred[t][val_y[t] == 1]), 1, 0)).item()
                this_val_perc += (1+beat_by) / val_pred[t].numel()

            this_val_prob += torch.sum(torch.nn.functional.softmax(val_pred[t], dim=-1)[val_y[t] == 1]).item()

        # get averages
        this_train_acc /= max(1, train_seen)
        this_train_perc /= max(1, train_seen)
        this_train_prob /= max(1, train_seen)

        this_val_acc /= max(1, val_seen)
        this_val_perc /= max(1, val_seen)
        this_val_prob /= max(1, val_seen)

        # save the metrics
        self.train_accs.append(this_train_acc)
        self.train_percs.append(this_train_perc)
        self.train_probs.append(this_train_prob)

        self.val_accs.append(this_val_acc)
        self.val_percs.append(this_val_perc)
        self.val_probs.append(this_val_prob)

        # append metrics to csv file
        with open(self.log_loc, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.train_accs)-1, this_train_prob, this_val_prob, this_train_perc, this_val_perc, this_train_acc, this_val_acc])

        # plot the metrics
        fig, ax = plt.subplots(3)

        ax[0].plot(self.val_probs)
        ax[0].plot(self.train_probs)
        ax[0].set_title("Softmax Prob of Evidence")
        ax[0].legend(["val_prob", "train_prob"])

        ax[1].plot(self.val_accs)
        ax[1].plot(self.train_accs)
        ax[1].set_title(r"% Evidence in Rank=1 Prediction")
        ax[1].legend(["val_acc", "train_acc"])

        ax[2].plot(self.val_percs)
        ax[2].plot(self.train_percs)
        ax[2].set_title("Percentile of Highest Evidence")
        ax[2].legend(["val_perc", "train_perc"])

        plt.tight_layout()
        plt.savefig(self.graff)
        plt.clf()

        # check metric for checkpoint saving
        if this_val_prob > self.best_val_prob:
            self.best_val_prob = this_val_prob
            self.save_checkpoint()
    

    def save_checkpoint(self):
        """ Save the state of the current model.
        """
        self.tokenizer.save_pretrained(CHECKPOINT+"-{}_tokenizer".format(len(self.val_percs)-1))
        self.model.save_pretrained(CHECKPOINT+"-{}".format(len(self.val_percs)-1))


def main():

    # load data
    train_data = SearchDataset(TRAIN_FILE, TRAIN_ENCODINGS, N_FRENS, NOISE_DECAY, device=torch.device("cuda"))
    val_data = SearchDataset(VAL_FILE, VAL_ENCODINGS, N_FRENS, NOISE_DECAY, device=torch.device("cuda"))

    # k_loss = TopKCrossEntropy(TOP_K)
    loss_fn = MaxPLoss

    # init stuff
    logger = SearchLogger()
    model = Searcher()
    model = model.cud()

    # only calc grad on the correct stuff
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.search_encoder.parameters():
        p.requires_grad = True
    for p in model.search_head.parameters():
        p.requires_grad = True

    # init training stuff
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=1000,
    )

    train(model, optimizer, train_data, loss_fn, val_data=val_data, batch_size=BATCH_SIZE, logger=logger, lr_scheduler=lr_scheduler, skip=SKIP)


if __name__== '__main__':
    main()
