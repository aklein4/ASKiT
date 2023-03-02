
import torch
from transformers import get_cosine_schedule_with_warmup

from agent import Agent

import json
import random
import csv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import sys
sys.path.append("../utils")
from train_utils import Logger, train, get_mem_use


# files to load data jsons from
TRAIN_FILE = "../local_data/hotpot_data/train.json"
VAL_FILE = "../local_data/hotpot_data/val.json"

# folder to save checkpoints to
CHECKPOINT = "./checkpoints/agent"
# csv file to save progress logs to
LOG = "./logs/agent.csv"
# png file to save graph of progress to
GRAFF = "./logs/agent.png"

# training hyperparameters
LR = 1e-6
BATCH_SIZE = 8

# number of actions to choose from, including submit
N_ACTIONS = 8

# only train of 1/skip of the data
SKIP = 2

# truncate the training data to this many examples
TRAIN_TRUNC = 20000
# truncate the validation data to this many examples
VAL_TRUNC = 2000

# if the memory usage is above this fraction of the total memory, clear cache
MEM_THRESH = 0.85


class ChooseDataset:

    def __init__(self, file, n_choices, device=torch.device("cpu"), trunc=1000000):
        """ Dataset for training the agent to choose the next action.
        - Does not shuffle evidence nor add noise and frens
        - Action choices are chosen randomly from the corpus

        Args:
            file (str): json file to load data from
            n_choices (int): Number of actions to choose from, including submit
            device (torch.device, optional): Device to store targets on. Defaults to torch.device("cpu").
            trunc (int, optional): Truncate data to this length. Defaults to 1000000.
        """

        # save params
        self.device = device
        self.n_choices = n_choices

        # json of all the data
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)

        # truncate 
        self.data = self.data[:trunc]
        
        # add raw (not split into articles) corpus for evidence indexing
        for p in self.data:
            p["raw_corpus"] = []
            for c in range(len(p["corpus"])):
                title = " "+ p["corpus_titles"][c] + ", "
                for s in p["corpus"][c]:
                    p["raw_corpus"].append(title + s)

        # data that will be sent out
        self.x = []
        self.y = []

        # prepare target data
        for p in self.data:
            
            # evidence actions, one-hot in second position
            for k in range(len(p["evidence_sentences"])):            
                self.y.append(torch.zeros([self.n_choices], device=self.device, dtype=torch.float32))
                self.y[-1][1] = 1
                
            # submit action, one-hot in first position
            self.y.append(torch.zeros([self.n_choices], device=self.device, dtype=torch.float32))
            self.y[-1][0] = 1
        
        # number of examples
        self.size = len(self.y)

        # map indices to a different random index
        self.shuffler = []

        # init missing data
        self.reset()
    

    def to(self, dev):
        # move targets to device
        self.device = torch.device(dev) if isinstance(dev, str) else dev
        for t in self.targets:
            for elem in t:
                elem.to_(self.device)
    

    def shuffle(self):
        # randomly shuffle indices and select random actions

        random.shuffle(self.shuffler)
        self._updateData()


    def reset(self):
        # reset to deterministic state

        self.shuffler = list(range(self.size))

        seed = random.randrange(0xFFFF)
        random.seed(0)
        self._updateData()
        random.seed(seed)


    def _updateData(self):
        """ Create the input data to be sent out
        """

        self.x = []

        for p in tqdm(self.data, leave=False, desc="Loading"):
            
            question = p["question"]
            corpus = p["raw_corpus"]
            
            # evidence to accumulate
            evidence = ""
            
            for e in p["evidence_raw_ids"]:
                
                # actions to choose from in this state
                curr_acts = [corpus[e]]

                # cannot have the same evidence twice
                avail = corpus.copy()
                avail.pop(e)

                # choose random actions to fill the rest of the choices
                curr_acts += random.choices(avail, k = self.n_choices - 2)
            
                # save as tuple
                self.x.append((question, evidence, curr_acts))
                
                # add correct action to the next state's evidence
                evidence += curr_acts[0]
            
            # submit action
            curr_acts = random.choices(corpus, k = self.n_choices - 1)
            self.x.append((question, evidence, curr_acts))

        # make sure that everything lines up
        assert len(self.x) == len(self.y)


    def __len__(self):
        # number of examples
        return self.size


    def __getitem__(self, getter):
        """ Get a batch at index, of length batchsize

        Args:
            getter (_type_): Either int for index, or (index: int, batchsize: int) tuple

        Returns:
            tuple: ((questions, evidence, actions), y) tuple, questions and evidence are lists of strings, actions is a list of lists of strings, y is a tensor of targets
        """

        # clear cache if memory is getting full
        if get_mem_use() >= MEM_THRESH:
            torch.cuda.empty_cache()
        
        # unpack index and batchsize
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # get the indices we are going to use
        indices = self.shuffler[index:index+batchsize]

        x = ([], [], [])
        y = []
        
        # unpack data tuples onto batch tuples
        for i in indices:
            q, e, a = self.x[i]
            
            x[0].append(q)
            x[1].append(e)
            x[2].append(a)
            
            y.append(self.y[i])

        return x, torch.stack(y)


class AgentLogger(Logger):
    def __init__(self, log_loc=LOG, graff=GRAFF):
        """ Keeps track of metrics throughout training to log, checkpoint, and save

        Args:
            log_loc (str, optional): Location to save metric csv. Defaults to LOG.
            graff (str, optional): Location to save metric graph. Defaults to GRAFF.
        """

        # accuracies
        self.train_accs = []
        self.val_accs = []

        # correct action probabilities
        self.train_probs = []
        self.val_probs = []

        # this is the checkpoint metric
        self.best_val_prob = 0

        # metric object
        self.p_fn = PMetric()

        # save locations
        self.log_loc = log_loc
        self.graff = graff

        # create metric file and write header
        with open(self.log_loc, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(["epoch", "train_prob", "val_prob", "train_acc", "val_acc"])


    def initialize(self, model):
        # get reference to the model so we can save it
        self.model = model
    

    def log(self, train_log, val_log):
        """ Log the metrics from the current epoch, and save the model if it is the best so far

        Args:
            train_log (tuple): (train_pred, train_y) lists of tensors
            val_log (tuple): (val_pred, val_y) lists of tensors
        """

        # unpack training data into tensors
        train_pred, train_y = train_log
        train_pred = torch.cat(train_pred, dim=0)
        train_y = torch.cat(train_y, dim=0)

        # unpack validation data into tensors
        val_pred, val_y = val_log
        val_pred = torch.cat(val_pred, dim=0)
        val_y = torch.cat(val_y, dim=0)

        # calculate train metrics
        train_metric = self.p_fn(train_pred, train_y, dtype=np.float32)
        self.train_probs.append(train_metric[0])
        self.train_accs.append(train_metric[1])
        
        # calculate val metrics
        val_metric = self.p_fn(val_pred, val_y, dtype=np.float32)
        self.val_probs.append(val_metric[0])
        self.val_accs.append(val_metric[1])
        
        # append metrics to csv file
        with open(self.log_loc, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.train_accs)-1, self.train_probs[-1], self.val_probs[-1], self.train_accs[-1], self.val_accs[-1]])

        # plot the metrics
        fig, ax = plt.subplots(2)

        ax[0].plot(self.val_probs)
        ax[0].plot(self.train_probs)
        ax[0].set_title(r"% Policy Choose Correct")
        ax[0].legend(["val_prob", "train_prob"])

        ax[1].plot(self.val_accs)
        ax[1].plot(self.train_accs)
        ax[1].set_title(r"% Greedy Choose Correct")
        ax[1].legend(["val_acc", "train_acc"])

        plt.tight_layout()
        plt.savefig(self.graff)
        plt.clf()

        # check metric for checkpoint saving
        if self.best_val_prob < self.val_probs[-1] or True: # overriden to save every epoch
            self.best_val_prob = self.val_probs[-1]
            self.save_checkpoint()
    

    def save_checkpoint(self):
        # save the model to a new folder

        # create folder
        folder = CHECKPOINT + "_{}".format(len(self.val_accs)-1)
        os.makedirs(folder, exist_ok=True)

        # save act_model
        self.model.act_tokenizer.save_pretrained(os.path.join(folder, "act_tokenizer"))
        self.model.act_encoder.save_pretrained(os.path.join(folder, "act_encoder"))
        
        # save sub_model
        self.model.sub_tokenizer.save_pretrained(os.path.join(folder, "sub_tokenizer"))
        self.model.sub_encoder.save_pretrained(os.path.join(folder, "sub_encoder"))


def PLoss(pred, target):
    """ Creates a loss function that maximizes the average log probability of the correct action

    Args:
        pred (tensor): [batch size, num actions] prediction tensor
        target (tensor): [batch size, num actions] target tensor

    Returns:
        _type_: [1] loss tensor
    """
    assert pred.shape == target.shape
    
    # get log probabilities of correct actions
    probs = torch.nn.functional.log_softmax(pred, dim=-1)
    chosen = probs[target == 1]
    
    # retur average, negative to minimize
    return -torch.sum(chosen) / pred.shape[0]


class PMetric:
    def __init__(self):
        """ Calculates probability and accuracy metrics
        """

        # train function print name
        self.title = "p/acc"
    

    def __call__(self, pred, target, dtype=np.float16):
        """ Calculate the probability and accuracy metrics, returned as a numpy array

        Args:
            pred (tensor): [batch size, num actions] prediction tensor
            target (tensor): [batch size, num actions] target tensor
            dtype (np.dtype, optional): Type of array to return. Defaults to np.float16.

        Returns:
            _type_: [prob, acc] numpy array
        """
        assert pred.shape == target.shape
        
        # calculate policy probabilities
        probs = torch.nn.functional.softmax(pred, dim=-1)
        probs = probs[target == 1]
        
        # calcualte accuracy
        acc = torch.sum(torch.argmax(pred, dim=1) == torch.argmax(target, dim=1)).item() / pred.shape[0]

        # return rounded averages in numpy array
        return np.array([round(torch.sum(probs).item() / pred.shape[0], 3), round(acc, 3)]).astype(dtype)


def main():

    # load data
    train_data = ChooseDataset(TRAIN_FILE, N_ACTIONS, device=torch.device("cuda"), trunc=TRAIN_TRUNC)
    val_data = ChooseDataset(VAL_FILE, N_ACTIONS, device=torch.device("cuda"), trunc=VAL_TRUNC)

    # init agent
    model = Agent()
    model.to("cuda")

    # init loss function pointer
    loss_fn = PLoss

    # init our stuff
    logger = AgentLogger(log_loc=LOG, graff=GRAFF)
    met = PMetric()

    # init torch stuff
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=15000,
        num_training_steps=80000,
    )

    # train indefinitely
    train(model, optimizer, train_data, loss_fn, val_data=val_data, batch_size=BATCH_SIZE, logger=logger, lr_scheduler=lr_scheduler, skip=SKIP, rolling_avg=0.99, metric=met)


if __name__== '__main__':
    main()
