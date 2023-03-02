
import torch
from transformers import get_cosine_schedule_with_warmup

from searcher import Searcher
from agent import Agent
from environment import Environment

import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../utils")
from train_utils import Logger, train, get_mem_use


AGENT_CHECK = "./checkpoints/agent_0"
SEARCH_CHECK = "./checkpoints/searcher-p"


# path to json with training data in it
TRAIN_FILE = "../local_data/hotpot_data/train.json"
# path to encodings of training corpuses
TRAIN_ENCODINGS = "../local_data/corpus_encodings/train.pt"

# path to file with val data in it
VAL_FILE = "../local_data/hotpot_data/val.json"
# path to encodings of training corpuses
VAL_ENCODINGS = "../local_data/corpus_encodings/val.pt"

# folder to save checkpoints to
CHECKPOINT = "./checkpoints/ppo"
# csv file to save progress logs to
LOG = "./logs/ppo.csv"
# png file to save graph of progress to
GRAFF = "./logs/ppo.png"

# training hyperparameters
LR = 1e-6
BATCH_SIZE = 8

# number of actions to choose from, including submit
N_ACTIONS = 8

CLIP_ALPHA = 0.1

# only train of 1/skip of the data
SKIP = 10

# start the training data at this index
TRAIN_START = 20000

# truncate the validation data to this many examples
VAL_TRUNC = 2000


class PPOLogger(Logger):
    def __init__(self, train_env, val_env, log_loc=LOG, graff=GRAFF):
        """ Keeps track of metrics throughout training to log, checkpoint, and save

        Args:
            log_loc (str, optional): Location to save metric csv. Defaults to LOG.
            graff (str, optional): Location to save metric graph. Defaults to GRAFF.
        """

        # accuracies
        self.train_accs = []
        self.val_accs = []

        # correct action probabilities
        self.train_f1s = []
        self.val_f1s = []

        # this is the checkpoint metric
        self.best_f1 = 0

        # metric objects
        self.train_env = train_env
        self.val_env = val_env

        # save locations
        self.log_loc = log_loc
        self.graff = graff

        # create metric file and write header
        with open(self.log_loc, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(["epoch", "train_acc", "val_acc", "train_f1", "val_f1"])


    def initialize(self, model):
        # get reference to the model so we can save it
        self.model = model
    

    def log(self, train_log, val_log):
        """ Log the metrics from the current epoch, and save the model if it is the best so far

        Args:
            train_log (tuple): (train_pred, train_y) lists of tensors
            val_log (tuple): (val_pred, val_y) lists of tensors
        """

        train_f1, train_acc = self.train_env.evaluate()
        self.train_f1s.append(train_f1)
        self.train_accs.append(train_acc)

        val_f1, val_acc = self.val_env.evaluate()
        self.val_f1s.append(val_f1)
        self.val_accs.append(val_acc)

        # append metrics to csv file
        with open(self.log_loc, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.train_accs)-1, self.train_accs[-1], self.val_accs[-1], self.train_f1s[-1], self.val_f1s[-1]])

        # plot the metrics
        fig, ax = plt.subplots(2)

        ax[0].plot(self.val_accs)
        ax[0].plot(self.train_accs)
        ax[0].set_title(r"% Perfect Evidence Selection")
        ax[0].legend(["val_acc", "train_acc"])

        ax[1].plot(self.val_f1s)
        ax[1].plot(self.train_f1s)
        ax[1].set_title(r"F1 Score")
        ax[1].legend(["val_f1", "train_f1"])

        plt.tight_layout()
        plt.savefig(self.graff)
        plt.clf()

        # check metric for checkpoint saving
        if self.best_f1 < self.val_f1s[-1] or True: # overriden to save every epoch
            self.best_f1 = self.val_f1s[-1]
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


def PPOLoss(pred, target):
    old_policy, advantage = target

    assert pred.shape == old_policy.shape and pred.shape == advantage.shape
    
    ratio = pred / old_policy
    clipped_ratio = torch.clip(ratio, 1-CLIP_ALPHA, 1+CLIP_ALPHA)
    
    A_r = advantage * ratio
    A_clipped = advantage * clipped_ratio

    minned = torch.minimum(A_r, A_clipped)

    # return average, negative to minimize
    return -torch.sum(minned) / pred.shape[0]


def main():

    search = Searcher(load=SEARCH_CHECK)
    search.to("cuda")

    model = Agent(load=AGENT_CHECK)
    model.to("cuda")

    # load data
    train_env = Environment(TRAIN_FILE, TRAIN_ENCODINGS, search, model, N_ACTIONS-1, device=torch.device("cuda"), skip=SKIP, data_start=TRAIN_START)
    val_env = Environment(VAL_FILE, VAL_ENCODINGS, search, model, N_ACTIONS-1, device=torch.device("cuda"), data_end=VAL_TRUNC)

    # init agent
    model = Agent()

    # init loss function pointer
    loss_fn = PPOLoss

    # init our stuff
    logger = PPOLogger(train_env, val_env, log_loc=LOG, graff=GRAFF)

    # init torch stuff
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10000,
        num_training_steps=100000,
    )

    # train indefinitely
    train(model, optimizer, train_env, loss_fn, val_data=None, batch_size=BATCH_SIZE, logger=logger, lr_scheduler=lr_scheduler, skip=SKIP, rolling_avg=0.99)


if __name__== '__main__':
    main()