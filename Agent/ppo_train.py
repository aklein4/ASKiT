
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


# checkpoint locations to load pretrained models
AGENT_CHECK = "checkpoints/agent-pre"
SEARCH_CHECK = "checkpoints/searcher-p"


# path to json with training data in it
TRAIN_FILE = "../local_data/hotpot_data/train.json"
# path to encodings of training corpuses
TRAIN_ENCODINGS = "../local_data/corpus_encodings/train.pt"

# path to file with val data in it
VAL_FILE = "../local_data/hotpot_data/val.json"
# path to encodings of training corpuses
VAL_ENCODINGS = "../local_data/corpus_encodings/val.pt"

# folder to save checkpoints to
CHECKPOINT = "./checkpoints/quasi_ppo"
# csv file to save progress logs to
LOG = "./logs/quasi_ppo.csv"
# png file to save graph of progress to
GRAFF = "./logs/quasi_ppo.png"

# training hyperparameters
LR = 1e-6
BATCH_SIZE = 6

# number of actions to choose from, including submit
N_ACTIONS = 6

# clip hyperparameter for PPO Loss
CLIP_ALPHA = 0.2

# evey epoch, add 1/SKIP episodes to the replay buffer
SKIP = 300

# start the training data at this index (we pretrained on the first 20000 elements)
TRAIN_START = 20000

# truncate the validation data to this many examples
VAL_TRUNC = 500

# load the starting replay buffer from this location
INIT_BUF = None # "checkpoints/replay_buffer.pt"
# before the first epoch, fill the replay buffer with this many examples
MIN_BUF = 1000
# discard the oldest examples once the replay buffer eaches this size
MAX_BUF = 5000

# reduce training epoch size for debugging
TRAIN_SKIP = 1

# temperature coef for exploration sampling
EXPLORE_COEF = 3

# device to run training on
DEVICE = torch.device("cuda")


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

        # get metrics greedy rollout metrics from the training set
        train_f1, train_acc = self.train_env.evaluate()
        self.train_f1s.append(train_f1)
        self.train_accs.append(train_acc)

        # get greedy rollout metrics from the validation set
        val_f1, val_acc = self.val_env.evaluate()
        self.val_f1s.append(val_f1)
        self.val_accs.append(val_acc)

        # append metrics to csv file
        with open(self.log_loc, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.train_accs)-2, self.train_accs[-1], self.val_accs[-1], self.train_f1s[-1], self.val_f1s[-1]])

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
        folder = CHECKPOINT + "_{}".format(len(self.val_accs)-2)
        os.makedirs(folder, exist_ok=True)

        # save act_model
        self.model.act_tokenizer.save_pretrained(os.path.join(folder, "act_tokenizer"))
        self.model.act_encoder.save_pretrained(os.path.join(folder, "act_encoder"))
        
        # save sub_model
        self.model.sub_tokenizer.save_pretrained(os.path.join(folder, "sub_tokenizer"))
        self.model.sub_encoder.save_pretrained(os.path.join(folder, "sub_encoder"))


def PPOLoss(pred, target):
    """
    Calculate the loss according to Proximal Policy Optimization (PPO)
    """

    # old_policy are probabilities from when data was originally collected,
    # advantage is how good an action is relative to expectation
    mask, old_policy, advantage = target

    assert pred.shape == old_policy.shape and pred.shape == advantage.shape and pred.shape == mask.shape
    
    J = torch.nn.functional.log_softmax(pred, dim=-1)[mask] * advantage[mask]
    return torch.sum(J) / pred.shape[0]

    # model output is logits -> convert to probabilities
    #probs = torch.nn.functional.softmax(pred, dim=-1)

    # apply action masking to all tensors
    #probs, old_policy, advantage = probs[mask], old_policy[mask], advantage[mask]

    # we use the old policy to regularize the current one
    #ratio = probs / old_policy

    # clip to avoid the policy from making too big a change at once
    #clipped_ratio = torch.clip(ratio, 1-CLIP_ALPHA, 1+CLIP_ALPHA)
    
    # we want higher advantage choices to have higher ratio
    #A_r = advantage * ratio
    #A_clipped = advantage * clipped_ratio

    # this handles high/low clip vs. +/- advantage
    # (just think about the cases)
    #minned = torch.minimum(A_r, A_clipped)

    # return average, negative to minimize
    #return -torch.sum(minned) / pred.shape[0]


def main():

    # load semantic search model
    search = Searcher(load=SEARCH_CHECK)
    search = search.to(DEVICE)

    # load agent model
    model = Agent(load=AGENT_CHECK)
    model = model.to(DEVICE)

    # load data
    train_env = Environment(TRAIN_FILE, TRAIN_ENCODINGS, search, model, N_ACTIONS, device=torch.device(DEVICE), skip=SKIP, data_start=TRAIN_START, max_buf=MAX_BUF, min_buf=MIN_BUF, init_buffer=INIT_BUF, exploration_coefficient=EXPLORE_COEF)
    val_env = Environment(VAL_FILE, VAL_ENCODINGS, search, model, N_ACTIONS, device=torch.device(DEVICE), data_end=VAL_TRUNC, max_buf=1)

    # init loss function pointer
    loss_fn = PPOLoss

    # init our stuff
    logger = PPOLogger(train_env, val_env, log_loc=LOG, graff=GRAFF)
    logger.initialize(model)
    logger.log(None, None)

    # init torch stuff
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10000,
        num_training_steps=50000,
    )

    # train indefinitely
    train(model, optimizer, train_env, loss_fn, val_data=None, batch_size=BATCH_SIZE, logger=logger, lr_scheduler=lr_scheduler, skip=TRAIN_SKIP, rolling_avg=0.99)


if __name__== '__main__':
    main()
