
import torch
from transformers import get_cosine_schedule_with_warmup

from agent import Agent

import json
import random
import csv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("../utils")
from train_utils import Logger, train, get_mem_use


TRAIN_FILE = "../local_data/hotpot_data/train.json"
VAL_FILE = "../local_data/hotpot_data/val.json"

CHECKPOINT = "./checkpoints/agent"
LOG = "./logs/agent.csv"
GRAFF = "./logs/agent.png"

LR = 1e-6
BATCH_SIZE = 8

N_ACTIONS = 8

SKIP = 1
TRUNC = 20000

MEM_THRESH = 0.8


class ChooseDataset:

    def __init__(self, file, n_choices, device=torch.device("cpu")):

        self.device = device
        self.n_choices = n_choices

        # json of all this data
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)
        self.data = self.data[:TRUNC]
        
        for p in self.data:
            p["raw_corpus"] = []
            for c in range(len(p["corpus"])):
                title = " "+ p["corpus_titles"][c] + ", "
                for s in p["corpus"][c]:
                    p["raw_corpus"].append(title + s)

        self.x = []
        
        self.y = []
        for p in self.data:
            
            for k in range(len(p["evidence_sentences"])):            
                self.y.append(torch.zeros([self.n_choices], device=self.device, dtype=torch.float32))
                self.y[-1][1] = 1
                  
            self.y.append(torch.zeros([self.n_choices], device=self.device, dtype=torch.float32))
            self.y[-1][0] = 1
        
        self.size = len(self.y)

        # map indices to a different random index
        self.shuffler = []

        # init all this empty stuff
        self.reset()
    

    def to(self, dev):
        self.device = torch.device(dev) if isinstance(dev, str) else dev
        for t in self.targets:
            for elem in t:
                elem.to_(self.device)
    

    def shuffle(self):

        random.shuffle(self.shuffler)

        self._updateData()


    def reset(self):

        self.shuffle = list(range(self.size))

        seed = random.randrange(0xFFFF)
        random.seed(0)
        self._updateData()
        random.seed(seed)


    def _updateData(self):

        self.x = []

        for p in tqdm(self.data, leave=False, desc="Loading"):
            
            question = p["question"]
            corpus = p["raw_corpus"]
            
            evidence = ""
            
            for e in range(p["evidence_raw_ids"]):
                
                curr_acts = [corpus[e]]

                avail = corpus.copy()
                avail.pop(e)
                curr_acts += random.choices(avail, self.n_choices - 2)
            
                self.x.apppend((question, evidence, curr_acts))
                
                evidence += curr_acts[0]
            
            curr_acts = random.choices(corpus, self.n_choices - 1)
            self.x.append((question, evidence, curr_acts))

        assert len(self.x) == len(self.y)


    def __len__(self):
        return self.size


    def __getitem__(self, getter):
        if get_mem_use() >= MEM_THRESH:
            torch.cuda.empty_cache()

        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        indices = self.shuffler[index:index+batchsize]

        x = ([], [], [])
        y = []
        
        for i in indices:
            q, e, a = self.x[i]
            
            x[0].append(q)
            x[1].append(e)
            x[2].append(a)
            
            y.append(self.y[i])

        return x, torch.stack(y)


class AgentLogger(Logger):
    def __init__(self, log_loc=LOG, graff=GRAFF):

        # whole bunch of stuff to track
        self.train_accs = []
        self.val_accs = []

        self.train_probs = []
        self.val_probs = []

        # this is the checkpoint metric
        self.best_val_acc = 0

        self.p_fn = PMetric()

        # save locations
        self.log_loc = log_loc
        self.graff = graff

        # create metic file and write header
        with open(self.log_loc, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(["epoch", "train_prob", "val_prob", "train_acc", "val_acc"])


    def initialize(self, model):
        """ Get the models components as references so we can save checkpoints

        Args:
            model (_type_): Searcher model that is training
        """
        # these are the only components that we need to save
        self.model = model
    

    def log(self, train_log, val_log):

        train_pred, train_y = train_log
        train_pred = torch.cat(train_pred, dim=0)
        train_y = torch.cat(train_y, dim=0)

        val_pred, val_y = val_log
        val_pred = torch.cat(val_pred, dim=0)
        val_y = torch.cat(val_y, dim=0)

        self.train_probs.append(self.p_fn(train_pred, train_y))
        self.val_probs.append(self.p_fn(val_pred, val_y))
        
        self.train_accs.append(torch.sum(torch.argmax(train_pred, dim=1) == torch.argmax(train_y, dim=1)) / train_pred.shape[0])
        self.val_accs.append(torch.sum(torch.argmax(val_pred, dim=1) == torch.argmax(val_y, dim=1)) / val_pred.shape[0])
        
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
        if self.best_val_acc < self.val_accs[-1]:
            self.best_val_acc = self.val_accs[-1]
            self.save_checkpoint()
    

    def save_checkpoint(self):
        folder = CHECKPOINT + "_{}".format(len(self.val_accs)-1)
        os.makedirs(folder, exist_ok=True)

        self.model.act_tokenizer.save_pretrained(os.path.join(folder, "act_tokenizer"))
        self.model.act_encoder.save_pretrained(os.path.join(folder, "act_encoder"))
        
        self.model.sub_tokenizer.save_pretrained(os.path.join(folder, "sub_tokenizer"))
        self.model.sub_encoder.save_pretrained(os.path.join(folder, "sub_encoder"))


def PLoss(pred, target):
    assert pred.shape == target.shape
        
    probs = torch.nn.functional.log_softmax(pred, dim=-1)
    chosen = probs[target == 1]
    
    return -torch.sum(chosen) / pred.shape[0]


class PMetric:
    def __init__(self):
        self.title = 'p'
    
    def __call__(self, pred, target):
        assert pred.shape == target.shape
        
        probs = torch.nn.functional.softmax(pred, dim=-1)
        chosen = probs[target == 1]
        
        return torch.sum(chosen) / pred.shape[0]


def main():

    train_data = ChooseDataset(TRAIN_FILE, N_ACTIONS, device=torch.device("cuda"))
    val_data = ChooseDataset(VAL_FILE, N_ACTIONS, device=torch.device("cuda"))

    model = Agent()
    model = model.cuda()

    loss_fn = PLoss

    logger = AgentLogger(log_loc=LOG, graff=GRAFF)
    met = PMetric()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10000,
        num_training_steps=50000,
    )

    train(model, optimizer, train_data, loss_fn, val_data=val_data, batch_size=BATCH_SIZE, logger=logger, lr_scheduler=lr_scheduler, skip=SKIP, rolling_avg=0.99, metric=met)


if __name__== '__main__':
    main()
