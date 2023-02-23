
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from agent import Agent

import json
import random
import csv
import matplotlib.pyplot as plt

import sys
sys.path.append("../utils")
from train_utils import Logger, train


TRAIN_FILE = "../local_data/hotpot_data/val.json"
TRAIN_ENCODINGS = "../local_data/corpus_encodings/val.pt"

VAL_FILE = "../local_data/hotpot_data/val.json"
VAL_ENCODINGS = "../local_data/corpus_encodings/val.pt"

CHECKPOINT = "./checkpoints/Agent"
LOG = "./logs/Agent.log"
GRAFF = "./logs/Agent.png"

LR = 1e-6
BATCH_SIZE = 32

N_FRENS = 1
NOISE_DECAY = 2
TOP_K = 5


class AgentDataset:

    def __init__(self, file, corpus_encodings, n_frens=None, noise_decay=None, device=torch.device("cpu")):

        self.device = device

        # json of all this data
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)

        # how big is it?
        self.size = len(self.data)

        # load all of the embeddings
        self.corpus = torch.load(corpus_encodings)
        for i in range(len(self.corpus)):
            self.corpus[i] = self.corpus[i].to(torch.float32).to(self.device)
            self.corpus[i].requires_grad = False

        # check that things match up
        assert len(self.corpus) == self.size

        # get targets with 1 as evidence, zero else (corresponding to emeddings)
        self.targets = []
        for i in range(len(self.data)):
            targ = torch.zeros((self.corpus[i].shape[0],), dtype=torch.float32, device=self.device)
            targ[self.data[i]["evidence_raw_ids"]] = 1
            self.targets.append(targ)
        
        # number of extra corpuses per question
        self.n_frens = n_frens
        # amount of noise evidence that is added (exponential distribution)
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

        # figure out how much noise will go with each question
        self.noise_generator = torch.distributions.exponential.Exponential(rate=self.noise_decay) if self.noise_decay is not None else None
        self.noise_amounts = None if self.noise_decay is None else []
        self.noise = None if self.noise_decay is None else []

        # map indices to a different random index
        self.shuffler = []

        # init all this empty stuff
        self.reset()
    

    def cpu(self):
        self.device = torch.device("cpu")
        self._update_device(self)

    def cuda(self):
        self.device = torch.device("cuda")
        self._update_device(self)

    def _update_device(self):
        for i in range(self.size):
            self.corpus[i] = self.corpus[i].to(self.device)
            self.targets[i] = self.targets[i].to(self.device)
    

    def shuffle(self):

        self._generateEvidence()
        self._generateNoise()
        self._generateStates()

        random.shuffle(self.frens)
        random.shuffle(self.shuffler)

        self._updateData()


    def reset(self):

        self.frens = list(range(self.size))
        self.frens.append(self.frens.pop(0))
        self.shuffler = list(range(self.size))

        torch.manual_seed(0)
        self._generateNoise()
        torch.manual_seed(random.randrange(0xfffe))
        
        random.seed(0)
        self._generateEvidence()
        self._generateStates()
        random.seed(torch.randint(0xfffe, size=(1,)).item())

        self._updateData()


    def _generateEvidence(self):
        self.temp_evidence = []

        for i in range(len(self)):
            n_evidence = len(self.data[i]["evidence_sentences"])

            self.temp_evidence.append(
                random.choices(
                    range(n_evidence),
                    k = random.randrange(n_evidence)
                )
            )


    def _generateNoise(self):

        if self.noise_decay is None:
            self.noise_amounts = None
            self.noise = None

        else:
            self.noise_amounts = torch.round(self.noise_generator.sample((self.size,))).tolist()
            self.noise = []
            for i in range(self.size):
                noise_i = []
                for n in range(int(self.noise_amounts[i])):
                    article = torch.randint(len(self.data[i]["corpus_titles"]), size=(1,)).item()
                    sentence = torch.randint(len(self.data[i]["corpus"][article]), size=(1,)).item()
                    noise_i.append((article, sentence))
                self.noise.append(noise_i)


    def _generateStates(self):
        self.states = []

        for i in range(len(self)):
            p = self.data[i]

            parts = []
            for e in self.temp_evidence[i]:
                parts.append(" " + p["evidence_titles"][e] + ": " + p["evidence_sentences"][e])

            for n_title, n_ind in self.noise[i]:
                parts.append(" " + p["corpus_titles"][n_title] + ": " + p["corpus"][n_title][n_ind])

            random.shuffle(parts)
            
            state = p["question"]
            for part in parts:
                state += part
            self.states.append(state)


    def _updateData(self):
        self.prev_x_1 = None

        self.x = []
        self.y = []

        for i in range(len(self)):

            target = self.targets[i].clone()
            target[self.temp_evidence[i]] = 0
            target = [target]

            x_corpus = [self.corpus[i]]

            for c in range(self.n_frens):
                fren = self.corpus[self.frens[(i+c) % self.size]]

                x_corpus.append(fren)
                target.append(torch.zeros([fren.shape[0]], dtype=torch.float32, device=self.device))

            x_corpus = torch.cat(x_corpus)
            target = torch.cat(target)
            assert x_corpus.shape[0] == target.shape[0]

            self.x.append((self.states[i], x_corpus))
            self.y.append(target)


    def __len__(self):
        return self.size

    def __getitem__(self, getter):
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        indices = self.shuffler[index:index+batchsize]

        x_0 = []
        x_1 = []
        y = []
        for ind in indices:
            state, enc = self.x[ind]
            x_0.append(state)
            x_1.append(enc)
            y.append(self.y[ind])

        if self.prev_x_1 is not None:
            for k in range(len(self.prev_x_1)):
                self.prev_x_1[k].detach_()
        self.prev_x_1 = x_1

        return (x_0, x_1), y


class TopKCrossEntropy(torch.nn.Module):

    def __init__(self, k):
        super().__init__()

        self.k = k

    def forward(self, pred, target):
        assert len(pred) == len(target)

        pred_stack = []
        target_stack = []

        for i in range(len(pred)):
            vals, inds = torch.topk(pred[i], self.k)
            pred_stack.append(pred[i][inds])
            target_stack.append(target[i][inds])

        pred_batch = torch.stack(pred_stack)
        target_batch = torch.stack(target_stack)

        return torch.nn.functional.cross_entropy(pred_batch, target_batch)


class AgentLogger(Logger):
    def __init__(self):
        self.train_accs = []
        self.val_accs = []

        self.train_percs = []
        self.val_percs = []

        self.best_val_perc = 0

        with open(LOG, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(["epoch", "train_perc", "val_perc", "train_acc", "val_acc"])


    def initialize(self, model):
        self.model = model
    

    def log(self, train_log, val_log):

        this_train_acc = 0
        this_train_perc = 0
        train_seen = 0

        this_val_acc = 0
        this_val_perc = 0
        val_seen = 0

        train_pred_batched, train_y_batched = train_log
        train_pred = []
        train_y = []
        for i in range(len(train_pred_batched)):
            train_pred += train_pred_batched[i]
            train_y += train_y_batched[i]
        assert len(train_pred) == len(train_y)
        
        for t in range(len(train_pred)):
            if torch.all(torch.sum(train_y[t]) == 0):
                continue
            train_seen += 1

            highest_ev = torch.max(train_pred[t][train_y[t] == 1]).item()
            if highest_ev == torch.max(train_pred[t]).item():
                this_train_acc += 1
                this_train_perc += 1
            else:
                beat_by = torch.sum(torch.where(train_pred[t] > highest_ev, 1, 0)).item()
                this_train_perc += 1 - (beat_by / train_pred[t].shape[0]-1)

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

            highest_ev = torch.max(val_pred[t][val_y[t] == 1]).item()
            if highest_ev == torch.max(val_pred[t]).item():
                this_val_acc += 1
                this_val_perc += 1
            else:
                beat_by = torch.sum(torch.where(val_pred[t] > highest_ev, 1, 0)).item()
                this_val_perc += 1 - (beat_by / val_pred[t].shape[0]-1)

        this_train_acc /= max(1, train_seen)
        this_train_perc /= max(1, train_seen)

        this_val_acc /= max(1, val_seen)
        this_val_perc /= max(1, val_seen)

        self.train_accs.append(this_train_acc)
        self.train_percs.append(this_train_perc)

        self.val_accs.append(this_val_acc)
        self.val_percs.append(this_val_perc)

        with open(LOG, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.train_accs)-1, this_train_perc, this_val_perc, this_train_acc, this_val_acc])

        fig, ax = plt.subplots(2)

        ax[0].plot(self.val_percs)
        ax[0].plot(self.train_percs)
        ax[0].set_title("Percentile of Highest Evidence")
        ax[0].legend(["val_perc", "train_perc"])

        ax[1].plot(self.val_accs)
        ax[1].plot(self.train_accs)
        ax[1].set_title(r"% Evidence in Rank=1 Prediction")
        ax[1].legend(["val_acc", "train_acc"])

        plt.savefig(GRAFF)
        plt.clf()

        if this_val_perc > self.best_val_acc:
            self.best_val_acc = this_val_perc
            torch.save(self.model.state_dict(), CHECKPOINT+"-{}.pt".format(len(self.val_percs)-1))


def main():

    train_data = AgentDataset(TRAIN_FILE, TRAIN_ENCODINGS, N_FRENS, NOISE_DECAY, device=torch.device("cuda"))
    val_data = AgentDataset(VAL_FILE, VAL_ENCODINGS, N_FRENS, NOISE_DECAY, device=torch.device("cuda"))

    k_loss = TopKCrossEntropy(TOP_K)

    logger = AgentLogger()
    model = Agent()
    model.L_qF.requires_grad = True
    model = model.cuda()

    optimizer = torch.optim.AdamW(params=model.L_qF.parameters(), lr=LR)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=50000,
    )

    train(model, optimizer, train_data, k_loss, val_data=val_data, batch_size=BATCH_SIZE, logger=logger, lr_scheduler=lr_scheduler)


if __name__== '__main__':
    main()
