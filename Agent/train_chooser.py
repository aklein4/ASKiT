
import torch
from transformers import get_cosine_schedule_with_warmup

from searcher import Searcher
from chooser import Chooser
from train_searcher import SearchLogger, MaxPLoss

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
TRAIN_ENCODINGS = "../local_data/corpus_encodings/train.pt"

VAL_FILE = "../local_data/hotpot_data/val.json"
VAL_ENCODINGS = "../local_data/corpus_encodings/val.pt"

CHECKPOINT = "./checkpoints/chooser-drop"
LOG = "./logs/chooser-drop.csv"
GRAFF = "./logs/chooser-drop.png"

LR = 1e-6
BATCH_SIZE = 4

N_FRENS = 1
NOISE_DECAY = 2
TOP_K = 10

SKIP = 1
TRUNC = 20000

MEM_THRESH = 0.8


class ChooseDataset:

    def __init__(self, file, corpus_encodings, searcher, top_k, n_frens=None, noise_decay=None, device=torch.device("cpu")):

        self.device = device
        self.searcher = searcher
        self.top_k = top_k

        # json of all this data
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)
        self.data = self.data[:TRUNC]
        
        for p in self.data:
            p["raw_corpus"] = []
            for c in range(len(p["corpus"])):
                title = " "+ p["corpus_titles"][c] + ": "
                for s in p["corpus"][c]:
                    p["raw_corpus"].append(title + s)

        # load all of the embeddings
        self.corpus = torch.load(corpus_encodings)
        self.corpus = self.corpus[:TRUNC]
        for i in range(len(self.corpus)):
            self.corpus[i] = self.corpus[i].to(self.device)
            self.corpus[i].requires_grad = False
            assert self.corpus[i].shape[0] == len(self.data[i]["raw_corpus"])

        self.size = len(self.data)
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
                    k = random.randrange(n_evidence+1)
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

        self.x = []
        self.y = []

        for i in tqdm(range(len(self)), desc="Loading", leave=False):
            state = self.states[i]

            target = self.targets[i].clone()
            target[self.temp_evidence[i]] = 0
            target = [target]

            x_corpus = [self.corpus[i]]
            raw_corpus = self.data[i]["raw_corpus"].copy()

            for c in range(self.n_frens):
                fren_ind = self.frens[(i+c) % self.size]

                fren = self.corpus[fren_ind]    
                x_corpus.append(fren)

                raw_corpus += self.data[fren_ind]["raw_corpus"]

                target.append(torch.zeros([fren.shape[0]], dtype=torch.float32, device=self.device))

            assert len(x_corpus) == len(target)
            if sum(c.shape[0] for c in x_corpus) != len(raw_corpus):
                raise ValueError("x_corpus size ({}) != raw_corpus size ({}) at index {}".format(sum(c.shape[0] for c in x_corpus), len(raw_corpus), i))

            actions = []

            top_inds = None
            with torch.no_grad():
                x_corpus = torch.cat(x_corpus)
                target = torch.cat(target)

                scores = self.searcher(([state], [x_corpus.to(torch.float32)]))
                _, top_inds = torch.topk(scores[0], self.top_k)

            target = target[top_inds]
            target = torch.cat([target, torch.zeros([1,], dtype=target.dtype, device=target.device)])
            if torch.sum(target).item() == 0:
                target[-1] = 1

            for ind in range(top_inds.shape[0]):
                actions.append(raw_corpus[top_inds[ind]])
            actions.append(" .") # TODO: should this be "" or "<something>"

            self.x.append((state, actions))
            self.y.append(target)


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

        x_states = []
        x_actions = []
        y = []
        for ind in indices:
            state, actions = self.x[ind]
            x_states.append([state] * len(actions))
            x_actions.append(actions)

            y.append(self.y[ind].cuda())

        return (x_states, x_actions), y


class ChooseLogger(SearchLogger):
    def __init__(self, log_loc=LOG, graff=GRAFF):
        super().__init__(log_loc=log_loc, graff=graff)


    def initialize(self, model: Chooser):
        self.tokenizer = model.tokenizer
        self.encoder = model.encoder
        self.head = model.head
    

    def save_checkpoint(self):
        folder = CHECKPOINT + "_{}".format(len(self.val_percs)-1)
        os.makedirs(folder, exist_ok=True)

        self.tokenizer.save_pretrained(os.path.join(folder, "tokenizer"))
        self.encoder.save_pretrained(os.path.join(folder, "encoder"))
        torch.save(self.head.state_dict(), os.path.join(folder, "head.pt"))


class PMetric:
    def __init__(self):
        self.title = 'p'
    
    def __call__(self, pred, target):
        assert len(pred) == len(target)

        pred_stack = []
        target_stack = []

        b_size = min([p.numel() for p in pred])

        for i in range(len(pred)):
            vals, inds = torch.topk(pred[i], b_size)
            pred_stack.append(pred[i][inds])
            target_stack.append(target[i][inds])

        pred_batch = torch.stack(pred_stack)
        target_batch = torch.stack(target_stack)

        log_p = torch.nn.functional.softmax(pred_batch, dim=-1)

        loss = torch.sum(torch.where(target_batch == 1, log_p, torch.zeros_like(log_p)))
        return loss.item() / len(pred)


def main():

    searcher = Searcher()
    searcher = searcher.cuda()
    searcher.eval()

    train_data = ChooseDataset(TRAIN_FILE, TRAIN_ENCODINGS, searcher, TOP_K, N_FRENS, NOISE_DECAY, device=torch.device("cuda"))
    val_data = ChooseDataset(VAL_FILE, VAL_ENCODINGS, searcher, TOP_K, N_FRENS, NOISE_DECAY, device=torch.device("cuda"))

    # k_loss = TopKCrossEntropy(TOP_K)
    loss_fn = MaxPLoss

    logger = ChooseLogger(log_loc=LOG, graff=GRAFF)
    met = PMetric()

    model = Chooser()
    model = model.cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=15000,
        num_training_steps=80000,
    )

    train(model, optimizer, train_data, loss_fn, val_data=val_data, batch_size=BATCH_SIZE, logger=logger, lr_scheduler=lr_scheduler, skip=SKIP, rolling_avg=0.99, metric=met)


if __name__== '__main__':
    main()
