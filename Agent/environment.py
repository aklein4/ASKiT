
import torch

import json
import random
import numpy as np

import sys
sys.path.append("../utils")
from train_utils import get_mem_use
from tqdm import tqdm


MAX_DEPTH = 5

MEM_THRESH = 0.85


class Environment:

    def __init__(self, file, corpus_encodings, search, agent, top_k, device=torch.device("cpu"), skip=1, data_start=0, data_end=10000000, max_buf=100000):

        self.top_k = top_k
        self.device = device
        self.skip = skip
        self.max_buf = max_buf

        self.search = search
        self.agent = agent

        # json of all this data
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)

        # reduce data size
        self.data = self.data[data_start:data_end]

        # how big is it?
        self.size = len(self.data)

        # generate the raw corpus for each question
        for p in self.data:
            p["raw_corpus"] = []
            for c in range(len(p["corpus"])):
                title = " "+ p["corpus_titles"][c] + ", "
                for s in p["corpus"][c]:
                    p["raw_corpus"].append(title + s)

        # load all of the embeddings
        self.corpus = torch.load(corpus_encodings)
        for i in range(len(self.corpus)):
            self.corpus[i] = self.corpus[i].to(self.device)
            self.corpus[i].requires_grad = False

        # reduce corpus size
        self.corpus = self.corpus[data_start:data_end]

        # check that things match up
        assert len(self.corpus) == self.size

        # hold all of the data that we want to train on
        self.replay_buffer = []

        self.shuffler = []
        self.item_shuffler = []

        self.reset()


    def reset(self):
        self.shuffler = list(range(self.size))
        self.item_shuffler = list(range(len(self)))

    def shuffle(self):
        random.shuffle(self.shuffler)
        self.fillBuffer()
        self.item_shuffler = list(range(len(self)))
        random.shuffle(self.item_shuffler)


    def __len__(self):
        return len(self.replay_buffer)
    

    def __getitem__(self, getter):

        # clear cache if memory is getting full
        if get_mem_use() >= MEM_THRESH:
            torch.cuda.empty_cache()
        
        # unpack index and batchsize
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # get the indices we are going to use
        indices = self.item_shuffler[index:index+batchsize]

        x = ([], [], [])
        y = ([], [])
        
        # unpack data tuples onto batch tuples
        for i in indices:
            q, e, a, p, A = self.replay_buffer[i]
            
            x[0].append(q)
            x[1].append(e)
            x[2].append(a)
            
            y[0].append(p)
            y[1].append(A)

        return x, (torch.stack(y[0]), torch.stack(y[1]))


    def evaluate(self):
        self.reset()
        self.search.eval()
        self.agent.eval()

        f1s = 0
        correct = 0
        num_seen = 0
        
        with torch.no_grad():
            with tqdm(range(0, self.size, 1+round(self.step/10)), leave=False, desc="Evaluating") as pbar:
                for i in pbar:
                    chosen = self.greedyRollout(i, "", self.data[i]["raw_corpus"], self.corpus[i].float())
                    
                    f1s += self.getF1(i, chosen)
                    correct += self.getCorrect(i, chosen)
                    num_seen += 1

                    pbar.set_postfix({'acc': correct/num_seen, 'f1': f1s/num_seen})

        return f1s / num_seen, correct / num_seen


    def getF1(self, q_id, chosen):

        p = self.data[q_id]

        gold = p["raw_corpus"]

        correct = 0
        for c in chosen:
            if c in gold:
                correct += 1

        precision = correct / max(len(chosen), 1)
        recall = correct / max(1, len(gold))

        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        return f1


    def getCorrect(self, q_id, chosen):
        return self.getF1(q_id, chosen) == 1


    def fillBuffer(self):
        
        self.search.eval()
        self.agent.eval()

        self.replay_buffer = self.replay_buffer[:self.max_buf]

        with torch.no_grad():
            for i in tqdm(range(0, self.size, self.skip), leave=False, desc="Exploring"):
                q_ind = self.shuffler[i]
                p = self.data[q_ind]

                question = p["question"]
                evidence = ""

                avail_text = p["raw_corpus"].copy()
                avail_encodings = self.corpus[q_ind].float()

                chosen = []

                while True:
                
                    """ Get the available actions """

                    scores = self.search.forward(([question + evidence], [avail_encodings]))[0]
                    _, top_inds = torch.topk(scores, min(scores.numel(), self.top_k))

                    action_set = []
                    for i in range(top_inds.shape[0]):
                        action_set += [avail_text[top_inds[i]]]

                    """ Get the rewards for each action """

                    rewards = [self.getF1(q_ind, chosen)]
                    
                    for i in range(top_inds.shape[0]):

                        act_ind = top_inds[i].item()

                        temp_chosen = chosen + [avail_text[act_ind]]
                        temp_evidence = evidence + avail_text[act_ind]

                        temp_avail_text = avail_text.copy()
                        temp_avail_text.pop(act_ind)

                        temp_avail_encodings = torch.cat([avail_encodings[:act_ind], avail_encodings[act_ind+1:]])   

                        r = self.getF1(q_ind, temp_chosen + self.greedyRollout(q_ind, temp_evidence, temp_avail_text, temp_avail_encodings))
                        rewards.append(r)

                    rewards = torch.tensor(rewards).to(self.device).float()

                    """ Calculate the advantage and save the data """

                    policy = self.agent.forward(([question], [evidence], [action_set]))[0]     
                    policy = torch.nn.functional.softmax(policy, dim=-1)

                    V_s = torch.sum(policy * rewards).item()
                    advantage = rewards - V_s

                    self.replay_buffer.append((question, evidence, action_set, policy.detach(), advantage.detach()))

                    """ Sample a random trajectory """

                    action = np.random.choice(np.arange(policy.numel()), p=policy.detach().cpu().numpy())

                    if action == 0 or len(chosen) == MAX_DEPTH:
                        break

                    else:
                        act_ind = top_inds[action-1].item()

                        chosen.append(avail_text[act_ind])

                        evidence += avail_text.pop(act_ind)
                        avail_encodings = torch.cat([avail_encodings[:act_ind], avail_encodings[act_ind+1:]])


    def greedyRollout(self, question_id, evidence, avail_text, avail_encodings):

        p = question = self.data[question_id]["question"]

        avail_text = avail_text.copy()
        chosen = []

        with torch.no_grad():
            while True:
                
                scores = self.search.forward(([question + evidence], [avail_encodings]))[0]
                _, top_inds = torch.topk(scores, min(scores.numel(), self.top_k))

                eval_actions = []
                for i in range(top_inds.shape[0]):
                    eval_actions += [avail_text[top_inds[i]]]

                policy = self.agent.forward(([question], [evidence], [eval_actions]))[0]

                action = torch.argmax(policy).item()

                if action == 0 or len(chosen) == MAX_DEPTH:
                    break

                else:
                    act_ind = top_inds[action-1].item()

                    chosen.append(avail_text[act_ind])

                    evidence += avail_text.pop(act_ind)
                    avail_encodings = torch.cat([avail_encodings[:act_ind], avail_encodings[act_ind+1:]])

        return chosen
