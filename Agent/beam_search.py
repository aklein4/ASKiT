
import torch
import torch.nn as nn
import torch.nn.functional as F

from searcher import Searcher
from agent import Agent

from tqdm import tqdm
import numpy as np
import json
import random


SEARCH_FILE = "checkpoints/searcher-p"
AGENT_FILE = "checkpoints/onehot_ppo_56"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_FILE = "../local_data/hotpot/hotpot_dev_distractor_v1.json"

N_ACTIONS = 8
MAX_DEPTH = 10
BEAM_SIZE = 5


def calcMetrics(pred, gold):

    # compare pred vs gold
    correct = 0
    for c in pred:
        if c in gold:
            correct += 1

    # calc stats
    precision = correct / max(len(pred), 1)
    recall = correct / max(1, len(gold))

    # turn into f1
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return int(f1==1.0), f1, precision, recall


def getDataPoint(d):
    p = {}
    p["question"] = d["question"]

    text_dict = {}
    encode_dict = {}
    for c in d["context"]:
        text_dict[c[0]] = ["{}, {}".format(c[0], c[1][i]) for i in range(len(c[1]))]
        encode_dict[c[0]] = ["{}: {}".format(c[0], c[1][i]) for i in range(len(c[1]))]

    p["evidence"] = [text_dict[e[0]][e[1]] for e in d["supporting_facts"]]

    p["text_corpus"] = []
    p["encode_corpus"] = []
    for k in text_dict.keys():
        p["text_corpus"] += text_dict[k]
        p["encode_corpus"] += encode_dict[k]

    return p


class ASKiT:
    def __init__(self):
        self.searcher = Searcher(load=SEARCH_FILE)
        self.searcher = self.searcher.to(DEVICE)
        self.searcher.eval()

        self.agent = Agent(load=AGENT_FILE)
        self.agent = self.agent.to(DEVICE)
        self.agent.eval()


    def getEvidence(self, question, text_corpus, encode_corpus, n_samples=1):
        encodings = self.searcher.encode(encode_corpus)

        best_chosen, best_log_prob = self.rollout(question, text_corpus, encodings, sample=False)
        
        for i in range(max(0, n_samples-1)):
            chosen, log_prob = self.rollout(question, text_corpus, encodings, sample=True)
            if log_prob > best_log_prob:
                best_chosen = chosen
                best_log_prob = log_prob
        
        return best_chosen


    def rollout(self, question, text_corpus, encodings, sample=False):

        avail_text = text_corpus.copy()
        avail_encodings = encodings.clone() 

        # fill chossen with only the evidence that this function chooses
        evidence = ""
        chosen = []

        log_prob = 0

        while True:
        
            if len(avail_text) < N_ACTIONS-1 or len(chosen) >= MAX_DEPTH:
                break

            # use search to get the top k actions
            scores = self.searcher.forward(([question + evidence], [avail_encodings]))[0]
            _, top_inds = torch.topk(scores, N_ACTIONS-1)

            # convert from indices to strings
            action_set = [None] # actions as strings
            action_inds = [None] # actions as indices
            for i in range(top_inds.shape[0]):
                action_inds += [top_inds[i]]
                action_set += [avail_text[top_inds[i]]]

            # use the agent to get the policy scores
            policy = self.agent.forward(([question], [evidence], [action_set[1:]]))[0]
            assert policy.numel() == len(action_set) and policy.numel() == len(action_inds)
            policy_dist = torch.distributions.Categorical(probs=torch.softmax(policy, dim=-1))

            action = torch.argmax(policy)
            if sample:
                action = policy_dist.sample()
            log_prob += policy_dist.log_prob(action)
            action = action.item()

            if action not in list(range(len(action_set))):
                print("Invalid action chosen: {} (only {} actions available)".format(action, len(action_set)))
                action = 0

            # stop if we submit, reach max depth, or run out of evidence needed for full stack
            if action == 0:
                break

            # continue sampling rollout
            else:

                # add the chosen action to the running evidence and chosen
                new_ev_str = action_set[action]
                chosen.append(new_ev_str)
                evidence += new_ev_str
            
                # remove the chosen action from the available evidence
                avail_text.pop(action_inds[action])    
                avail_encodings = torch.cat([avail_encodings[:action_inds[action]], avail_encodings[action_inds[action]+1:]])
                assert len(avail_text) == avail_encodings.shape[0]
        
        return chosen, log_prob


def main():

    torch.no_grad()

    askit = ASKiT()

    data = None
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    random.shuffle(data)

    tot_corr, tot_f1, tot_prec, tot_rec = 0, 0, 0, 0
    num_samples = 0

    pbar = tqdm(data)
    for d in pbar:
        p = getDataPoint(d)
        
        pred = askit.getEvidence(p["question"], p["text_corpus"], p["encode_corpus"], BEAM_SIZE)
        gold = p["evidence"]

        corr, f1, prec, rec = calcMetrics(pred, gold)
        tot_corr += corr
        tot_f1 += f1
        tot_prec += prec
        tot_rec += rec
        num_samples += 1

        pbar.set_postfix({"acc": tot_corr/num_samples, "f1": tot_f1/num_samples, "prec": tot_prec/num_samples, "rec": tot_rec/num_samples})
    pbar.close()

    print("\nFinal results:\nAccuracy: {}\nF1: {}\nPrecision: {}\nRecall: {}".format(tot_corr/num_samples, tot_f1/num_samples, tot_prec/num_samples, tot_rec/num_samples))

if __name__ == "__main__":
    main()
