
import torch
import torch.nn as nn
import torch.nn.functional as F

from searcher import Searcher
from agent import Agent
from asker import Asker

import sys
sys.path.append("../utils")
from train_utils import get_mem_use

from tqdm import tqdm
import numpy as np
import json
import random


SEARCH_FILE = "checkpoints/searcher-p"
AGENT_FILE = "checkpoints/onehot_ppo_56"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_FILE = "../local_data/hotpot/hotpot_dev_distractor_v1.json"

N_ACTIONS = 6
MAX_DEPTH = 10
BEAM_SIZE = 1


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

        self.asker = Asker()
        self.asker = self.asker.to(DEVICE)
        self.asker.eval()


    def getEvidence(self, question, text_corpus, encode_corpus, n_samples=1):
        encodings = self.searcher.encode(encode_corpus)

        best_chosen, best_log_prob = self.rollout(question, text_corpus, encodings, sample=False)
        
        for i in range(max(0, n_samples-1)):
            chosen, log_prob = self.rollout(question, text_corpus, encodings, sample=True)
            if log_prob > best_log_prob:
                best_chosen = chosen
                best_log_prob = log_prob
        
        return best_chosen


    def rollout(self, question, text_corpus, encodings, sample=False, get_avail=False, evidence=""):

        avail_text = text_corpus.copy()
        avail_encodings = encodings.clone() 

        # fill chossen with only the evidence that this function chooses
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

        if get_avail:
            return chosen, avail_text, avail_encodings
        return chosen, log_prob


    def beamSearch(self, question, text_corpus, encoding_corpus, n_beams=BEAM_SIZE):
        torch.cuda.empty_cache()
        encodings = self.searcher.encode(encoding_corpus)

        questions = [question]*n_beams
        evidences = [""]*n_beams
        chosens = [[]]*n_beams
        dones = torch.tensor([False]*n_beams).to(DEVICE)
        log_probs = torch.zeros(n_beams).to(DEVICE)

        avail_texts = [text_corpus.copy() for _ in range(n_beams)]
        avail_encodings = [encodings.clone() for _ in range(n_beams)]

        while True:
        
            torch.cuda.empty_cache()

            if min(len(a) for a in avail_texts) < N_ACTIONS-1 or max([len(c) for c in chosens]) >= MAX_DEPTH:
                break

            # use search to get the top k actions
            scores = self.searcher.forward(([questions[i] + evidences[i] for i in range(n_beams)], avail_encodings))
            top_inds = []
            for b in range(n_beams):
                _, top = torch.topk(scores[b], N_ACTIONS-1)
                top_inds.append(top)

            # convert from indices to strings
            action_sets = [[None] for _ in range(n_beams)] # actions as strings
            action_indss = [[None] for _ in range(n_beams)] # actions as indices
            for b in range(n_beams):
                for i in range(N_ACTIONS-1):
                    action_indss[b] += [top_inds[b][i]]
                    action_sets[b] += [avail_texts[b][top_inds[b][i]]]

            # use the agent to get the policy scores
            policy = self.agent.forward((questions, evidences, [a[1:] for a in action_sets]))
            policy = torch.log_softmax(policy, dim=-1)
            policy[dones,0] = 0
            policy[dones, 1:] = float('-inf')

            ratings = policy + log_probs.unsqueeze(-1)
            
            R = torch.topk(ratings.view(-1), n_beams)[1]
            new_beams = [(R[i]//policy.shape[1], R[i]%policy.shape[1]) for i in range(n_beams)]

            temp_questions, temp_evidences, temp_chosens = [], [], []
            temp_avail_texts, temp_avail_encodings = [], []
            temp_dones, temp_log_probs = [], []

            for b in range(n_beams):
                i, j = new_beams[b]
                temp_questions += [questions[i]]
                
                if j == 0:
                    temp_evidences += [evidences[i]]
                    temp_chosens += [chosens[i]]
                    temp_avail_texts += [avail_texts[i].copy()]
                    temp_avail_encodings += [avail_encodings[i].clone()]
                    temp_dones += [True]
                    temp_log_probs += [ratings[i, j]]
                
                else:
                    temp_evidences += [evidences[i] + " " + action_sets[i][j]]
                    temp_chosens += [chosens[i] + [action_sets[i][j]]]
                    temp_avail_texts += [avail_texts[i].copy()]
                    temp_avail_texts[-1].pop(action_indss[i][j])
                    temp_avail_encodings += [torch.cat([avail_encodings[i][:action_indss[i][j]], avail_encodings[i][action_indss[i][j]+1:]])]
                    temp_dones += [False]
                    temp_log_probs += [ratings[i, j]]

            questions, evidences, chosens = temp_questions, temp_evidences, temp_chosens
            avail_texts, avail_encodings = temp_avail_texts, temp_avail_encodings
            dones, log_probs = torch.tensor(temp_dones).to(DEVICE), torch.tensor(temp_log_probs).to(DEVICE)

            if torch.all(dones):
                break
        
        best_answer = torch.argmax(log_probs)

        return chosens[best_answer]
    

    def recursiveSearch(self, question, text_corpus, encoding_corpus, detailed=False):
        encodings = self.searcher.encode(encoding_corpus)

        avail_text = text_corpus.copy()
        avail_encodings = encodings.clone() 

        # fill chossen with only the evidence that this function chooses
        evidence = ""
        chosen = []
 
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

        action = torch.argmax(policy).item()

        if action == 0:
            action = 1

        # add the chosen action to the running evidence and chosen
        new_ev_str = action_set[action]
        chosen.append(new_ev_str)
        evidence += new_ev_str
    
        # remove the chosen action from the available evidence
        avail_text.pop(action_inds[action])    
        avail_encodings = torch.cat([avail_encodings[:action_inds[action]], avail_encodings[action_inds[action]+1:]])
        assert len(avail_text) == avail_encodings.shape[0]

        sub_questions = []
        sub_chosens = []

        while True:
        
            if len(avail_text) < N_ACTIONS-1 or len(chosen) >= MAX_DEPTH:
                break

            # use search to get the top k actions
            scores = self.searcher.forward(([question + evidence], [avail_encodings]))[0]
            _, top_inds = torch.topk(scores, N_ACTIONS-1)

            # use the agent to get the policy scores
            policy = self.agent.forward(([question], [evidence], [[avail_text[top_inds[i]] for i in range(top_inds.shape[0])]]))[0]

            if torch.argmax(policy) == 0:
                break

            new_question = self.asker(question, evidence)[0]
            new_chosen, avail_text, avail_encodings = self.rollout(new_question, avail_text, avail_encodings, get_avail=True, evidence=evidence)
            
            evidence += " " + " ".join(new_chosen)
            chosen += new_chosen

            sub_questions.append(new_question)
            sub_chosens.append(new_chosen)

        if detailed:
            return chosen, sub_questions, sub_chosens
        return chosen


def main():

    torch.no_grad()

    askit = ASKiT()

    data = None
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    random.shuffle(data)

    tot_corr, tot_f1, tot_prec, tot_rec = 0, 0, 0, 0
    tot_subs = 0
    num_samples = 0

    #pbar = tqdm(data)
    for d in data:

        if get_mem_use() >= 0.8:
            torch.cuda.empty_cache()

        p = getDataPoint(d)
        
        chosen, sub_questions, sub_chosens = askit.recursiveSearch(p["question"], p["text_corpus"], p["encode_corpus"], detailed=True)
        tot_subs += len(sub_questions)

        print("\n--------------------\n")
        print("Question:\n - {}".format(p["question"]))
        print(" - {}".format(chosen[0]))
        for i in range(len(sub_questions)):
            print("    {}: {}".format(i, sub_questions[i]))
            for j in range(len(sub_chosens[i])):
                print("     - {}".format(sub_chosens[i][j]))
        input("\n...")

        # pred = chosen
        # pred = askit.beamSearch(p["question"], p["text_corpus"], p["encode_corpus"], BEAM_SIZE)
        # gold = p["evidence"]

        # corr, f1, prec, rec = calcMetrics(pred, gold)
        # tot_corr += corr
        # tot_f1 += f1
        # tot_prec += prec
        # tot_rec += rec
        # num_samples += 1

        # pbar.set_postfix({"acc": tot_corr/num_samples, "f1": tot_f1/num_samples, "prec": tot_prec/num_samples, "rec": tot_rec/num_samples, "subs": tot_subs/num_samples})
    pbar.close()

    print("\nFinal results:\nAccuracy: {}\nF1: {}\nPrecision: {}\nRecall: {}".format(tot_corr/num_samples, tot_f1/num_samples, tot_prec/num_samples, tot_rec/num_samples))

if __name__ == "__main__":
    main()
