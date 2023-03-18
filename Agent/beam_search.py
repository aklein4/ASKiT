
import torch
import torch.nn as nn
import torch.nn.functional as F

from searcher import Searcher
from agent import Agent

from tqdm import tqdm
import numpy as np
import json


SEARCH_FILE = "checkpoints/searcher-p"
AGENT_FILE = "checkpoints/onehot_ppo_56"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_FILE = "../local_data/hotpot/hotpot_dev_distractor_v1.json"

N_ACTIONS = 8
MAX_DEPTH = 8
SAMPLES_PER = 10


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
        self.search = Searcher(load=SEARCH_FILE)
        self.search = self.search.to(DEVICE)

        self.model = Agent(load=AGENT_FILE)
        self.model = self.model.to(DEVICE)


    def getEvidence(self, question, corpus, encodings):
        chosen = self.greedyRollout(question, corpus, encodings, self.model, N_ACTIONS, device=torch.device(DEVICE))
        return chosen


    def greedyRollout(self, question_id, evidence, avail_text, avail_encodings, start_depth=0):
        # finish a greedy rollout from the current state

        # get the question using the id
        question = self.data[question_id]["question"]

        # make sure that we don't modify the original text corpus
        avail_text = avail_text.copy()
        
        # fill chossen with only the evidence that this function chooses
        chosen = []

        # go until end
        with torch.no_grad():
            while True:
                
                
                # use search to get the top k actions
                scores = self.search.forward(([question + evidence], [avail_encodings]))[0]
                _, top_inds = torch.topk(scores, self.top_k-1)

                # convert from indices to strings
                action_set = [None] # actions as strings
                action_inds = [None] # actions as indices
                for i in range(top_inds.shape[0]):
                    action_inds += [top_inds[i]]
                    action_set += [avail_text[top_inds[i]]]

                # use the agent to get the policy scores
                policy = self.agent.forward(([question], [evidence], [action_set[1:]]), debug=True)[0]

                assert policy.numel() == len(action_set) and policy.numel() == len(action_inds)

                # choose action greedily
                action = torch.argmax(policy).item()
                    
                if action not in list(range(len(action_set))):
                    print("Invalid action chosen: {} (only {} actions available)".format(action, len(action_set)))
                    action = 0

                # stop if we submit, reach max depth, or run out of evidence needed for full stack
                if action == 0 or len(chosen)+start_depth == MAX_DEPTH or len(avail_text)-1 < self.top_k:
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

        
        # clear cache if memory is getting full
        if get_mem_use() >= MEM_THRESH:
            torch.cuda.empty_cache()

        # return the chosen evidence that was chosen during this rollout
        return chosen


def main():

    torch.no_grad()

    askit = ASKiT()

    data = None
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    for d in tqdm(data):
        p = getDataPoint(d)
        
        corpus = p["text_corpus"].copy()
        encodings = search.encode(p["encode_corpus"])

        chosen = greedyRollout(p["question"], corpus, encodings, model, N_ACTIONS, device=torch.device(DEVICE))

if __name__ == "__main__":
    main()
