
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
SAMPLES_PER = 10
NUM_Q = 100


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


def main():

    torch.no_grad()

    # # load semantic search model
    # search = Searcher(load=SEARCH_FILE)
    # search = search.to(DEVICE)

    # # load agent model
    # model = Agent(load=AGENT_FILE)
    # model = model.to(DEVICE)

    data = None
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    for d in data:
        p = getDataPoint(d)
        print(p)
        exit()


if __name__ == "__main__":
    main()
