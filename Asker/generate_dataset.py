import json
import random
import math
import os
import torch
from tqdm import tqdm
import sys

sys.path.append("../Agent")
from searcher import Searcher
from agent import Agent
from environment import Environment

OUTFILE = "generated_data/generated_training_data.json"

#checkpoint locations to load pretrained models
AGENT_CHECK = "../../checkpoints/agent-pre"
SEARCH_CHECK = "../../checkpoints/searcher-p"


# path to json with training data in it
TRAIN_FILE = "../../local_data/hotpot_data/train.json"
# path to encodings of training corpuses
TRAIN_ENCODINGS = "../../local_data/corpus_encodings/train.pt"

# path to file with val data in it
VAL_FILE = "../../local_data/hotpot_data/val.json"
# path to encodings of training corpuses
VAL_ENCODINGS = "../../local_data/corpus_encodings/val.pt"

# number of actions to choose from, including submit
N_ACTIONS = 8

# evey epoch, add 1/SKIP episodes to the replay buffer
SKIP = 300


# truncate the validation data to this many examples
VAL_TRUNC = 500

# before the first epoch, fill the replay buffer with this many examples
MIN_BUF = 1000
# discard the oldest examples once the replay buffer eaches this size
MAX_BUF = 10000

# reduce training epoch size for debugging
TRAIN_SKIP = 1

# device to run training on
DEVICE = torch.device("cuda")


def main():
    # load search model
    search = Searcher(load=SEARCH_CHECK)
    search = search.to(DEVICE)

    # load agent model
    model = Agent(load=AGENT_CHECK)
    model = model.to(DEVICE)

    t_env = Environment(TRAIN_FILE, TRAIN_ENCODINGS, search, model, N_ACTIONS, device=torch.device(DEVICE), skip=SKIP, data_start=TRAIN_START, max_buf=MAX_BUF, min_buf=MIN_BUF)
    
    t_data = t_env.data
    t_corpus = t_env.corpus
    data_list = []
    with torch.no_grad():
        with tqdm(0, len(t_data)) as p:
            for i in p:
                question = t_data[i]["question"]
                chosen = t_env.greedyRollout(i, "", t_data[i]["raw_corpus"], t_corpus.corpus[i].float())
                data_list.append({"question": question, "chosen": '<sep>'.join(chosen)})
                break
    
    with open(OUTFILE, 'w') as f:
        json.dump(data_list, f)


if __name__== '__main__':
    main()
