import json
import random
import math
import os
import torch
from tqdm import tqdm
import sys

sys.path.append("../Agent/")
from searcher import Searcher
from agent import Agent
from environment import Environment

OUTFILE = "generated_data/generated_training_data.json"
SMALL_OUTFILE = "generated_data/generated_training_data_small.json"

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
    print("Loading models...")
    # load search model
    search = Searcher(load=SEARCH_CHECK)
    search = search.to(DEVICE)

    # load agent model
    model = Agent(load=AGENT_CHECK)
    model = model.to(DEVICE)
    print("Done.")

    print("Initializing Environment...")
    t_env = Environment(TRAIN_FILE, TRAIN_ENCODINGS, search, model, N_ACTIONS, device=torch.device(DEVICE), skip=SKIP, max_buf=MAX_BUF, min_buf=MIN_BUF)
    t_data = t_env.data
    t_corpus = t_env.corpus
    print("Done.")

    print("Beginning example generation...")
    data_list = []
    with torch.no_grad():
        #with tqdm(0, len(t_data)) as p:
        for i in range(len(t_data)):
            if i % 100 == 0: 
                print("Generated " + str(i) + " / " + str(len(t_data)) + " examples.")
            question = t_data[i]["question"]
            chosen = t_env.greedyRollout(i, "", t_data[i]["raw_corpus"], t_corpus[i].float())
            data_list.append({"question": question, "chosen": '<sep>'.join(chosen)})
            if i == 1000:
                with open(SMALL_OUTFILE, 'w') as f:
                    json.dump(data_list, f)
            elif i % 5000 == 0:
                print("Saving...")
                with open(OUTFILE, 'w') as f:
                    json.dump(data_list, f)
                print("Save complete. Resuming generation...")
    print("Done.")
    print("Writing final JSON...")
    with open(OUTFILE, 'w') as f:
        json.dump(data_list, f)
    print("Done.")
    print("Generation complete. Have a nice day!")

if __name__== '__main__':
    main()
