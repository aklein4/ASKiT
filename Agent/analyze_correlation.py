
import torch
import torch.nn as nn
import torch.nn.functional as F

from searcher import Searcher
from agent import Agent
from environment import Environment

from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


SEARCH_FILE = "checkpoints/searcher-p"
AGENT_FILE = "checkpoints/onehot_ppo_56"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_FILE = "../local_data/hotpot_data/val.json"
ENCODINGS_FILE = "../local_data/corpus_encodings/val.pt"

N_ACTIONS = 8
SAMPLES_PER = 10
NUM_Q = 100


def main():

    torch.no_grad()

    # load semantic search model
    search = Searcher(load=SEARCH_FILE)
    search = search.to(DEVICE)

    # load agent model
    model = Agent(load=AGENT_FILE)
    model = model.to(DEVICE)

    env = Environment(DATA_FILE, ENCODINGS_FILE, search, model, N_ACTIONS, device=torch.device(DEVICE), data_end=NUM_Q)

    tot_corr = 0
    num_sampled = 0

    overall_f1s = []
    normed_probs = []
    overall_corr = None

    with tqdm(range(env.size)) as pbar:
        for q_id in pbar:
            f1s, log_probs = [], []

            for _ in range(SAMPLES_PER):
                f1, log_prob, num_actions = env.sampleRollout(q_id)

                f1s.append(f1)
                log_probs.append(log_prob.item())

                overall_f1s.append(f1)
                normed_probs.append(log_prob.item()/num_actions)

            if (num_sampled+1) % 10 == 0:
                overall_corr = spearmanr(a=np.array(overall_f1s), b=np.array(normed_probs))[0]
        
            if len(set(f1s)) == 1 or len(set(log_probs)) == 1:
                continue

            num_sampled += 1
            tot_corr += spearmanr(a=np.array(f1s), b=np.array(log_probs))[0] 

            pbar.set_postfix({"avg": tot_corr/num_sampled, "overall": overall_corr})

    plt.scatter(overall_f1s, normed_probs)
    plt.savefig("./logs/correlation.png")
            



if __name__ == "__main__":
    main()
