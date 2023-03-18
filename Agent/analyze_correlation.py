
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
from collections import Counter


SEARCH_FILE = "checkpoints/searcher-p"
AGENT_FILE = "checkpoints/onehot_ppo_56"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_FILE = "../local_data/hotpot_data/val.json"
ENCODINGS_FILE = "../local_data/corpus_encodings/val.pt"

N_ACTIONS = 8
SAMPLES_PER = 10
NUM_Q = 1000


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

    score_ranks = []
    prob_ranks = []

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

            sorted_f1s = sorted(f1s, reverse=True)
            sorted_probs = sorted(log_probs, reverse=True)
            for f1 in f1s:
                score_ranks.append(sorted_f1s.index(f1) + 1)
                sorted_f1s[sorted_f1s.index(f1)] = None
            for prob in log_probs:
                prob_ranks.append(sorted_probs.index(prob) + 1)
                sorted_probs[sorted_probs.index(prob)] = None

            pbar.set_postfix({"avg": tot_corr/num_sampled, "overall": overall_corr})

            if (num_sampled+1) % 100 == 0:
                plt.scatter(overall_f1s, normed_probs)
                plt.savefig("./logs/correlation.png")

                x = score_ranks
                y = prob_ranks

                # count the occurrences of each point
                c = Counter(zip(x,y))
                # create a list of the sizes, here multiplied by 10 for scale
                s = [1000*c[(xx,yy)]/len(x) for xx,yy in zip(x,y)]

                plt.clf()
                plt.scatter(score_ranks, prob_ranks, s=s)
                plt.savefig("./logs/rank_correlation.png")
                plt.clf()

                torch.save((score_ranks, prob_ranks), "./logs/rank_data.pt")
    
    plt.scatter(overall_f1s, normed_probs)
    plt.savefig("./logs/correlation.png")

    x = score_ranks
    y = prob_ranks

    # count the occurrences of each point
    c = Counter(zip(x,y))
    # create a list of the sizes, here multiplied by 10 for scale
    s = [1000*c[(xx,yy)]/len(x) for xx,yy in zip(x,y)]

    plt.clf()
    plt.scatter(score_ranks, prob_ranks, s=s)
    plt.savefig("./logs/rank_correlation.png")

    torch.save((score_ranks, prob_ranks), "./logs/rank_data.pt")


if __name__ == "__main__":
    main()
