
from agent import Agent

import torch
import json
from tqdm import tqdm
import random

TEST_FILE = "../local_data/hotpot_data/val.json"
ENCODINGS = "../local_data/corpus_encodings/val.pt"

K_TOP = 16

def main():
    
    data = None
    with open(TEST_FILE, 'r') as f:
        data = json.load(f)

    model = Agent()
    # model = model.cuda()
    
    encodings = torch.load(ENCODINGS, map_location=torch.device("cpu"))

    tot_misses = 0
    tot_bests = 0
    tot_correct = 0
    tot_perc = 0
    num_seen = 0
    tot_size = 0
    tot_top_5 = 0

    ind = -1
    for p in (pbar := tqdm(data)):
        ind += 1
        
        corpse = [encodings[ind].to(torch.float32)]
        for i in range(0):
            corpse.append(random.choice(encodings).to(torch.float32))
        corpse = torch.cat(corpse)

        corpus_scores = model.forward(([p["question"]], [corpse]))[0]
        tot_size += corpse.shape[0]

        # ranks = []
        # for s in p["evidence_raw_ids"]:
        #     ranks.append(torch.sum(torch.where(corpus_scores > corpus_scores[s], 1, 0)).item() + 1)
        
        # misses = (sum(ranks) - sum(range(1, len(ranks)+1))) / len(ranks)
        # tot_misses += misses

        tot_perc += 1 - (torch.sum(torch.where(corpus_scores > torch.max(corpus_scores[p["evidence_raw_ids"]]), 1, 0)) / (corpus_scores.shape[0] - 1)).item()
        tot_bests += torch.sum(torch.where(corpus_scores > torch.max(corpus_scores[p["evidence_raw_ids"]]), 1, 0)).item() + 1
        tot_correct += 1 if torch.max(corpus_scores) == torch.max(corpus_scores[p["evidence_raw_ids"]]).item() else 0
        tot_top_5 += 1 if torch.sum(torch.where(corpus_scores > torch.max(corpus_scores[p["evidence_raw_ids"]]), 1, 0)).item() < K_TOP else 0
        num_seen += 1
        pbar.set_postfix({"avg_size": tot_size/num_seen, "avg_perc": tot_perc/num_seen, "avg_best": tot_bests/num_seen, "acc": tot_correct/num_seen, "top_{}".format(K_TOP): tot_top_5/num_seen})

        # if min(ranks) == 1:
        #     print('\n', model.q)
        #     print(evidence[ranks.index(min(ranks))])
        #     input(">>>")
            

if __name__ == '__main__':
    main()