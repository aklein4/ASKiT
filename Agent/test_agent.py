
import torch

from searcher import Searcher
from chooser import Chooser

import json
from tqdm import tqdm
import random

TEST_FILE = "../local_data/hotpot_data/val.json"
ENCODINGS = "../local_data/corpus_encodings/val.pt"

CHOOSER_CHECK = "./checkpoints/chooser-nat"

K_TOP = 3
MAX_DEPTH = 7

def main():
    
    torch.no_grad()

    data = None
    with open(TEST_FILE, 'r') as f:
        data = json.load(f)

    for p in data:
        p["raw_corpus"] = []
        for c in range(len(p["corpus"])):
            title = " "+ p["corpus_titles"][c] + ": "
            for s in p["corpus"][c]:
                p["raw_corpus"].append(title + s)

    encodings = torch.load(ENCODINGS, map_location=torch.device("cpu"))

    search = Searcher()
    choose = Chooser(load=CHOOSER_CHECK)

    tot_correct = 0
    tp = 0
    fp = 0
    fn = 0

    num_seen = 0

    ind = -1
    for p in (pbar := tqdm(data)):
        ind += 1

        state = p["question"]
        avail_text = p["raw_corpus"].copy()
        avail_corpse = encodings[ind].float()
        chosen = []

        while True:
            
            scores = search.forward(([state], [avail_corpse]))[0]
            _, top_inds = torch.topk(scores, min(K_TOP, len(avail_text)))

            eval_states = [state] * (top_inds.shape[0] + 1)
            eval_actions = [" ."]
            for i in range(top_inds.shape[0]):
                eval_actions += [avail_text[top_inds[i]]]

            evals = choose.forward(([eval_states], [eval_actions]))[0]
            evals[0] = 0

            action = torch.argmax(evals).item()

            if action == 0 or len(chosen) == MAX_DEPTH:
                break
            else:
                act_ind = top_inds[action-1].item()

                state += avail_text.pop(act_ind)
                avail_corpse = torch.cat([avail_corpse[:act_ind], avail_corpse[act_ind+1:]])
                chosen.append(act_ind)

        corr = True

        true_ev = p["evidence_raw_ids"]
        for t in true_ev:
            if t in chosen:
                tp += 1
            else:
                corr = False
                fn += 1
        
        for c in chosen:
            if c not in true_ev:
                corr = False
                fp += 1

        if corr:
            tot_correct += 1
        num_seen += 1

        f1 = tp / (tp + (fp + fn)/2)        
        pbar.set_postfix({"acc": tot_correct/num_seen, "F1": f1})
    
    print("Exact %:", round(100*tot_correct/num_seen, 1))
    print("F1 Score:", round(100*tp / (tp + (fp + fn)/2), 1))

if __name__ == '__main__':
    main()