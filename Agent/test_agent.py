
import torch

from searcher import Searcher
from agent import Agent

import json
from tqdm import tqdm
import random

TEST_FILE = "../local_data/hotpot_data/val.json"
ENCODINGS = "../local_data/corpus_encodings/val.pt"

SEARCH_CHECK = "./checkpoints/searcher-p"
AGENT_CHECK = "./checkpoints/agent_0"

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
            title = " "+ p["corpus_titles"][c] + ", "
            for s in p["corpus"][c]:
                p["raw_corpus"].append(title + s)

    encodings = torch.load(ENCODINGS, map_location=torch.device("cpu"))

    search = Searcher(load=SEARCH_CHECK)
    agent = Agent(load=AGENT_CHECK)

    tot_correct = 0
    tot_f1 = 0
    num_seen = 0

    shuffler = list(range(len(data)))
    random.shuffle(shuffler)

    for data_ind in (pbar := tqdm(shuffler)):

        p = data[data_ind]

        question = p["question"]
        evidence = ""

        avail_text = p["raw_corpus"].copy()
        avail_corpse = encodings[data_ind].float()
        chosen = []

        while True:
            
            scores = search.forward(([question + evidence], [avail_corpse]))[0]
            _, top_inds = torch.topk(scores, min(K_TOP, len(avail_text)))

            eval_actions = []
            for i in range(top_inds.shape[0]):
                eval_actions += [avail_text[top_inds[i]]]

            policy = agent.forward(([question], [evidence], [eval_actions]))[0]

            action = torch.argmax(policy).item()

            if action == 0 or len(chosen) == MAX_DEPTH:
                break

            else:
                act_ind = top_inds[action-1].item()

                this_evidence = avail_text[act_ind]
                

                chosen.append(this_evidence)
                evidence += avail_text.pop(act_ind)

                avail_corpse = torch.cat([avail_corpse[:act_ind], avail_corpse[act_ind+1:]])

        print('\n', question)

        print("\nChosen Evidence:\n")
        for e in chosen:
            print('\n', e)

        print("\nTrue Evidence:\n")
        for t in p["evidence_raw_ids"]:
            print('\n', p["raw_corpus"][t])

        temp = []
        for c in chosen:
            temp.append(p["raw_corpus"].index(c))
        chosen = temp

        tp = 0
        fp = 0
        fn = 0

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

        f1 = tp / (tp + (fp + fn)/2)   
        tot_f1 += f1

        print("\n--- Correct:", corr, "F1:", round(100*f1, 1))
        input("...")

        num_seen += 1
        pbar.set_postfix({"acc": tot_correct/num_seen, "F1": tot_f1/num_seen})
    
    print("Exact %:", round(100*tot_correct/num_seen, 1))
    print("F1 Score:", round(100*tp / (tp + (fp + fn)/2), 1))

if __name__ == '__main__':
    main()