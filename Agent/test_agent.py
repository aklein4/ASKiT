
from agent import Agent

import torch
import json
from tqdm import tqdm
import statistics

TEST_FILE = "../local_data/hotpot/data/val.json"

USE_TITLE = False

def main():
    
    data = None
    with open(TEST_FILE, 'r') as f:
        data = json.load(f)

    model = Agent()
    model = model.cuda()
    
    tot_misses = 0
    tot_bests = 0
    tot_correct = 0
    tot_perc = 0
    num_seen = 0
    for p in (pbar := tqdm(data)):
        
        model.setQuestion(p["question"])

        text_corpus = []
        for i in range(len(p["corpus"])):
            sub = p["corpus"][i]
            if USE_TITLE:
                name = p["corpus_titles"][i]
                for s in sub:
                    text_corpus.append(name + ". " + s)
            else:
                text_corpus += sub
            
        evidence = []
        if USE_TITLE:
            for i in range(len(p["evidence_sentences"])):
                s = p["evidence_sentences"][i]
                name = p["evidence_titles"][i]
                evidence.append(name + ". " + s)
        else:
            evidence = p["evidence_sentences"]
        
        corpus_encoding = model.encode(text_corpus)
        corpus_scores = model._Q_b(corpus_encoding)
        
        #print("\ncorpus_scores:", [round(corpus_scores[i].item(), 3) for i in range(corpus_scores.shape[0])])
        
        evidence_encoding = model.encode(evidence)
        evidence_scores = model._Q_b(evidence_encoding)

        #print("\nevidence_scores:", [round(evidence_scores[i].item(), 3) for i in range(evidence_scores.shape[0])])

        ranks = []
        for i in range(evidence_scores.shape[0]):
            ranks.append(torch.sum(torch.where(corpus_scores > evidence_scores[i], 1, 0)).item() + 1)
        
        misses = (sum(ranks) - sum(range(1, len(ranks)+1))) / len(ranks)

        tot_misses += misses
        tot_perc += 1 - (misses / len(text_corpus))
        tot_bests += min(ranks)
        tot_correct += 1 if min(ranks) == 1 else 0
        num_seen += 1
        pbar.set_postfix({"avg_misses": tot_misses/num_seen, "avg_perc": tot_perc/num_seen, "avg_best": tot_bests/num_seen, "acc:": tot_correct/num_seen})

        if min(ranks) == 1:
            print('\n', model.q)
            print(evidence[ranks.index(min(ranks))])
            input(">>>")
            

if __name__ == '__main__':
    main()