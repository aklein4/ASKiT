
import torch

from searcher import Searcher

import json
from tqdm import tqdm

IN_FILE = "../local_data/hotpot_data/val.json"
OUT_FILE = "../local_data/data/val.pt"

BATCH_SIZE = 128

def main():

    torch.no_grad()

    data = None
    with open(IN_FILE, 'r') as f:
        data = json.load(f)

    model = Searcher()
    model = model.cuda()

    # {question, answer, corpus, encodings, corpus_titles, evidence_labels, gold_inds}
    new_data = []

    for p in tqdm(data):

        out = {}

        out["question"] = p["question"]
        out["answer"] = p["answer"]

        out["corpus"] = []
        out["corpus_titles"] = []
        out["encodings"] = []

        out["evidence_labels"] = []
        out["gold_inds"] = []

        for i in range(len(p["corpus"])):
            sub = p["corpus"][i]
            name = p["corpus_titles"][i]
            for s in sub:
                out["corpus"].append("From the article '{}': ".format(name) + s)
                out["corpus_titles"].append(name)

                label = 1 if s in p["evidence_sentences"] else 0
                out["evidence_labels"].append(label)
                if label == 1:
                    out["gold_inds"].append(len(out["corpus"])-1)
        
        for i in range(0, len(out["corpus"]), BATCH_SIZE):   
            enc = model.encode(out["corpus"][i:i+BATCH_SIZE])

            out["encodings"].append(enc.cpu())

        out["encodings"] = torch.cat(out["encodings"], dim=0)                
        out["evidence_labels"] = torch.tensor(out["evidence_labels"])

        new_data.append(out)

    torch.save(new_data, OUT_FILE)

if __name__ == '__main__':
    main()
