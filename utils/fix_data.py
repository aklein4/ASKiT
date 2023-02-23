
import json
import random
import math
import os
from tqdm import tqdm


# input json
FILE = "../local_data/hotpot_data/train.json"


def main():

    data = None
    with open(FILE, 'r') as f:
        data = json.load(f)

    for i in tqdm(range(len(data))):
        p = data[i]

        text_corpus = []
        for i in range(len(p["corpus"])):
            sub = p["corpus"][i]
            for s in sub:
                text_corpus.append(s)
        
        evidence_ids = [text_corpus.index(e) for e in p["evidence_sentences"]]

        p["evidence_raw_ids"] = evidence_ids
    
    with open(FILE, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()