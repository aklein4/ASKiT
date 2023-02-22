
import json
import random
import math
import os
from tqdm import tqdm

# input json
INPUT = "../local_data/hotpot/hotpot_train_v1.1.json"

# output dir to put stuff in
OUTPUT_DIR = "../local_data/hotpot/data/"

VAL_PERC = 0.1

def main():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    temp_data = None
    with open(INPUT, 'r') as f:
        temp_data = json.load(f)

    data = []
    for p in tqdm(temp_data):
        if p["answer"] in ["yes", "no", "noanswer"]:
            continue

        fixed = {
            "question": p["question"],
            "answer": p["answer"],
        }

        try:
            evidence_sentences = []
            for s in p['supporting_facts']:
                c_ind = [c[0] for c in p['context']].index(s[0])
                evidence_sentences.append(p['context'][c_ind][1][s[1]])
            fixed["evidence_sentences"] = evidence_sentences
        except:
            continue

        fixed["evidence_titles"] = [s[0] for s in p["supporting_facts"]]
        fixed["evidence_ids"] = [s[1] for s in p["supporting_facts"]]

        fixed["corpus"] = [c[1] for c in p['context']]
        fixed["corpus_titles"] = [c[0] for c in p['context']]

        data.append(fixed)

    random.shuffle(data)

    val_num = math.ceil(len(data) * VAL_PERC)
    
    val_data = data[:val_num]
    print("Validation Size:", len(val_data))
    with open(OUTPUT_DIR+"val.json", 'w') as f:
        json.dump(val_data, f, indent=4)
    
    train_data = data[val_num:]
    print("Train Size:", len(train_data))
    with open(OUTPUT_DIR+"train.json", 'w') as f:
        json.dump(train_data, f, indent=4)

    print("Done.")

if __name__ == '__main__':
    main()