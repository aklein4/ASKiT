
import json
import random

DATASET = "../local_data/hotpot_train_v1.1.json"

def main():
    
    data = None
    with open(DATASET, 'r') as f:
        data = json.load(f)

    random.shuffle(data)

    for p in data:
        question = p['question']
        level = p['level']
        asnwer = p['answer']

        evidence = []
        for s in p['supporting_facts']:
            c_ind = [c[0] for c in p['context']].index(s[0])
            evidence.append(p['context'][c_ind][1][s[1]])

        print("Question:", question)
        print("Level:", level)
        print("Answer:", asnwer)

        print("Evidence:")
        for e in evidence:
            print(" -", e)
        
        input("")

if __name__ == '__main__':
    main()