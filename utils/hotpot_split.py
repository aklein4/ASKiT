
import json
import random
import math

INPUT = "../local_data/hotpot/hotpot_train_v1.1.json"

OUTPUT_DIR = "../local_data/hotpot/data/"

VAL_PERC = 0.1

def main():
    
    data = None
    with open(INPUT, 'r') as f:
        data = json.load(f)

    random.shuffle(data)

    val_num = math.ceil(len(data) * VAL_PERC)
    
    val_data = data[:val_num]
    with open(OUTPUT_DIR+"val.json", 'w') as f:
        json.dump(val_data, f)
    
    train_data = data[val_num:]
    with open(OUTPUT_DIR+"train.json", 'w') as f:
        json.dump(train_data, f)

if __name__ == '__main__':
    main()