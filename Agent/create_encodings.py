
import torch

from searcher import Searcher

import json
from tqdm import tqdm

IN_FILE = "../local_data/hotpot_data/train.json"
OUT_FILE = "../local_data/corpus_encodings/train.pt"

BATCH_SIZE = 128

def main():

    torch.no_grad()

    data = None
    with open(IN_FILE, 'r') as f:
        data = json.load(f)

    model = Searcher()
    model = model.cuda()
    
    encodings = []
    mem_use = 0

    curr_text = []
    sizes = []

    with tqdm(data) as pbar:
        for p in pbar:

            text_corpus = []
            for i in range(len(p["corpus"])):
                sub = p["corpus"][i]
                name = p["corpus_titles"][i]
                for s in sub:
                    text_corpus.append(name + ": " + s)
            curr_text += text_corpus

            sizes.append(len(text_corpus))
            
            if len(sizes) < BATCH_SIZE and len(encodings) + len(sizes) < len(data):
                continue

            enc_p = model.encode(curr_text).to(torch.float16)
            
            mem_use += enc_p.numel() * enc_p.element_size()
            
            for e in torch.split(enc_p, sizes):
                encodings.append(e)

            sizes = []
            curr_text = []

            pbar.set_postfix({"Memory (GB)": round(mem_use*1e-9, 3)})

    torch.save(encodings, OUT_FILE)

    print("Total Memory Size:", round(mem_use*1e-9, 3))
    print("Example Shape:", tuple(encodings[0].shape))

if __name__ == '__main__':
    main()
