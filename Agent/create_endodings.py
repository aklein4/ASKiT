
import torch

from agent import Agent

import json
from tqdm import tqdm

IN_FILE = "../local_data/hotpot_data/train.json"
OUT_FILE = "../local_data/corpus_encodings/train.pt"


def main():

    data = None
    with open(IN_FILE, 'r') as f:
        data = json.load(f)

    model = Agent()

    encodings = []
    mem_use = 0
    for p in (pbar := tqdm(data)):

        text_corpus = []
        for i in range(len(p["corpus"])):
            sub = p["corpus"][i]
            name = p["corpus_titles"][i]
            for s in sub:
                text_corpus.append(name + ". " + s)

        enc_p = model.encode(text_corpus).to(torch.float16)
        encodings.append(enc_p)

        mem_use += enc_p.numel() * enc_p.element_size()

        pbar.set_postfix({"Memory (GB)": round(mem_use*1e-9, 3)})

    torch.save(encodings, OUT_FILE)

    print("Total Memory Size:", round(mem_use*1e-9, 3))
    print("Example Shape:", tuple(encodings.shape))

if __name__ == '__main__':
    main()