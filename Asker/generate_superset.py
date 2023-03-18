import json
import random
import math
import os
import torch
from tqdm import tqdm
import sys
from transformers import T5ForConditionalGeneration, AutoTokenizer

OUTFILE = "generated_data/generated_super_training_data.json"
SMALL_OUTFILE = "generated_data/generated_super_training_data_small.json"
DATA_PATH = "generated_data/generated_training_data.json"

from datasets import load_dataset, load_metric, list_metrics

ASKER_MODEL = "matthv/first_t5-end2end-questions-generation"
GENERATOR_ARGS = {
  "max_length": 128,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
}

# device to run training on
DEVICE = torch.device("cuda")


def main():
    #Initialize tokenizer and asker
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
    asker = T5ForConditionalGeneration.from_pretrained(ASKER_MODEL)
    asker = asker.to("cuda")

    # Load Data
    data = load_dataset("json", data_files=DATA_PATH, split='train[:2%]')

    # Consider '<sep>' token
    tokenizer.sep_token = '<sep>'
    tokenizer.add_tokens(['<sep>'])
    asker.resize_token_embeddings(len(tokenizer))

    print("Beginning example generation...")
    data_list = []
    with torch.no_grad():
        #with tqdm(0, len(t_data)) as p:
        for i in tqdm(range(len(data))):
            all_evidence = data[i]["chosen"].split('<sep>')
            for j in range(1, len(all_evidence) - 1):
                ques_ev = data[i]['question'] + ' '.join(all_evidence[:j]) + " </s>"
                to_invert = "generate question: " + ' '.join(all_evidence[j + 1:]) + " </s>"
                input_ids = tokenizer.encode(to_invert, return_tensors="pt", truncation=True).to("cuda")
                res = asker.generate(input_ids, **GENERATOR_ARGS)
                inverted_question = tokenizer.batch_decode(res, skip_special_tokens=True)
                data_list.append({"question_ev": ques_ev, "inverted_ques": inverted_question})
            if i % 100 == 0: 
                print("Gone through " + str(i) + " / " + str(len(data)) + " examples.")   

            if i == 1000:
                print("Saving small...")
                with open(SMALL_OUTFILE, 'w') as f:
                    json.dump(data_list, f, indent=4)
                print("Small save complete. Resuming generation...")

            elif i % 5000 == 0:
                print("Saving...")
                with open(OUTFILE, 'w') as f:
                    json.dump(data_list, f, indent=4)
                print("Save complete. Resuming generation...")

    print("Done.")

    print("Writing final JSON...")
    with open(OUTFILE, 'w') as f:
        json.dump(data_list, f, indent=4)
    print("Done.")
    print("Generation complete. Have a nice day!")

if __name__== '__main__':
    main()
