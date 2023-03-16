import torch
import json
import random

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer, EvalPrediction, Trainer, TrainingArguments
from datasets import load_dataset

ASKER_MODEL = "ThomasSimonini/t5-end2end-question-generation"
GENERATOR_ARGS = {
  "max_length": 128,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
}


DEVICE = torch.device("cuda")

DATA_PATH = "generated_data/generated_training_data_small.json"


def appendGenPrefix(data):
    for d in data:
        d["chosen"] = "generate question: " + d["chosen"].strip()

def removeSepToken(data):
    for d in data:
        d["chosen"] = d["chosen"].replace("<sep>", "")

def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
        print("Size: " + str(len(data)))
        print(data[0])
        appendGenPrefix(data)
        print(data[0])
        removeSepToken(data)
        print(data[0])
    #print("Loading data...")
    #data = load_dataset("json", DATA_PATH)
    #print("Done.")
    #[print(data[i]) for i in range(10)]


if __name__== '__main__':
    main()
