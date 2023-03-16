
import torch

import sys
sys.path.append('../Asker')
from searcher import Searcher
from agent import Agent
from environment import Environment

import json
import random

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer, EvalPrediction, Trainer, TrainingArguments

ASKER_MODEL = "ThomasSimonini/t5-end2end-question-generation"
GENERATOR_ARGS = {
  "max_length": 128,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
}


DEVICE = torch.device("cuda")

DATA_PATH = "ASKiT/Asker/generated_data/generated_training_data_small.json"



def main():
    print("Loading data...")
    data = load_dataset("json", DATA_PATH)
    print("Done.")
    [print(data[i]) for i in range(10)]


if __name__== '__main__':
    main()
