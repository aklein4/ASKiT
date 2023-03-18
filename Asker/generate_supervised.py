from datasets import load_dataset, load_metric, list_metrics
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer, EvalPrediction, Trainer, TrainingArguments

from tqdm import tqdm
import torch
from typing import Dict, List, Optional

import dataclasses
from dataclasses import dataclass, field

import logging
import os
import sys

import numpy as np
import sys
sys.path.append('../Agent')
from searcher import Searcher
from agent import Agent
from environment import Environment

import json
import random

from transformers import T5ForConditionalGeneration, AutoTokenizer

ASKER_MODEL = "matthv/first_t5-end2end-questions-generation"
GENERATOR_ARGS = {
  "max_length": 128,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
}

INPUT_DATA = "../../local_data/hotpot_data/val.json"
INPUT_ENCODINGS = "../../local_data/corpus_encodings/val.pt"

OUTPUT_DATA = ""

AGENT_CHECKPOINT = "../../checkpoints/agent-pre"
SEARCHER_CHECKPOINT = "../../checkpoints/searcher-p"

STATES_PER = 1

DEVICE = torch.device("cpu")


def main():
    # load semantic search model
    search = Searcher(load=SEARCHER_CHECKPOINT)
    search = search.to(DEVICE)

    # load agent model
    chooser = Agent(load=AGENT_CHECKPOINT)
    chooser = chooser.to(DEVICE)

    example_tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
    example_asker = T5ForConditionalGeneration.from_pretrained(ASKER_MODEL)

    # load data
    env = Environment(
        INPUT_DATA,
        INPUT_ENCODINGS,
        search, chooser,
        8, device=torch.device(DEVICE),
        data_end=10
    )

    for i in range(len(env.data)):
        question = env.data[i]['question']
        chosen = env.greedyRollout(i, "", env.data[i]["raw_corpus"], env.corpus[i].float())
        amalg = '<sep>'.join(chosen)
        input_string = "generate question: " + amalg + " </s>"
        input_ids = example_tokenizer.encode(input_string, return_tensors="pt", truncation=True)
        res = example_asker.generate(input_ids, **GENERATOR_ARGS)
        output = example_tokenizer.batch_decode(res, skip_special_tokens=True)
        #output = [item.split("<sep>") for item in output][0][0].split("?")[0]+"?"        
        print("\n-----------------")
        print('Original Question\n', question)
        print('Chosen Evidence\n', chosen)
        # print('Target Evidence\n', target_evidence)
        print('New Question\n', output)

if __name__ == "__main__":
    main()