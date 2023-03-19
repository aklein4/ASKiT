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

ASKER_MODEL = "matthv/third_t5-end2end-questions-generation"
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
    torch.no_grad()    

    # load semantic search model
    search = Searcher(load=SEARCHER_CHECKPOINT)
    search = search.to(DEVICE)

    # load agent model
    chooser = Agent(load=AGENT_CHECKPOINT)
    chooser = chooser.to(DEVICE)


    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
    asker = T5ForConditionalGeneration.from_pretrained(ASKER_MODEL)

    search.eval()
    chooser.eval()
    asker.eval()

    # load data
    env = Environment(
        INPUT_DATA,
        INPUT_ENCODINGS,
        search, chooser,
        8, device=torch.device(DEVICE),
        data_end=30
    )

    for i in range(len(env.data)):
        question = env.data[i]['question'] + "<sep>"
        chosen = env.greedyRollout(i, "", env.data[i]["raw_corpus"], env.corpus[i].float())
        input_list = []
        for j in range(len(chosen)):
            question += chosen[j]
            input_list.append(question)
        for k in range(len(input_list)):
            input_string = "generate question: " + input_list[k] + " </s>"
            input_ids = tokenizer.encode(input_string, return_tensors="pt", truncation=True)
            res = asker.generate(input_ids, **GENERATOR_ARGS)
            output = tokenizer.batch_decode(res, skip_special_tokens=True)
        #output = [item.split("<sep>") for item in output][0][0].split("?")[0]+"?"        
            print("\n-----------------")
            print('Original Question and Evidence\n', input_list[k])
            print('All evidence\n', chosen)
        # print('Target Evidence\n', target_evidence)
            print('New Question\n', output)

if __name__ == "__main__":
    main()
