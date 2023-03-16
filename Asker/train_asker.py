import torch
import json
import random
from typing import Dict, List, Optional

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

MAX_INPUT_LENGTH = 512

MAX_TARGET_LENGTH = 128


def removeSepToken(data):
    for d in data:
        d["chosen"] = d["chosen"].replace("<sep>", "")


def main():
    data = load_dataset("json", data_files=DATA_PATH)
    
    # Load tokenizer/asker model
    asker = T5ForConditionalGeneration.from_pretrained(ASKER_MODEL)
    tokenizer = T5TokenizerFast.from_pretrained("t5-base", model_max_length=512)
    
    # Consider '<sep>' token
    tokenizer.sep_token = '<sep>'
    tokenizer.add_tokens(['<sep>'])
    asker.resize_token_embeddings(len(tokenizer))
    
    # Define some in-line processing functions
    def addGenPrefix(example):
        example['chosen'] = "generate question: " + example['chosen'].strip()
        return example

    def convertToFeatures(example_batch):

        input_encodings = tokenizer.batch_encode_plus(example_batch['context'], 
                                                        max_length=MAX_INPUT_LENGTH, 
                                                        add_special_tokens=True,
                                                        truncation=True, 
                                                        pad_to_max_length=True)

        target_encodings = tokenizer.batch_encode_plus(example_batch['questions'], 
                                                        max_length=MAX_TARGET_LENGTH, 
                                                        add_special_tokens=True,
                                                        truncation=True, pad_to_max_length=True)
                                                        
        encodings = {
            'input_ids': input_encodings['input_ids'], 
            'attention_mask': input_encodings['attention_mask'],
            'decoder_input_ids': target_encodings['input_ids']
            ,'decoder_attention_mask': target_encodings['attention_mask']
        }

        return encodings

    def addEOS(example):
        example['question'] = example['question'] + " </s>"
        example['chosen'] = example['chosen'] + " </s>"
        return example
    
    def removeSepTokens(example):
        example['questions'] = example['questions'].replace('<sep>', '')
        return example

    # Process data
    tok_data = data.map(addGenPrefix)
    tok_data = tok_data.map(addEOS)
    tok_data = tok_data.map(convertToFeatures, batched=True)
    print(tok_data["train"][0]["chosen"])
    #tok_data  = map(add_eos_examples, data)
    #tok_data = tok_data.map(convert_to_features, batched=True)
    #print(tok_data[0]["question"])

    #print("Loading data...")
    #data = load_dataset("json", DATA_PATH)
    #print("Done.")
    #[print(data[i]) for i in range(10)]


if __name__== '__main__':
    main()
