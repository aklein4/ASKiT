import torch

from datasets import load_dataset, load_metric, list_metrics
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, T5ForConditionalGeneration, T5TokenizerFast

from tqdm import tqdm

from typing import Dict, List, Optional

import dataclasses
from dataclasses import dataclass, field

import logging
import os
import sys

import numpy as np
import torch

from huggingface_hub import login

from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    EvalPrediction,
    DataCollator,
    Trainer,
    TrainingArguments)
import wandb

OUTPUT_DIR = "new_models/testpoint"

login('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')
raw_dataset = load_dataset("squad_modified_for_t5_qg.py")
checkpoint = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenizer.sep_token = '<sep>'

tokenizer.add_tokens(['<sep>'])
print(len(tokenizer))
model.resize_token_embeddings(len(tokenizer))

max_input_length =  512
max_target_length = 64

def convert_to_features(example_batch):

    input_encodings = tokenizer.batch_encode_plus(example_batch['context'], 
                                                  max_length=max_input_length, 
                                                  add_special_tokens=True,
                                                  truncation=True, 
                                                  pad_to_max_length=True)
    
    target_encodings = tokenizer.batch_encode_plus(example_batch['questions'], 
                                                   max_length=max_target_length, 
                                                   add_special_tokens=True,
                                                   truncation=True, pad_to_max_length=True)
                                                   
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids']
        ,'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings

def add_eos_examples(example):
  example['context'] = example['context'] + " </s>"
  example['questions'] = example['questions'] + " </s>"
  return example


def add_special_tokens(example):
  example['questions'] = example['questions'].replace("{sep_token}", '<sep>')
  return example

tokenized_dataset  = raw_dataset.map(add_eos_examples)
tokenized_dataset = tokenized_dataset.map(add_special_tokens)
tokenized_dataset  = tokenized_dataset.map(convert_to_features,  batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(
    ["context", "questions"]
)

train_dataset = tokenized_dataset["train"]
valid_dataset = tokenized_dataset["validation"]

columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

torch.save(train_dataset, 'train_data1.pt')
torch.save(valid_dataset, 'valid_data1.pt')

"""
@dataclass
class T2TDataCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    
    #Take a list of samples from a Dataset and collate them into a batch.
    #Returns:
    #A dictionary of tensors
    
    
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100 
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])
    
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
        'decoder_attention_mask': decoder_attention_mask
    }
  
training_args = TrainingArguments(output_dir=OUTPUT_DIR, 
                                  per_device_train_batch_size=4, 
                                  per_device_eval_batch_size=4,
                                  gradient_accumulation_steps=16,
                                  learning_rate=1e-4, 
                                  num_train_epochs=7,
                                  logging_steps=100,
                                  run_name="end2end-questions-generation",
                                  evaluation_strategy="steps",
                                  save_steps=500,
                                  report_to="wandb",
                                  push_to_hub=True,
                                  push_to_hub_model_id="t5-end2end-questions-generation")

logger = logging.getLogger(__name__)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=T2TDataCollator()
)

# Training
trainer.train()

# When training is done, we push the fine-tuned model to the Hub
trainer.push_to_hub("t5-end2end-questions-generation")
"""

