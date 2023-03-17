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

checkpoint = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
model.resize_token_embeddings(32101)

train_dataset = torch.load('train_data1.pt')
valid_dataset = torch.load('valid_data1.pt')

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
                                  evaluation_strategy="steps",
                                  save_steps=500)

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


