import torch
import json
import random
import logging
import os
import sys
from typing import Dict, List, Optional
from tqdm import tqdm
from huggingface_hub import login
import wandb


import dataclasses
from dataclasses import dataclass, field


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer, EvalPrediction, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, list_metrics

ASKER_MODEL = "ThomasSimonini/t5-end2end-question-generation"


DEVICE = torch.device("cuda")

DATA_PATH = "generated_data/generated_training_data_small.json"

MAX_INPUT_LENGTH = 512

MAX_TARGET_LENGTH = 128

OUTPUT_DIR = "models/checkpoints/"


def main():
    wandb.login()
    login('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')
    tr_data = load_dataset("json", data_files=DATA_PATH, split='train[:90%]')
    v_data = load_dataset("json", data_files=DATA_PATH, split='train[90%:]')

    # Load tokenizer/asker model
    asker = T5ForConditionalGeneration.from_pretrained(ASKER_MODEL)
    tokenizer = T5TokenizerFast.from_pretrained("t5-base", model_max_length=512)
    
    asker = asker.to(DEVICE)

    # Consider '<sep>' token
    tokenizer.sep_token = '<sep>'
    tokenizer.add_tokens(['<sep>'])
    asker.resize_token_embeddings(len(tokenizer))
    
    # Define some in-line processing functions
    def addGenPrefix(example):
        example['chosen'] = "generate question: " + example['chosen'].strip()
        return example

    def convertToFeatures(example_batch):
        # Note pad_to_max_length will be removed, use padding instead if issues arise
        input_encodings = tokenizer.batch_encode_plus(example_batch['chosen'], 
                                                        max_length=MAX_INPUT_LENGTH, 
                                                        add_special_tokens=True,
                                                        truncation=True, 
                                                        padding='max_length')

        target_encodings = tokenizer.batch_encode_plus(example_batch['question'], 
                                                        max_length=MAX_TARGET_LENGTH, 
                                                        add_special_tokens=True,
                                                        truncation=True, 
                                                        padding='max_length')
                                                        
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
        example['chosen'] = example['chosen'].replace('<sep>', '')
        return example

    # Process data
    tr_tok_data = tr_data.map(addGenPrefix)
    tr_tok_data = tr_tok_data.map(addEOS)
    tr_tok_data = tr_tok_data.map(removeSepTokens)
    tr_tok_data = tr_tok_data.map(convertToFeatures, batched=True)

    v_tok_data = v_data.map(addGenPrefix)
    v_tok_data = v_tok_data.map(addEOS)
    v_tok_data = v_tok_data.map(removeSepTokens)
    v_tok_data = v_tok_data.map(convertToFeatures, batched=True)
    
    tr_tok_data = tr_tok_data.remove_columns(["question", "chosen"])
    v_tok_data = v_tok_data.remove_columns(["question", "chosen"])


    columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
    tr_tok_data.set_format(type='torch', columns=columns)
    v_tok_data.set_format(type='torch', columns=columns)

    torch.save(tr_tok_data, 'train_data_small.pt')
    torch.save(v_tok_data, 'valid_data_small.pt')

    # In-line initialize DataCollator    
    @dataclass
    class T2TDataCollator():
        def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
            """
            Take a list of samples from a Dataset and collate them into a batch.
            Returns:
            A dictionary of tensors
            """
            
            input_ids = torch.stack([example['input_ids'] for example in batch])
            lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
            lm_labels[lm_labels[:, :] == 0] = -100 
            attention_mask = torch.stack([example['attention_mask'] for example in batch])
            decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])
            

            d = {
                'input_ids': input_ids, 
                'attention_mask': attention_mask,
                'labels': lm_labels, 
                'decoder_attention_mask': decoder_attention_mask
            }
            return d
        
    training_args = TrainingArguments(output_dir=OUTPUT_DIR,
                                      per_device_train_batch_size=1, 
                                      per_device_eval_batch_size=1,
                                      optim='adamw_torch',
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
        model=asker,
        args=training_args,
        train_dataset=tr_tok_data,
        eval_dataset=v_tok_data,
        data_collator=T2TDataCollator()
    )

    # Training
    trainer.train()

    trainer.push_to_hub("t5-end2end-questions-generation")
    wandb.finish()


if __name__== '__main__':
    main()
