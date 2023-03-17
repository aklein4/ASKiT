import transformers
from datasets import load_dataset, load_metric
from pynvml import *
import nltk
nltk.download('punkt')
import string
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

MAX_INPUT_LENGTH = 512

MAX_TARGET_LENGTH = 128

OUTPUT_DIR = "modelz"

DATA_PATH = "generated_data/generated_training_data_small.json"


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

model_checkpoint = "t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tr_data = load_dataset("json", data_files=DATA_PATH, split='train[:2%]')
v_data = load_dataset("json", data_files=DATA_PATH, split='train[2%:3%]')

print("Initializing... 1.")
print_gpu_utilization()

def addGenPrefix(example):
        example['chosen'] = "generate question: " + example['chosen'].strip()
        return example


def addEOS(example):
    if len(example['question']) > MAX_TARGET_LENGTH:
         example['question'] = example['question'][:MAX_TARGET_LENGTH]
    if len(example['chosen']) > MAX_INPUT_LENGTH:
         example['chosen'] = example['chosen'][:MAX_INPUT_LENGTH]
    #example['question'] = example['question'] + " </s>"
    #aexample['chosen'] = example['chosen'] + " </s>"
    return example


def removeSepTokens(example):
    example['chosen'] = example['chosen'].replace('<sep>', '')
    return example


e_tr_data = tr_data.map(addGenPrefix)
e_tr_data = e_tr_data.map(removeSepTokens)
e_tr_data = e_tr_data.map(addEOS)
e_v_data = v_data.map(addGenPrefix)
e_v_data = e_v_data.map(removeSepTokens)
e_v_data = e_v_data.map(addEOS)


def preprocess_data(examples):
    model_inputs = tokenizer(examples["chosen"], max_length=MAX_INPUT_LENGTH, truncation=True)

    #with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["question"], max_length=MAX_TARGET_LENGTH, text_target = examples['question'],
                        truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tr_tok_data = e_tr_data.map(preprocess_data, batched=True)
v_tok_data = e_v_data.map(preprocess_data, batched=True)

print("Data processed...2.")
print_gpu_utilization()

batch_size = 4
model_dir = OUTPUT_DIR
args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model_init=model_init.to("cuda"),
    args=args,
    train_dataset=tr_tok_data,
    eval_dataset=v_tok_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


print("About to train...")
print_gpu_utilization()
trainer.train()

trainer.save_model()