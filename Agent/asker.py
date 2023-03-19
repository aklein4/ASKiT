
import torch
import torch.nn as nn
import torch.nn.functional as F 

from transformers import AutoTokenizer, T5ForConditionalGeneration


ASKER_MODEL = "matthv/third_t5-end2end-questions-generation"
GENERATOR_ARGS = {
  "max_length": 128,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
}

class Asker(nn.Module):
    
    def __init__(self, load=None):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained(ASKER_MODEL)


    def forward(self, question, evidence):

        input_string = "generate question: " + question + "<sep>" + " ".join(evidence) + " </s>"

        input_ids = self.tokenizer.encode(input_string, return_tensors="pt", truncation=True)
        res = self.model.generate(input_ids, **GENERATOR_ARGS)
        output = self.tokenizer.batch_decode(res, skip_special_tokens=True)

        return output
