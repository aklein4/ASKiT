
import torch
import torch.nn as nn
import torch.nn.functional as F 

from transformers import AutoModel, AutoTokenizer


"""
https://huggingface.co/mrm8488/bert-medium-finetuned-squadv2
"""

TOKENIZER = None
MODEL = None
HEAD = None

class Responder(nn.Module):
    
    def __init__(self, load=None):
        super().__init__()
        
        self.act_tokenizer = None
        self.act_encoder = None
        self.act_head = nn.Linear(256, 1, bias=True)
        if load is None:
            self.act_tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-mini-5-finetuned-squadv2")
            self.act_encoder = AutoModel.from_pretrained("mrm8488/bert-mini-5-finetuned-squadv2")
        else:
            self.act_tokenizer = AutoTokenizer.from_pretrained(load + "/act_tokenizer")
            self.act_encoder = AutoModel.from_pretrained(load + "/act_encoder")
            self.act_head.load_state_dict(torch.load(load + "/act_head.pt"))

        self.sub_tokenizer = None
        self.sub_encoder = None
        self.sub_head = nn.Linear(256, 1, bias=True)
        if load is None:
            self.sub_tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-mini-5-finetuned-squadv2")
            self.sub_encoder = AutoModel.from_pretrained("mrm8488/bert-mini-5-finetuned-squadv2")
        else:
            self.sub_tokenizer = AutoTokenizer.from_pretrained(load + "/sub_tokenizer")
            self.sub_encoder = AutoModel.from_pretrained(load + "/sub_encoder")
            self.sub_head.load_state_dict(torch.load(load + "/sub_head.pt"))
        
    
    def subForward(self, x):
        # x is list of states

        toks = self.sub_tokenizer(x, [""]*len(x), padding=True, truncation=True, return_tensors='pt')
        cls_enc = self.sub_encoder(toks["input_ids"], token_type_ids=toks["token_type_ids"], attention_mask=toks['attention_mask']).last_hidden_state[:,0]

        preds = self.sub_head(cls_enc)

        # only batch dimension
        return preds[:,0]


    def actForward(self, x):
        # tuple of lists

        states, actions = x

        toks = self.act_tokenizer(states, actions, padding=True, truncation=True, return_tensors='pt')
        cls_enc = self.act_encoder(toks["input_ids"], token_type_ids=toks["token_type_ids"], attention_mask=toks['attention_mask']).last_hidden_state[:,0]

        preds = self.act_head(cls_enc)

        # only batch dimension
        return preds[:,0]

    
    def forward(self, x):
        return self.actForward(x), self.subForward(x[0])