
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


class Chooser(nn.Module):
    
    def __init__(self, load=None):
        super().__init__()
        
        self.tokenizer = None
        self.encoder = None
        self.head = nn.Sequential(
            nn.Linear(256, 64, bias=True),
            nn.Dropout(p=0.1),
            nn.ELU(),
            nn.Linear(64, 1, bias=True)
        )

        if load is None:
            self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-mini-5-finetuned-squadv2")
            self.encoder = AutoModel.from_pretrained("mrm8488/bert-mini-5-finetuned-squadv2")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(load + "/tokenizer")
            self.encoder = AutoModel.from_pretrained(load + "/encoder")
            self.head.load_state_dict(torch.load(load + "head.pt"))

        # self.sub_tokenizer = None
        # self.sub_encoder = None
        # self.sub_head = nn.Linear(256, 1, bias=True)
        # if load is None:
        #     self.sub_tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-mini-5-finetuned-squadv2")
        #     self.sub_encoder = AutoModel.from_pretrained("mrm8488/bert-mini-5-finetuned-squadv2")
        # else:
        #     self.sub_tokenizer = AutoTokenizer.from_pretrained(load + "/sub_tokenizer")
        #     self.sub_encoder = AutoModel.from_pretrained(load + "/sub_encoder")
        #     self.sub_head.load_state_dict(torch.load(load + "/sub_head.pt"))
        
    """ Should probably use a seperate model for choosing to submit """
    # def subForward(self, x):
    #     # x is list of states

    #     toks = self.sub_tokenizer(x, [""]*len(x), padding=True, truncation=True, return_tensors='pt')
    #     cls_enc = self.sub_encoder(toks["input_ids"], token_type_ids=toks["token_type_ids"], attention_mask=toks['attention_mask']).last_hidden_state[:,0]

    #     preds = self.sub_head(cls_enc)

    #     # only batch dimension
    #     return preds[:,0]


    def forward(self, x):
        # tuple of lists

        states, actions = x

        vec_states = states
        vec_actions = actions

        batched = isinstance(states[0], list)
        if batched:
            vec_states = []
            vec_actions = []

            for l in range(len(states)):
                vec_states += states[l]
                vec_actions += actions[l]

        try:
            toks = self.tokenizer(vec_states, vec_actions, padding=True, return_tensors='pt')
            cls_enc = self.encoder(
                    toks["input_ids"].to(self.encoder.device),
                    token_type_ids=toks["token_type_ids"].to(self.encoder.device),
                    attention_mask=toks['attention_mask'].to(self.encoder.device)
            ).last_hidden_state[:,0]

            preds = self.head(cls_enc)[:,0]
        except:
            print("Encoding Error Caught")
            preds = torch.zeros([len(vec_states)], device=self.encoder.device, dtype=torch.float32, requires_grad=True)

        start = 0
        if batched:
            pred_list = []
            end = start

            for l in range(len(states)):
                end = start + len(states[l])
                pred_list.append(preds[start:end])
                start = end

            return pred_list

        return preds
