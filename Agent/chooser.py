
import torch
import torch.nn as nn
import torch.nn.functional as F 

from transformers import AutoModel, AutoTokenizer


"""
https://huggingface.co/mrm8488/bert-medium-finetuned-squadv2
"""
PRETRAINED = "mrm8488/bert-mini-5-finetuned-squadv2"

class Chooser(nn.Module):
    
    def __init__(self, load=None):
        """ More accurate search model to choose evidence sentences.
        - Uses each sentence as evidence, so slower but more accurate than Searcher

        Args:
            load (_type_, optional): Folder of checkpoint to load, otherwise pretrained. Defaults to None.
        """
        super().__init__()
        
        # transformer
        self.tokenizer = None
        self.encoder = None

        # turn transformer output into single scalar
        self.head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(256, 64, bias=True),
            nn.ELU(),
            nn.Linear(64, 1, bias=False)
        )

        # from cls version
        # self.head = nn.Sequential(
        #     nn.Linear(256, 64, bias=False),
        #     nn.Dropout(p=0.1),
        #     nn.ELU(), # tanh is probably better
        #     nn.Linear(64, 1, bias=False)
        # )

        # load model data
        if load is None:
            self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
            self.encoder = AutoModel.from_pretrained(PRETRAINED)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(load + "/tokenizer")
            self.encoder = AutoModel.from_pretrained(load + "/encoder")
            self.head.load_state_dict(torch.load(load + "head.pt"))

        # TODO: Seperate model for submission action?


    def forward(self, x):
        """ Get scores for a set of states and action sentences.

        Args:
            x (_type_): (states, actions) tuple of lists, lists can be nested to keep track of batches

        Returns:
            _type_: If non-batched, 1d tensor of scores. If batched, list of 1d score tensors
        """

        states, actions = x

        # transformer can only handle single level lists
        vec_states = states
        vec_actions = actions

        # check if lists are nested
        batched = isinstance(states[0], list)

        # cat batches into one long list
        if batched:
            vec_states = []
            vec_actions = []

            for l in range(len(states)):
                vec_states += states[l]
                vec_actions += actions[l]

        assert len(vec_actions) == len(vec_states) # each action must correspond 1-to-1 to a state

        try:
            # get output prediction from each state-action pair
            toks = self.tokenizer(vec_states, vec_actions, padding=True, return_tensors='pt')
            out = self.encoder(
                    toks["input_ids"].to(self.encoder.device),
                    token_type_ids=toks["token_type_ids"].to(self.encoder.device),
                    attention_mask=toks['attention_mask'].to(self.encoder.device)
            ).pooler_output

            # get 1d tensor
            preds = self.head(out)[:,0]

        except:
            # sometimes it fails (input too long or invalid vocab?) so we just return zeros to avoid termination
            preds = torch.zeros([len(vec_states)], device=self.encoder.device, dtype=torch.float32, requires_grad=True)

        # if batched, we convert 1d tensor into list of corresponding batches
        if batched:
            start = 0
            pred_list = []
            end = start

            # segment pred into batches corresponding to original batch sizes
            for l in range(len(states)):
                end = start + len(states[l])
                pred_list.append(preds[start:end])
                start = end

            # return list of 1d tensors
            return pred_list

        # return 1d tensor
        return preds
