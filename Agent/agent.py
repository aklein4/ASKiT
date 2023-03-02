
import torch
import torch.nn as nn
import torch.nn.functional as F 

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForPreTraining


ACT_PRETRAINED = "prajjwal1/bert-mini"
SUB_PRETRAINED = "deepset/tinyroberta-squad2"


class Agent(nn.Module):
    
    def __init__(self, load=None):
        """ More accurate search model to choose evidence sentences.
        - Uses each sentence as evidence, so slower but more accurate than Searcher

        Args:
            load (_type_, optional): Folder of checkpoint to load, otherwise pretrained. Defaults to None.
        """
        super().__init__()
    
        # load model data
        if load is None:
            self.act_tokenizer = AutoTokenizer.from_pretrained(ACT_PRETRAINED)
            self.act_encoder = BertForPreTraining.from_pretrained(ACT_PRETRAINED)
            self.sub_tokenizer = AutoTokenizer.from_pretrained(SUB_PRETRAINED)
            self.sub_encoder = AutoModelForQuestionAnswering.from_pretrained(SUB_PRETRAINED)
            
        else:
            self.act_tokenizer = AutoTokenizer.from_pretrained(load + "/act_tokenizer")
            self.act_encoder = BertForPreTraining.from_pretrained(load + "/act_encoder")
            self.sub_tokenizer = AutoTokenizer.from_pretrained(load + "/sub_tokenizer")
            self.sub_encoder = AutoModelForQuestionAnswering.from_pretrained(load + "/sub_encoder")


    def rateSub(self, x):
        
        questions, evidence = x
        assert len(questions) == len(evidence)
        
        toks = self.sub_tokenizer(questions, evidence, padding=True, return_tensors="pt")
        out = self.sub_encoder(
            toks["input_ids"].to(self.sub_encoder.device),
            # token_type_ids = toks["token_type_ids"].to(self.sub_encoder.device),
            attention_mask = toks["attention_mask"].to(self.sub_encoder.device)
        )
        
        preds = -(out.start_logits[:,0] + out.start_logits[:,1]) / 2
        
        return preds.unsqueeze(1)


    def forward(self, x):

        questions, evidence, actions = x
        assert len(questions) == len(evidence) and len(evidence) == len(actions)

        n_actions = len(actions[0])
        assert max(1 if len(a)!=n_actions else 0 for a in actions) == 0
        
        states = []
        for i in range(len(questions)):
            states.append(questions[i] + " " + evidence[i])

        vec_states = []
        vec_actions = []

        for l in range(len(states)):
            vec_states += [states[l]] * n_actions
            vec_actions += actions[l]

        # get output prediction from each state-action pair
        try:
            toks = self.act_tokenizer(vec_states, vec_actions, padding=True, return_tensors='pt')
            preds = self.act_encoder(
                toks["input_ids"].to(self.act_encoder.device),
                token_type_ids = toks["token_type_ids"].to(self.act_encoder.device),
                attention_mask = toks["attention_mask"].to(self.act_encoder.device)
                ).seq_relationship_logits[:,1]

            preds = torch.reshape(preds, (len(states), n_actions))
        except KeyboardInterrupt:
            exit(0)
        except:
            preds = torch.zeros((len(states), n_actions), device=self.act_encoder.device, dtype=torch.float32, requires_grad=True)

        try:
            sub_ratings = self.rateSub((questions, evidence))
        except KeyboardInterrupt:
            exit(0)
        except:
            sub_ratings = torch.zeros((len(states), 1), device=self.sub_encoder.device, dtype=torch.float32, requires_grad=True)

        preds = torch.cat([sub_ratings, preds], dim=1)

        # return 1d tensor
        return preds

