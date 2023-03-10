
import torch
import torch.nn as nn
import torch.nn.functional as F 

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForPreTraining


# pretrained for next sentence prediction
ACT_PRETRAINED = "prajjwal1/bert-mini"

# pretrained for question answering
SUB_PRETRAINED = "deepset/tinyroberta-squad2"

# smaller model
# SUB_PRETRAINED = "deepset/minilm-uncased-squad2"


class Agent(nn.Module):
    
    def __init__(self, load=None):
        """ An RL agent to choose the next action given the current question and evidence state.
        - act_model is a policy model to evaluate actions (collecting evidence)
        - sub_model is a policy model to evaluate answer submissions

        Args:
            load (str, optional): Folder of checkpoint to load, otherwise pretrained. Defaults to None.
        """
        super().__init__()
    
        # handles ratings for actions
        self.act_encoder = None
        self.act_tokenizer = None

        # handles ratings for submissions
        self.sub_encoder = None
        self.sub_tokenizer = None

        # Load pretrained models
        if load is None:
            self.act_tokenizer = AutoTokenizer.from_pretrained(ACT_PRETRAINED)
            self.act_encoder = BertForPreTraining.from_pretrained(ACT_PRETRAINED)
            self.sub_tokenizer = AutoTokenizer.from_pretrained(SUB_PRETRAINED)
            self.sub_encoder = AutoModelForQuestionAnswering.from_pretrained(SUB_PRETRAINED)
            
        # load checkpoint
        else:
            self.act_tokenizer = AutoTokenizer.from_pretrained(load + "/act_tokenizer")
            self.act_encoder = BertForPreTraining.from_pretrained(load + "/act_encoder")
            self.sub_tokenizer = AutoTokenizer.from_pretrained(load + "/sub_tokenizer")
            self.sub_encoder = AutoModelForQuestionAnswering.from_pretrained(load + "/sub_encoder")


    def rateSub(self, x):
        """Given a question and evidence, return a rating of submitting for an answer.

        Args:
            x (tuple): (question, evidence) lists with the same length

        Returns:
            torch.tensor: [num_questions, 1] tensor of ratings
        """

        # split tuple
        questions, evidence = x
        assert len(questions) == len(evidence)
        
        # get output prediction from each question-evidence pair
        toks = self.sub_tokenizer(questions, evidence, padding=True, return_tensors="pt", truncation=True, max_length=512)
        out = self.sub_encoder(
            toks["input_ids"].to(self.sub_encoder.device),
            # token_type_ids = toks["token_type_ids"].to(self.sub_encoder.device),
            attention_mask = toks["attention_mask"].to(self.sub_encoder.device)
        )
        
        # get average of start and end [CLS] logits, negative so higher is better
        preds = -(out.start_logits[:,0] + out.start_logits[:,1]) / 2
        
        return preds.unsqueeze(1)


    def forward(self, x, debug=False):
        """ Given a question, evidence, and list of actions, return a policy rating of each action.

        Args:
            x (tuple): (questions, evidence, actions), questions and evidence are lists of strings, actions is a list of lists of strings

        Returns:
            torch.tensor: [num_questions, num_actions] tensor of ratings
        """

        # unpack tuple
        questions, evidence, actions = x
        assert len(questions) == len(evidence) and len(evidence) == len(actions)

        # check that all actions have the same length
        n_actions = len(actions[0])
        assert max(1 if len(a)!=n_actions else 0 for a in actions) == 0
        
        # turn each question-evidence pair into a state
        states = []
        for i in range(len(questions)):
            states.append(questions[i] + " " + evidence[i])

        # vectorize the batched actions
        vec_states = []
        vec_actions = []
        for l in range(len(states)):
            vec_states += [states[l]] * n_actions
            vec_actions += actions[l]

        # get output prediction from each state-action pair
        toks = self.act_tokenizer(vec_states, vec_actions, padding=True, return_tensors='pt', truncation=True, max_length=512)
        preds = self.act_encoder(
            toks["input_ids"].to(self.act_encoder.device),
            token_type_ids = toks["token_type_ids"].to(self.act_encoder.device),
            attention_mask = toks["attention_mask"].to(self.act_encoder.device)
            ).seq_relationship_logits[:,0]

        # reshape to [num_questions, num_actions] batches
        preds = torch.reshape(preds, (len(states), n_actions))

        # get rating of submitting for each question
        sub_ratings = self.rateSub((questions, evidence))

        # add the submission ratings to the action ratings in the first column
        preds = torch.cat([sub_ratings, preds], dim=1)
        
        return preds

