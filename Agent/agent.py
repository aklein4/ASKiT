
import torch
import torch.nn as nn

from transformers import AutoTokenizer, BertForQuestionAnswering, AutoModel
from sentence_transformers import SentenceTransformer, util

"""
At some point this will handle the RL agent functionality
"""

ENCODING_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'

SEARCH_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
#SEARCH_MODEL = "checkpoints/Agent-unnorm-77_6"
SEARCH_TOKEN_SUFFIX = ""

QA_MODEL = "deepset/bert-base-cased-squad2"


# Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Agent(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.q = None
        self.F = []
        self.state = None
        self.h_qF = None
    
        self.L_b = SentenceTransformer(ENCODING_MODEL)
        
        self.L_qF_tokenizer = AutoTokenizer.from_pretrained(SEARCH_MODEL+SEARCH_TOKEN_SUFFIX)
        self.L_qF = AutoModel.from_pretrained(SEARCH_MODEL)

        self.b_activation = nn.Sigmoid()

        self.qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
        self.qa_model = BertForQuestionAnswering.from_pretrained(QA_MODEL)
        self.a_activation = nn.Sigmoid()
    

    def encode(self, corpus):
        return self.L_b.encode(corpus, convert_to_tensor=True)


    def setQuestion(self, question):
        self.q = question
        self.reset()

    def reset(self):
        self.F = []
        self.state = self.q
        self._updateEncoding()
    
    def _updateEncoding(self):
        self.h_qF = self.L_qF.encode(self.state, convert_to_tensor=True)


    def readFact(self, fact):
        self.F.append(fact)
        self.state += " " + fact
        self._updateEncoding()


    def forward(self, x):
        sentences, corpuses = x
        assert len(sentences) == len(corpuses)

        encoded_input = self.L_qF_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(corpuses[0].device)
        model_output = self.L_qF(**encoded_input, return_dict=True)
  
        h_qF = mean_pooling(model_output, encoded_input["attention_mask"])

        preds = []
        for i in range(len(sentences)):
            preds.append(corpuses[i] @ h_qF[i])

        return preds


    # TODO: implement these 2 functions
    def Qstate(self, state):
        pass
    def Qsubmit(self, state):
        pass


    def getAction(self, state, text_corpus, encodings, top_k):

        encoding_evals = self.forward(([state], [encodings]))[0]
        search_probs = torch.nn.functional.softmax(encoding_evals, dim=-1)

        top_vals, top_inds = torch.topk(encoding_evals, top_k)
        
        new_states = []
        for i in range(top_inds.shape[0]):
            new_states.append(state + text_corpus[top_inds[i]])

        Q_vals = self.Qstate(new_states)

        Q_sub_val = self.Qsubmit(state)

        if Q_sub_val.item() > torch.max(Q_vals).item():
            return -1, search_probs

        return torch.argmax(Q_vals), search_probs
    

    def _Q_b(self, b):
        return self.b_activation(b @ self.h_qF)

    def _Q_a(self):
        
        inputs = self.qa_tokenizer(self.q, self.state[len(self.q)+1:], return_tensors="pt")
        outputs = self.qa_model(**inputs)

        starts = self.a_activation(outputs.start_logits)
        ends = self.a_activation(outputs.end_logits)

        return (starts.T + ends) / 2
