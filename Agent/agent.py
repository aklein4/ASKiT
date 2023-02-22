
import torch
import torch.nn as nn

from transformers import AutoTokenizer, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer, util

ENCODING_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
QA_MODEL = "deepset/bert-base-cased-squad2"

class Agent(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.q = None
        self.F = []
        self.state = None
        self.h_qF = None

        self.L_b = SentenceTransformer(ENCODING_MODEL)
        self.L_qF = SentenceTransformer(ENCODING_MODEL)
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


    def _Q_b(self, b):
        return self.b_activation(b @ self.h_qF)

    def _Q_a(self):
        
        inputs = self.qa_tokenizer(self.q, self.state[len(self.q)+1:], return_tensors="pt")
        outputs = self.qa_model(**inputs)

        starts = self.a_activation(outputs.start_logits)
        ends = self.a_activation(outputs.end_logits)

        return (starts.T + ends) / 2