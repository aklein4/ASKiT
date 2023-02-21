
import torch
import torch.nn as nn
import torch.nn.functional as F 


class Responder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # we might not want to use the large models
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')
        qa_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
        
        # grab the modules so that we can build on top of them
        self.bert = qa_model.bert
        self.W_ss = qa_model.qa_outputs
        
        # grab the size of bert's output
        self.bert_output_size = self.W_ss.in_features
        
        # initialize the extra layers for yes/no questions
        self.W_type = nn.Linear(self.bert_output_size, 1) # 0=yn, 1=ss
        self.W_yn = nn.Sequential(
            nn.Linear(self.bert_output_size, 512),
            nn.Dropout(p=0.5),
            nn.ELU(),
            nn.Linear(512, 1)
        ) # 0=no, 1=yes
        
    
    def forward(self, x):
        """
        Note: This is currently built to only handle one sentence at a time, and will break on batches
        """
        
        q, f = x
        
        # see paper for info on extra tokens
        q_tokens = ['[CLS]'] + self.tokenizer.tokenize(q) + ['[SEP]']
        f_tokens = self.tokenizer.tokenize(f) + ['[SEP]']
        tokens = q_tokens + f_tokens
        
        # turn into tensor
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        
        # sentence masking
        segments_ids = [0] * (len(q_tokens)) + [1] * (len(f_tokens))
        segments_tensor = torch.tensor([segments_ids])
        
        # get encoding from bert
        embeddings = self.bert(tokens_tensor, token_type_ids=segments_tensor)[0]
        cls_embedding = embeddings[:,0]
        sentence_embeddings = embeddings[:,1:]
        
        # check if this is substring type
        ss_type = torch.sigmoid(self.W_type(cls_embedding)) >= 0.0 # 0.5
        
        # yes/no question
        if not torch.all(ss_type):
            yn_pred = torch.sigmoid(self.W_yn(cls_embedding))
            return "yes" if torch.all(yn_pred >= 0.5) else "no"
        
        # ss question
        output = torch.squeeze(self.W_ss(embeddings))
        start, end = torch.argmax(output[:,0]), torch.argmax(output[:,1])
        return tokens[start:end+1]