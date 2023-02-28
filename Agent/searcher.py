
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


ENCODING_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' # pretrained

# SEARCH_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' # pretrained
SEARCH_MODEL = "checkpoints/searcher-p_noscale"
SEARCH_TOKEN_SUFFIX = "_tokenizer"

SEARCH_HEAD = None


# Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Searcher(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.encoder = SentenceTransformer(ENCODING_MODEL)
        
        self.search_tokenizer = AutoTokenizer.from_pretrained(SEARCH_MODEL+SEARCH_TOKEN_SUFFIX)
        self.search_encoder = AutoModel.from_pretrained(SEARCH_MODEL)
        self.search_head = nn.Identity()
        
        if SEARCH_HEAD is not None:
            self.search_head = nn.Linear(1, 1, bias=True)
            self.search_head.load_state_dict(torch.load(SEARCH_HEAD))


    def encode(self, corpus):
        return self.encoder(corpus, convert_to_tensor=True)


    def forward(self, x):
        sentences, corpuses = x
        assert len(sentences) == len(corpuses)

        search_ins = self.search_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(corpuses[0].device)
        search_outs = self.search_encoder(**search_ins, return_dict=True)
  
        h = mean_pooling(search_outs, search_ins["attention_mask"])

        preds = []
        for i in range(len(sentences)):
            preds.append(self.search_head(corpuses[i] @ h[i]))

        return preds
