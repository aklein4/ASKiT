
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


# model to use to encode corpus
ENCODING_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' # pretrained

# model to use to create latent search vectors
# SEARCH_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' # pretrained
SEARCH_MODEL = "checkpoints/searcher-p"
SEARCH_TOKEN_SUFFIX = "_tokenizer"


# Take average of all tokens, see pretrained model on huggingface
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Searcher(nn.Module):

    def __init__(self):
        """ Semantic search model.
        - Encodes corpus into vector for each sentence
        - Encodes current state into vector
        - Rates corpus sentences as u_sentence^T * v_state
        """
        super().__init__()
    
        # should not be trained
        self.encoder = SentenceTransformer(ENCODING_MODEL)
        
        # is trained
        self.search_tokenizer = AutoTokenizer.from_pretrained(SEARCH_MODEL+SEARCH_TOKEN_SUFFIX)
        self.search_encoder = AutoModel.from_pretrained(SEARCH_MODEL)


    def encode(self, corpus):
        """ Encode a list of sentences into a corpus matrix

        Args:
            corpus (_type_): List of sentences in the corpus

        Returns:
            _type_: Matrix with dim_0 = sentences, dim_1 = encodings
        """
        return self.encoder(corpus, convert_to_tensor=True)


    def forward(self, x):
        """ Given a state and a corpus encoding, return ratings for each element in the corpus

        Args:
            x (_type_): (states, corpuses) tuple of lists, states are strings and corpuses are corpus matrixes

        Returns:
            _type_: List of 1d rating tensors
        """

        # get lists from tuple
        sentences, corpuses = x
        assert len(sentences) == len(corpuses)

        # encode each sentence
        search_ins = self.search_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(corpuses[0].device)
        search_outs = self.search_encoder(**search_ins, return_dict=True)
  
        # convert to vectors as seen in pretrained
        h = mean_pooling(search_outs, search_ins["attention_mask"])

        # get list of ratings
        preds = []
        for i in range(len(sentences)):
            # creates 1d vector, with v[s] is rating of s-th sentence in corpus
            scores = corpuses[i] @ h[i]
            preds.append(scores)

        return preds
