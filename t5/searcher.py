
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import os


# model to use to create latent search vectors
PRETRAINED_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' # pretrained


# Take average of all tokens, (see pretrained model on huggingface)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Searcher(nn.Module):

    def __init__(self, load=None):
        """ Semantic search model.
        - Encodes corpus into vector for each sentence
        - Encodes current state into vector
        - Rates corpus sentences as u_sentence^T * v_state

        Args:
            load (str, optional): Folder of checkpoint to load, otherwise pretrained. Defaults to None.
        """
        super().__init__()
    
        # should not be trained
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        self.encoder_encoder = AutoModel.from_pretrained(PRETRAINED_MODEL)
        
        # used to create latent search vectors
        self.search_tokenizer = None 
        self.search_encoder = None 

        # load checkpoint
        if load is not None:
            self.search_tokenizer = AutoTokenizer.from_pretrained(os.path.join(load, 'tokenizer'))
            self.search_encoder = AutoModel.from_pretrained(os.path.join(load, 'model'))

        # load pretrained
        else:
            self.search_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            self.search_encoder = AutoModel.from_pretrained(PRETRAINED_MODEL)


    def encode(self, corpus):
        """ Encode a list of sentences into a corpus matrix

        Args:
            corpus (list): List of sentences in the corpus

        Returns:
            tensor: [sentences, latent_space] encoding matrix
        """
        search_ins = self.encoder_tokenizer(corpus, padding=True, truncation=True, return_tensors='pt').to(self.encoder_encoder.device)
        search_outs = self.encoder_encoder(**search_ins, return_dict=True)
    
        # convert to vectors as seen in pretrained
        h = mean_pooling(search_outs, search_ins["attention_mask"])
        return h


    def forward(self, x):
        """ Given a state and a corpus encoding, return ratings for each element in the corpus

        Args:
            x (tuple): (states, corpuses) tuple of lists, states are strings and corpuses are corpus matrixes

        Returns:
            list: List of 1d rating tensors
        """

        # get lists from tuple
        sentences, corpuses = x
        if isinstance(sentences, str) and not isinstance(corpuses, list):
            sentences = [sentences]
            corpuses = [corpuses]

        assert len(sentences) == len(corpuses)

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
