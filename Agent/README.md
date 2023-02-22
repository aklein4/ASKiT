# Agent

## Goal

This model will be bread (though possibly not the butter) of this project. It will control how we gather evidence statements from the corpus, based on concepts from the Latent Retrieval paper. However, the difference between our model and theirs is that they use single-hop reasoning (they only need to find one piece of evidence per question), while we need to construct a chain of evidence. To achieve this, we will turn the problem from unsupervised identification to semi-supervised reinforcement. The details of implementation are currently written in a notebook and will be tedious to write digitally.

## TODO

 - [x] Find a pretrained semantic search encoder
     - [ ] Fine-tune encoder on sentences that require a bt more context (ex. incorporate title)?
 - [ ] Pretrain recurrent question encoder using known HotpotQA evidence chains
 - [ ] Fine-tune evidence chaining and answer submission using RL
 
## Research

This paper - [https://arxiv.org/pdf/1906.00300.pdf#page=10&zoom=100,89,695](https://arxiv.org/pdf/1906.00300.pdf#page=10&zoom=100,89,695) - shows how to do this when we only need to find one piece of evidence. This pretrained model - [https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) will be the one that we base the latent encoder on. For actual question answering, we can use pretrained BERT from the transformers library.

## Notes

 - How much do we discount future rewards, since later on we will want to collect as few pieces of evidence as necessary?
 - Reward shaping might take some tuning, but possible Values should probably be in some predefined range to help with model activation choices.
 - How do we both choose an answer and evaluate its value? This will take some thought, and may require 2 seperate models.
