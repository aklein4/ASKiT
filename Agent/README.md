# Agent

## Goal

This model will be bread (though possibly not the butter) of this project. It will control how we gather evidence statements from the corpus, based on concepts from the Latent Retrieval paper. However, the difference between our model and theirs is that they use single-hop reasoning (they only need to find one piece of evidence per question), while we need to construct a chain of evidence. To achieve this, we will turn the problem from unsupervised identification to semi-supervised reinforcement. This will require 2 models: 1 sementic search model to search the corpus using pre-encoded representations, and a policy model to choose an evidence statement from those narrowed down by semantic search. These models can be easily pretrained, and will fine-tuned using RL (PPO algorithm).

## TODO

 - [x] Semantic Search Model Pretraining
     - [x] Find a pretrained semantic search encoder
     - [x] Pretrain state encoder on hotpot multihop-reasoning questions
 - [x] Policy Model Pretraining
     - [x] Find small question answering model (preferably with SQUAD 2.0 capabilities)
     - [x] Pretrain to identify hotpot evidence
 - [ ] Fine-tune full system using RL
     - [x] Create RL environment/PPO script
     - [ ] Fine-tune Policy model
     - [ ] Fine-tune semantic search model
 
## Research

This paper - [https://arxiv.org/pdf/1906.00300.pdf#page=10&zoom=100,89,695](https://arxiv.org/pdf/1906.00300.pdf#page=10&zoom=100,89,695) - shows how to do this when we only need to find one piece of evidence. This pretrained model - [https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) will be the one that we base the latent encoder on. For actual question answering, we can use pretrained BERT from the transformers library.

## Notes

 - How much do we discount future rewards, since later on we will want to collect as few pieces of evidence as necessary
   - Use gamma=1 with f1 smoothing
 - We could use entropy/temperature coefficient during PPO exploration to help increase exploitation
 - How could wee measure the 'overall quality' of a set of evidence for beam-search approaches?
