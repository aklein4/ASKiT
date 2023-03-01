# Asker

## Goal

This will be the model that asks questions. Given a current question and a set of evidence that partially answers that question, what can we ask to help us figure out the rest? This model should take input as the current question (semantic search encoding) and possibly some other context (highly rated policy action statements) and produce a sub-question such that when ASKiT asnwers it, the answer will help figure out the original question.

## TODO

 - [ ] Find a pretrained model (next word prediction?)
 - [ ] Pretrain on inversion: given the current state, try and predict the original question
 - [ ] Fine-tune using non-differentiable optimization (policy evaluation or monte-carlo performance)
 - [ ] Pretrain within the RL framework

## Research

TBD

## Notes

 - Might also need some regulation during finetuning, such as an adverserial network, to make sure that the questions being asked are 'human like'
 - We need to make sure that this is actually 'asking' questions, rather than giving learned answers disguised as questions
