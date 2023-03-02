# Asker

## Goal

This will be the model that asks questions. Given a current question and a set of evidence that partially answers that question, what can we ask to help us figure out the rest? This model should take input as the current question (semantic search encoding) and possibly some other context (highly rated policy action statements) and produce a sub-question such that when ASKiT asnwers it, the answer will help figure out the original question.

## TODO

 - [ ] Get example sub-questions for pretraining
     - [x] Find a pretrained context+answer -> question model for Inverter
     - [ ] Fine-tune inverter on Hotpot?
     - [ ] Generate questions for missing context + correct answer for each possible state
 - [ ] Train Asker to generate questions given the state
     - [x] Find a pretrained model
     - [ ] Train to predict inverter-generated questions given the state
 - [ ] Fine-tune Asker using PPO
     - [ ] Want to maximize the final reward when we use the answer to the subquestion as evidence

## Research

Here [voidful/context-only-question-generator](https://huggingface.co/voidful/context-only-question-generator) is a model that we could start with for Asker. To train Asker, we could use [mrm8488/t5-base-finetuned-question-generation-ap](https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap) or [ThomasSimonini/t5-end2end-question-generation](https://huggingface.co/ThomasSimonini/t5-end2end-question-generation) as an inverter that takes the missing context and answer for a given state, and generate a question form that. Then, Asker would be trained to predict that question given the state (the inverter 'bridges the gap' between the state and final answer). We then fine-tune Asker with PPO

## Notes

 - Might also need some regulation during finetuning, such as an adverserial network, to make sure that the questions being asked are 'human like'
 - We need to make sure that this is actually 'asking' questions, rather than giving learned answers disguised as questions
 - Can we somehow incorporate Searcher's encoding when generating questions?
