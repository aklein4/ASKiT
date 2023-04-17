
# ASKiT 

## Overview

Stanford CS224N Final Project. A text-based multi-hop reasoning question-answer NLP model to address the research question: "What if language models could ask their own questions?"

## Logistics

 - For each feature/model, there is a seperate branch and folder with relevant information in its README. This helps keep tasks seperated. See the respective branch for the most up-to-date version of a folder. (To move the changes from a different branch into the current one, use git merge other-branch)
 - To avoid large file hell, there is a google drive that we can use to share data. Put this data in a folder named local_data to avoid it getting pushed.
 - Some ideas have had branches created then removed due to a change in planning, see commit history for details.

## Roadmap
 
 - [x] 1. RL search/answering (Agent)
     - [x] Semantic Search Pretraining
     - [x] Policy pretraining
     - [x] PPO RL fine-tuning
 - [x] 2. Recursive Question Asking (Phraser, Responder, Asker)
     - [x] Question-inversion pretraining
     - [ ] KL-Divergence/Log-prob Fine-tuning
     - [ ] PPO RL fine-tuning (HopPotQA supervised)

## Brainstorming

 - Here [https://github.com/ad-freiburg/large-qa-datasets](https://github.com/ad-freiburg/large-qa-datasets) are a bunch of question answering datasets that could be useful.
 - It seems that if you make a small change to ASKiT - allowing it to replace statements in the corpus with its questions, rather than just expanding the F set - then the program becomes Turing complete.
 - ASKiT might be able to become a powerful chat-bot if we train it to predict the next sentence of a statement, rather than just answer questions. We could keep a corpus that contains previous information from the conversation as well as some information database like Wikipedia. Sub-questions would also become sub-statements prompted by Asker, rather than pure questions. (Similar: [https://github.com/google-research/language/tree/master/language/realm](https://github.com/google-research/language/tree/master/language/realm))
