# Asker

## Goal

This model is the most novel part of our project. Given a question and a set of facts, this model should ask an easier sub-question, such that the answer will improve our fact set. This allows for powerful and explainable recursive multi-hop reasoning. Like Manager, this should be trained as an RL agent in the MDP:
 - State: The words that we currently have in our question.
 - Action: A word that is added to the end of our state in order to construct a full question.
 - Reward: A terminal reward based on the evaluation of Manager on the answer to the sub-question. Different reward shapes should be tested (ex. Packer reward mathod)

## Research

TBD

## Notes

 - Like Manager, this should be jump-started using Imitation of the questions created by Inverter.
 - It's possible for the questions that this asks to diverge from realistic questions, so we could regulate it with an adverserial network to make sure that its questions look 'human-like'.
  - We want to make sure that this model is asking QUESTIONS, not learned answers disguised as questions. We could regulate this using the Manager model: Plug the question in as a fact and check how that changes the evaluation. If the question itself suddenly makes the original question answerable, then this model is being to predictive.