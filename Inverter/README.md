
# Inverter

## Goal

We need to train Asker to ask questions that will lead to good evidence. However, for all of our known good evidence, we don't have questions that will lead to them. This model will create those questions. This model will take in a statement, and return a question (or a multiple) that would be answered by that statement.

## Research

TBD

## Notes

 - This should be pretty straight forward to train: Take a big set of question-answer pairs, then train it to predict the question based on the answer.
 - In order to generate good evidence questions, we may need to try multiple possabilities. This means take a piece of evidence, generate a sub-question for it, answer that question using Responder/Manager, and evaluate how well Manager evaluates that answer in context. We iterate over sub-questions to find the one with the highest evaluation, and use that one to train asker.