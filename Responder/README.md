# Responder

## Goal

Create a model that can take in a question and set of facts, and respond with an answer. The answer should be in the form of yes/no or a substring from the facts/answer.

## TODO

 - [x] Create a basic class for answer prediction with yes/no capabilities.
 - [ ] Train the W_yn and W_type matrices to handle yes/no questions
 - [ ] Fine-tune the model of HotpotQA?

## Research

This paper: [https://arxiv.org/abs/1910.02610](https://arxiv.org/abs/1910.02610) has a good example for a very similar task. They propose using pre-trained BERT to encode the question and facts, then having 3 matrices to convert that encoding into an answer: W_type to decide whether the question is yes/no or short answer, then depending on that prediction, passing the encoding into either W_yn or W_ss to get the yes/no or substring prediction respectively.

## Notes

 - PyTorch has a pretrained BERT for question-answering. The model has a pretrained encoder and W_ss that we can use, so we just need to train W_type and W_yn and fine tune.
 - This should be trained on HotpotQA, with the given evidence as facts. However, we must be sure not to overfit and 'memorize' the question-answers.
 - It may be helpful to mix up the order of the facts and add in irrelevant facts during training time.
