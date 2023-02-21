# Phraser

## Goal

Since our Responder model will only output answers in the form of a single word/phrase, we need a way to contextualize that answer so that it can be used as an upstream fact. So, this model will convert a question and answer into a full sentence that encapsulates the information of both.

## TODO

 - [x] Find and implement a pretrained qa2d model
 - [ ] Fine-tune the model on comparison, yes/no, and run-on questions

## Research

This dataset [https://paperswithcode.com/dataset/qa2d](https://paperswithcode.com/dataset/qa2d) and related research should be a good starting point.

## Notes

 - Is there a pretrained model to do this for us?
 - Make sure this is robust in handling short-answer and yes/no questions.
 - Could we get away with symbolic rules? (ex. Replace 'who'/'what' with the answer)