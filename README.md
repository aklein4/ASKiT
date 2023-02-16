
![# ASKiT](./ASKiT_header.png)

## Overview

Stanford CS224N Final Project. A multi-hop reasoning text-based question-answer NLP model to address the research question: "What if language models could ask their own questions?"

## Logistics

 - For each feature/model, there is a seperate branch with relevant information in its README. 
 - To avoid large file hell, there is a google drive that we can use to share data. Put this data in a folder named local_data to avoid it getting pushed.

## Notes

 - It would seem that ASKiT only requires a small change to be Turing complete - the easiest way to see this is to assume that assembly code is Turing complete. First, allow ASKiT to, instead of adding the answer of a question to a growing fact set, replace a statement in the corpus with the answer to that question. Then, because every assembly code instruction can be described by english (or some language), if you make every statement in the corpus something like "register 15 is..." or "memory 52 is..." you can effectively simulate every assembly instruction. To manage the program flow, you can keep a memory 'location' as a program counter, and issue questions like "what is the result of running the instruction at memory location 52?" and there could be a sentence in the corpus that says that says "location 52 says: What would be the value of register 5 if you set register 5 equal to the sum of register 4 and register 6, then did the instruction at location 53...". This should be able to simulate any program. Getting it to actually work may require some creativity, but if you assume that the models are good enough, they should be able to do it.
 
## Brainstorming

 - When collecting facts, how do we decide to just grab another sentence from the corpus vs asking a question?

### Sources

 - Header image: [https://www.zf.com/mobile/en/stories_6848.html](https://www.zf.com/mobile/en/stories_6848.html)
