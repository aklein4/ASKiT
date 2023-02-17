
![# ASKiT](./ASKiT_header.png)

## Overview

Stanford CS224N Final Project. A text-based multi-hop reasoning question-answer NLP model to address the research question: "What if language models could ask their own questions?"

## Logistics

 - For each feature/model, there is a seperate branch and folder with relevant information in its README. This helps keep tasks seperated.
 - To avoid large file hell, there is a google drive that we can use to share data. Put this data in a folder named local_data to avoid it getting pushed.

## Roadmap

 1. Responder, Phraser
 2. Manager
 3. Inverter
 4. Asker
 5. Manager w/ questions as actions

## Brainstorming

 - It seems that if you make a small change to ASKiT - allowing it to replace statements in the corpus with its questions, rather than just expanding the F set - then the program becomes Turing complete.
 - ASKiT might be able to become a powerful chat-bot if we train it to predict the next sentence of a statement, rather than just answer questions. We could keep a corpus that contains previous information from the conversation as well as some information database like Wikipedia. Sub-questions would also become sub-statements (thoughts?) prompted by Asker, rather than pure questions.

### Sources

 - Header image: [https://www.zf.com/mobile/en/stories_6848.html](https://www.zf.com/mobile/en/stories_6848.html)
