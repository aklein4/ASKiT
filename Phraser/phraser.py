
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class Phraser(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # https://huggingface.co/MarkS/bart-base-qa2d
        self.tokenizer = BartTokenizer.from_pretrained("MarkS/bart-base-qa2d")
        self.bert = BartForConditionalGeneration.from_pretrained("MarkS/bart-base-qa2d")

    def forward(self, questions, answers):
        if not isinstance(questions, list):
            questions = [questions]
            answers = [answers]
        n_queries = len(questions)

        input_str = ["question: "] * n_queries
        for i in range(n_queries):
            input_str[i] += questions[i]
            input_str[i] += " answer: "
            input_str[i] += answers[i]

        token_ids = self.tokenizer(input_str, return_tensors='pt', padding=True).input_ids
        output = self.bert.generate(token_ids)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

def main():
    test_q = ["What kind of animals are cats?", "What kind of animals are lizards?"]
    test_a = ["mammals", "reptiles"]

    model = Phraser()
    print(model.forward(test_q, test_a))

if __name__ == "__main__":
    main()

