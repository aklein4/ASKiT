
F1_REWARD = False

class Env:

    def __init__(self, answer, corpus):
        self.answer = answer
        self.corpus = corpus
    
    def reward(self, pred_answer, evidence):
        r = 0

        if self.answer in evidence:
            r += 1
        else:
            return 0

        if F1_REWARD:
            pred_words = pred_answer.split(" ")
            answer_words = self.answer.split(" ")

            tp, fp, fn = 0, 0, 0
            for w in pred_words:
                if w in answer_words:
                    tp += 1
                else:
                    fp += 1
            for w in answer_words:
                if w not in pred_words:
                    fn += 1

            r += tp / (tp + (fp + fn)/2)

        else:
            if pred_answer == self.answer:
                r += 1
        
        return r/2