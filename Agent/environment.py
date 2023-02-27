
import torch
import torch.nn as nn

import json
import random


N_FRENS = 1
TOP_K = 10

DATA_START = 20000
DATA_END = 100000

EPSILON_START = 0.5
EPSILON_SCHEDULE = 100


class Env:

    def __init__(self, file, corpus_encodings, n_frens=None, top_k=TOP_K, epsilon_start=EPSILON_START, epsilon_schedule=EPSILON_SCHEDULE, device=torch.device("cpu")):

        self.device = device

        # json of all this data
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)

        self.data = self.data[DATA_START:DATA_END]

        for p in self.data:
            p["raw_corpus"] = []
            for c in range(len(p["corpus"])):
                title = " "+ p["corpus_titles"][c] + ": "
                for s in p["corpus"][c]:
                    p["raw_corpus"].append(title + s)

        # how big is it?
        self.size = len(self.data)

        # load all of the embeddings
        self.corpus = torch.load(corpus_encodings)
        for i in range(len(self.corpus)):
            self.corpus[i] = self.corpus[i].to(self.device)
            self.corpus[i].requires_grad = False

        self.corpus = self.corpus[DATA_START:DATA_END]

        # check that things match up
        assert len(self.corpus) == self.size
        
        # number of extra corpuses per question
        self.n_frens = n_frens
    
        self.top_k = top_k

        self.epsilon_start = epsilon_start
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_start

        # holds all of the states that we go through, so that we can replay them
        # {q_id, state, corpse_ids}
        self.Q_replay_buffer = []

        # holds only the states that we decided to submit on, to train the submitter+responder
        # {q_id, state}
        self.answer_replay_buffer = []


    def cpu(self):
        self.device = torch.device("cpu")
        self._update_device(self)

    def cuda(self):
        self.device = torch.device("cuda")
        self._update_device(self)

    def _update_device(self):
        for i in range(self.size):
            self.corpus[i] = self.corpus[i].to(self.device)


    def renew_buffers(self, model, final_size, gamma_keep):

        with torch.no_grad():

            Q_keep = round(len(self.Q_replay_buffer) * gamma_keep)
            self.Q_replay_buffer = random.choices(population=self.Q_replay_buffer, k=Q_keep)

            answer_keep = round(len(self.answer_replay_buffer) * gamma_keep)
            self.answer_replay_buffer = random.choices(population=self.answer_replay_buffer, k=answer_keep)

            while len(self.Q_replay_buffer) < final_size:

                question_id = random.randrange(self.size)

                corpse_ids = [question_id]
                corpse = [self.corpus[question_id]]
                text_corpus = self.data[question_id]["raw_corpus"]

                for f in range(self.n_frens):
                    c = random.randrange(len(self.corpus))
                    corpse_ids.append(c)
                    corpse.append(self.corpus[c])
                    text_corpus += self.data[c]["raw_corpus"]

                corpse = torch.cat(corpse)

                state = self.data[question_id]["question"]

                while True:
                    self.Q_replay_buffer.append((question_id, state, corpse_ids))

                    action, search_probs = model.getAction(state, text_corpus, corpse, top_k=self.top_k)

                    # -1 will represent the submission of an answer
                    # TODO: figure out how to choose this with epsilon
                    if action == -1:
                        self.answer_replay_buffer.append((question_id, state))
                        break

                    # make random choice according to weighted greedy-epsilon
                    if random.random() < self.epsilon:
                        action = random.choices(range(search_probs.shape[0]), weights=search_probs.cpu().tolist(), k=1)

                    # otherwise we step with the collected evidence
                    state += text_corpus[action]
