
import torch

import json
import random
import numpy as np

import sys
sys.path.append("../utils")
from train_utils import get_mem_use
from tqdm import tqdm


MAX_DEPTH = 10

MEM_THRESH = 0.85


class Environment:

    def __init__(self, file, corpus_encodings, search, agent, top_k, device=torch.device("cpu"), skip=1, data_start=0, data_end=10000000, max_buf=100000, min_buf=100):

        self.top_k = top_k
        self.device = device
        self.skip = skip
        self.max_buf = max_buf
        self.min_buf = min_buf

        self.search = search
        self.agent = agent

        # json of all this data
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)

        # reduce data size
        self.data = self.data[data_start:data_end]

        # how big is it?
        self.size = len(self.data)

        # generate the raw corpus for each question
        for p in self.data:
            p["raw_corpus"] = []
            for c in range(len(p["corpus"])):
                title = " "+ p["corpus_titles"][c] + ", "
                for s in p["corpus"][c]:
                    p["raw_corpus"].append(title + s)

        # load all of the embeddings
        self.corpus = torch.load(corpus_encodings)
        for i in range(len(self.corpus)):
            self.corpus[i] = self.corpus[i].to(self.device)
            self.corpus[i].requires_grad = False

        # reduce corpus size
        self.corpus = self.corpus[data_start:data_end]

        # check that things match up
        assert len(self.corpus) == self.size

        # hold all of the data that we want to train on
        self.replay_buffer = []

        # shuffles outgoing data
        self.shuffler = []
        
        # shuffles internal data for buffer filling
        self.item_shuffler = []

        self.reset()


    def reset(self):
        # reset shufflers to ranges
        self.shuffler = list(range(self.size))
        self.item_shuffler = list(range(len(self)))

    def shuffle(self):
        # shuffle randomly and fill buffer
        random.shuffle(self.shuffler)
        self.fillBuffer()
        
        random.shuffle(self.item_shuffler)


    def __len__(self):
        # get the length of THE REPLAY BUFFER
        return len(self.replay_buffer)
    

    def __getitem__(self, getter):

        # clear cache if memory is getting full
        if get_mem_use() >= MEM_THRESH:
            torch.cuda.empty_cache()
        
        # unpack index and batchsize
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # get the indices we are going to use
        indices = self.item_shuffler[index:index+batchsize]

        # [questions], [evidence], [actions]
        x = ([], [], [])
        
        # [probs], [advantages]
        y = ([], [])
        
        # unpack data tuples onto batch tuples
        for i in indices:
            q, e, a, p, A = self.replay_buffer[i]
            
            x[0].append(q)
            x[1].append(e)
            x[2].append(a)
            
            y[0].append(p)
            y[1].append(A)

        # x stays lists (strings), y gets stacked to tensors
        return x, (torch.stack(y[0]), torch.stack(y[1]))


    def evaluate(self):
        # evalutate the model using greedy rollouts
        
        # reset the environment and agents
        self.reset()
        self.search.eval()
        self.agent.eval()

        # accumulate stats
        f1s = 0
        correct = 0
        num_seen = 0
        
        with torch.no_grad():
            with tqdm(range(0, self.size, 1+round(self.skip/5)), leave=False, desc="Evaluating") as pbar:
                
                # iterate through every question
                for i in pbar:
                    
                    # get result of question rollout
                    chosen = self.greedyRollout(i, "", self.data[i]["raw_corpus"], self.corpus[i].float())
                    
                    # get stats
                    f1s += self.getF1(i, chosen)
                    correct += self.getCorrect(i, chosen)
                    num_seen += 1

                    pbar.set_postfix({'acc': correct/num_seen, 'f1': f1s/num_seen})

        # return mean stats
        return f1s / num_seen, correct / num_seen


    def getF1(self, q_id, chosen):
        # get the f1 score of a chosen set of evidence
        # given the question id and the chosen evidence strings

        p = self.data[q_id]

        # get the gold evidence strings
        gold = p["evidence_raw_ids"].copy()
        for i in range(len(gold)):
            gold[i] = p["raw_corpus"][gold[i]]

        # compare pred vs gold
        correct = 0
        for c in chosen:
            if c in gold:
                correct += 1

        # calc stats
        precision = correct / max(len(chosen), 1)
        recall = correct / max(1, len(gold))

        # turn into f1
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        return f1


    def getCorrect(self, q_id, chosen):
        # get whether a chosen set of evidence is exactly correct
        return self.getF1(q_id, chosen) == 1


    def fillBuffer(self):
        # fill the buffer with stochastic rollout tuples
        
        # reset models
        self.search.eval()
        self.agent.eval()

        with torch.no_grad():
            
            # iterate through every question, skipping every self.skip
            ran = range(0, self.size, self.skip)
            
            # if the buffer is empty, fill it up to the minimum
            if len(self.replay_buffer) == 0:
                ran = range(self.min_buf)
            
            # iterate through the questions in ran
            for shuffle_index in tqdm(ran, leave=False, desc="Exploring"):
                
                # use shuffler to get the question
                q_ind = self.shuffler[shuffle_index]
                p = self.data[q_ind]

                # question is same throughout rollout
                question = p["question"]
                
                # evidence grows as we go
                evidence = ""

                # available evidence starts as the whole corpus, gets smaller
                avail_text = p["raw_corpus"].copy()
                # available encodings starts as the whole corpus, gets smaller
                avail_encodings = self.corpus[q_ind].float()

                # evidence that has been chosen during rollout
                chosen = []

                # go until we either stop or run out of evidence
                while True:
                
                    """ Get the available actions """

                    # use search to get the top k actions
                    scores = self.search.forward(([question + evidence], [avail_encodings]))[0]
                    _, top_inds = torch.topk(scores, self.top_k)

                    # convert from indices to strings
                    action_set = [None] # actions as strings
                    action_inds = [None] # actions as indices
                    for i in range(top_inds.shape[0]):
                        action_inds += [top_inds[i]]
                        action_set += [avail_text[top_inds[i]]]

                    """ Get the rewards for each action """

                    # get the reward for choosing each action, first action is submit -> reward is current f1
                    rewards = [self.getF1(q_ind, chosen)]
                    
                    # greedy rollout of all actions to get monte-carlo baseline
                    for curr_a in range(1, len(action_set)):

                        # the chosen set for this rollout
                        temp_chosen = chosen.copy() + [action_set[curr_a]]
                        
                        # the evidence string for this rollout
                        temp_evidence = evidence + action_set[curr_a]

                        # the available text for this rollout
                        temp_avail_text = avail_text.copy()
                        temp_avail_text.pop(action_inds[curr_a])

                        # the available encodings for this rollout (cat does implicit copy)
                        temp_avail_encodings = torch.cat([avail_encodings[:action_inds[curr_a]], avail_encodings[action_inds[curr_a]+1:]])   

                        # assert availabilities are the same length
                        assert len(temp_avail_text) == temp_avail_encodings.shape[0]

                        # roll this new stats out and get the final chosen set
                        temp_chosen + self.greedyRollout(q_ind, temp_evidence, temp_avail_text, temp_avail_encodings, start_depth=len(chosen))

                        # get the reward for this rollout
                        r = self.getF1(q_ind, temp_chosen)
                        rewards.append(r)

                    # rewards should be tensor
                    rewards = torch.tensor(rewards).to(self.device).float()

                    """ Calculate the advantage and save the data """

                    # get the policy probabilities for the current state
                    policy = self.agent.forward(([question], [evidence], [action_set]))[0]     
                    policy = torch.nn.functional.softmax(policy, dim=-1)

                    # use rewards and probs to get expected value
                    V_s = torch.sum(policy * rewards).item()
                    
                    # calculate the action-wise advantage vs expectation
                    advantage = rewards - V_s

                    # save (q, e, A, p, Adv) tuple to buffer
                    self.replay_buffer.append((question, evidence, action_set.copy(), policy.detach(), advantage.detach()))

                    """ Sample a random trajectory """

                    # sample a random action from the policy to continue the trajectory
                    action = np.random.choice(np.arange(policy.numel()), p=policy.detach().cpu().numpy())

                    # stop if we submit, reach max depth, or run out of evidence needed for full stack
                    if action == 0 or len(chosen) == MAX_DEPTH or len(avail_text)-1 < self.top_k:
                        break

                    # continue sampling rollout
                    else:

                        # add the chosen action to the running evidence and chosen
                        new_ev_str = action_set[action]
                        chosen.append(new_ev_str)
                        evidence += new_ev_str
                    
                        # remove the chosen action from the available evidence
                        avail_text.pop(action_inds[action])    
                        avail_encodings = torch.cat([avail_encodings[:action_inds[action]], avail_encodings[action_inds[action]+1:]])
                        assert len(avail_text) == avail_encodings.shape[0]

        # keep the buffer its max size, by removing the oldest tuples
        self.replay_buffer = self.replay_buffer[-self.max_buf:]


    def greedyRollout(self, question_id, evidence, avail_text, avail_encodings, start_depth=0):
        # finish a greedy rollout from the current state

        # get the question using the id
        question = self.data[question_id]["question"]

        # make sure that we don't modify the original text corpus
        avail_text = avail_text.copy()
        
        # fill chossen with only the evidence that this function chooses
        chosen = []

        # go until end
        with torch.no_grad():
            while True:
                
                
                # use search to get the top k actions
                scores = self.search.forward(([question + evidence], [avail_encodings]))[0]
                _, top_inds = torch.topk(scores, self.top_k)

                # convert from indices to strings
                action_set = [None] # actions as strings
                action_inds = [None] # actions as indices
                for i in range(top_inds.shape[0]):
                    action_inds += [top_inds[i]]
                    action_set += [avail_text[top_inds[i]]]

                # use the agent to get the policy scores
                policy = self.agent.forward(([question], [evidence], [action_set]))[0]

                # choose action greedily
                action = torch.argmax(policy).item()

                # stop if we submit, reach max depth, or run out of evidence needed for full stack
                if action == 0 or len(chosen)+start_depth == MAX_DEPTH or len(avail_text)-1 < self.top_k:
                    break

                # continue sampling rollout
                else:

                    # add the chosen action to the running evidence and chosen
                    new_ev_str = action_set[action]
                    chosen.append(new_ev_str)
                    evidence += new_ev_str
                
                    # remove the chosen action from the available evidence
                    avail_text.pop(action_inds[action])    
                    avail_encodings = torch.cat([avail_encodings[:action_inds[action]], avail_encodings[action_inds[action]+1:]])
                    assert len(avail_text) == avail_encodings.shape[0]

        # return the chosen evidence that was chosen during this rollout
        return chosen
