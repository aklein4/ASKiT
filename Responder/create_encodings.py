
import torch

import json
import os
from tqdm import tqdm

DATASET = "../local_data/hotpot/data/train.json"
OUTPUT_DIR = r"../local_data/qa_encodings/train"

SKIP = 1

BATCH_SIZE = 128

def main():
    
    data = None
    with open(DATASET, 'r') as f:
        data = json.load(f)

    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

    targets = []
    token_lists = []
    max_len = 0
    
    for p in tqdm(data[::SKIP]):
        question = p['question']
    
        if p['answer'] == 'noanswer':
            continue
    
        try:
            context_names = [c[0] for c in p['context']]
            evidence = ""
            for s in p['supporting_facts']:
                c_ind = context_names.index(s[0])
                evidence += p['context'][c_ind][1][s[1]]
        except:
            continue
    
        yn = int(p['answer'] in ['yes', 'no'])
        yes = int(p['answer'] == 'yes')
        targets.append((yn, yes))

        q_tokens = ['[CLS]'] + tokenizer.tokenize(question) + ['[SEP]']
        q_tokens = tokenizer.convert_tokens_to_ids(q_tokens)
        f_tokens = tokenizer.tokenize(evidence) + ['[SEP]']
        f_tokens = tokenizer.convert_tokens_to_ids(f_tokens)
        
        token_lists.append((q_tokens, f_tokens))
        max_len = max(max_len, len(q_tokens)+len(f_tokens))
    
    print("Questions indexed:", len(targets))

    tokens = []
    segments = []
    attentions = []

    for q, f in tqdm(token_lists):
        
        tok = q + f

        seg = [0] * len(q)
        seg += [1] * (max_len - len(seg))
        
        att = [1] * len(tok)
        att += [0] * (max_len - len(tok))
        
        tok += [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (max_len - len(tok))
        
        tokens.append(tok)
        segments.append(seg)
        attentions.append(att)
    
    tokens = torch.tensor(tokens, device='cuda')
    segments = torch.tensor(segments, device='cuda')
    attentions = torch.tensor(attentions, device='cuda')

    bert = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad').bert
    bert = bert.cuda()

    encodings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(targets), BATCH_SIZE)):
            enc = bert.forward(tokens[i:i+BATCH_SIZE].cuda(), segments[i:i+BATCH_SIZE].cuda(), attentions[i:i+BATCH_SIZE].cuda())[0][:,0]
            encodings.append(enc.to("cpu"))

    encodings = torch.cat(encodings)
    torch.save(encodings, os.path.join(OUTPUT_DIR, "all_encodings.pt"))

    yn_embeddings = []
    yns = []
    yeses = []
    ind = -1
    for yn, yes in targets:
        ind += 1
        if yn:
            yn_embeddings.append(encodings[ind])

        yns.append(yn)
        yeses.append(yes)
    
    types = torch.tensor(yns, dtype=torch.bool)
    torch.save(types, os.path.join(OUTPUT_DIR, "types.pt"))

    yn_embeddings = torch.cat(yn_embeddings)
    torch.save(encodings, os.path.join(OUTPUT_DIR, "yn_encodings.pt"))

    yn_answers = torch.tensor(yeses, dtype=torch.bool)
    torch.save(yn_answers, os.path.join(OUTPUT_DIR, "yn_answers.pt"))

if __name__ == '__main__':
    main()