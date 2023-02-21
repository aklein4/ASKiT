
import torch

from responder import Responder

import csv
import os
import sys
import matplotlib.pyplot as plt
sys.path.append("../utils")
from train_utils import Dataset, Logger, train


TRAIN_X = r"../local_data\qa_encodings\train\yn_encodings.pt"
TRAIN_Y = r"../local_data\qa_encodings\train\yn_answers.pt"

VAL_X = r"../local_data\qa_encodings\val\yn_encodings.pt"
VAL_Y = r"../local_data\qa_encodings\val\yn_answers.pt"

CHECKPOINT = "./checkpoints/W_yn.pt"
LOG = "./logs/W_yn.log"
GRAFF = "./logs/W_yn.png"

LR = 1e-4
BATCH_SIZE = 64
SAVE_SKIP = 1

def getModule(model):
    return model.W_yn


class LinearLogger(Logger):
    def __init__(self):
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0

        with open(LOG, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(["train_acc", "val_acc"])
    
    def initialize(self, model):
        self.model = model
    
    def log(self, train_log, val_log):
        train_pred, train_y = train_log
        train_acc = torch.sum((train_pred >= 0) == train_y).item() / train_pred.shape[0]

        val_pred, val_y = val_log
        val_acc = torch.sum((val_pred >= 0) == val_y).item() / val_pred.shape[0]

        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

        if len(self.train_accs) % SAVE_SKIP == 0:
            with open(LOG, 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                spamwriter.writerow([train_acc, val_acc])

            plt.plot(self.val_accs[::SAVE_SKIP])
            plt.plot(self.train_accs[::SAVE_SKIP])
            plt.legend(["val_acc", "train_acc"])
            plt.savefig(GRAFF)
            plt.clf()

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(self.model.state_dict(), CHECKPOINT)


def main():

    responder_model = Responder()
    model = getModule(responder_model)

    train_data = Dataset(TRAIN_X, TRAIN_Y)
    val_data = Dataset(VAL_X, VAL_Y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    logger = LinearLogger()

    train(model, optimizer, train_data, loss_fn, val_data=val_data, batch_size=BATCH_SIZE, logger=logger)


if __name__ == '__main__':
    main()