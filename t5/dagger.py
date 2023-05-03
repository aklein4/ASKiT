
import torch

class Dagger:
    def __init__(self, train_data, val_data, searcher, agent):

        self.train_data = train_data
        self.val_data = val_data

        self.searcher = searcher
        self.agent = agent
    

    def train(self, optimizer, epochs, lr, lr_schedular=None):

        for epoch in range(epochs):

            total_loss = 0
            total_acc = 0
            total = 0

            for p in self.train_data:

                optimizer.zero_grad()

                # get search vector
                search_vec = self.searcher.encode(batch["corpus"])

                # get action
                action = self.agent(search_vec)

                # get loss
                loss = self.agent.loss(action, batch["evidence_labels"])

                # backprop
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += self.agent.accuracy(action, batch["evidence_labels"])
                total += 1

            print("Epoch: {} | Loss: {} | Acc: {}".format(epoch, total_loss/total, total_acc/total))

            if lr_schedular is not None:
                lr_schedular.step()

            self.validate()

            self.agent.save("checkpoints/epoch_{}.pt".format(epoch))