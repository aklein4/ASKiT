
import torch

import os

class Dataset:

    def __init__(self, x_file, y_file):
        self.x = torch.load(x_file)
        self.y = torch.load(y_file)

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y sizes do not match!")
        self.size = self.x.shape[0]

        self.shuffler = torch.range(0, self.size)
        self.device = torch.device("cpu")

    def shuffle(self):
        self.shuffler = torch.randperm(self.size, device=self.device)

    def reset(self):
        self.shuffler = torch.range(0, self.size, device=self.device)


    def cpu(self):
        self.device = torch.device("cpu")
        self._update_device(self)

    def cuda(self):
        self.device = torch.device("cuda")
        self._update_device(self)

    def _update_device(self):
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        self.shuffler = self.shuffler.to(self.device)


    def __len__(self):
        return self.size
    
    def __getitem__(self, index, batchsize=1):
        x = self.x[self.shuffler[index : index+batchsize]]
        y = self.y[self.shuffler[index : index+batchsize]]
        return x, y


def train(model, optimizer, train_data, loss_fn, val_data=None, num_epochs=None, batch_size=1, shuffle_train=True):
    
    epoch = -1
    while num_epochs is None or epoch+1 < num_epochs:
        epoch += 1

        if shuffle_train:
            train_data.shuffle()
        total_loss = None
        loss_count = 0

        for b in range(0, len(train_data), batch_size):
            x, y = train_data[b, batch_size]

            pred = model.foward(x)

            loss = loss_fn(pred, y)

            loss.backward()
            # optimizer stuff...

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            loss_count += 1
        
        avg_loss = total_loss / loss_count

        # do the same for val if not None...

        

