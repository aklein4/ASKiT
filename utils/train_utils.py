
import torch

from tqdm import tqdm
import os

class Dataset:

    def __init__(self, x_file, y_file, target_type=torch.float32):
        self.x = torch.load(x_file)
        self.y = torch.load(y_file)

        self.y = self.y.to(target_type)
        if self.y.dim() == 1:
            self.y = torch.unsqueeze(self.y, -1)

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y sizes do not match! ({} and {})".format(self.x.shape[0], self.y.shape[0]))
        self.size = self.x.shape[0]

        self.shuffler = torch.arange(0, self.size)
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
    
    def __getitem__(self, getter):
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        x = self.x[self.shuffler[index : index+batchsize]]
        y = self.y[self.shuffler[index : index+batchsize]]
        return x, y


class Logger:
    
    def __init__(self):
        return
    
    def initialize(self, model):
        return
    
    def log(self, train_log, val_log):
        return


def train(model, optimizer, train_data, loss_fn, val_data=None, num_epochs=None, batch_size=1, shuffle_train=True, logger=None):
    
    if logger is not None:
        logger.initialize(model)
    
    print("\nModel: {} --- Number of parameters: {} ({} trainable)".format(model.__class__.__name__, sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("Train Set Size:", len(train_data))
    print("Validation Set Size:", ("None" if val_data is None else len(val_data)), '\n')
    
    epoch = -1
    while num_epochs is None or epoch+1 < num_epochs:
        epoch += 1

        if shuffle_train:
            train_data.shuffle()
        
        train_preds = []
        train_y = []

        model.train()

        with tqdm(range(0, len(train_data), batch_size), leave=False, desc="Training") as pbar:
            pbar.set_postfix({'epoch': epoch})
            for b in pbar:
                x, y = train_data[b, batch_size]

                pred = model.forward(x)

                loss = loss_fn(pred, y)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_preds.append(pred.detach())
                train_y.append(y.detach())
                
                pbar.set_postfix({'epoch': epoch, 'loss': loss.item()})
        
        train_log = (torch.cat(train_preds), torch.cat(train_y))
        
        val_log = None
        
        if val_data is not None:
            
            val_preds = []
            val_y = []
            
            model.eval()

            with torch.no_grad():
                with tqdm(range(0, len(val_data), batch_size), leave=False, desc="Validating") as pbar:
                    pbar.set_postfix({'epoch': epoch})
                    for b in pbar:
                        x, y = val_data[b, batch_size]

                        pred = model.forward(x)
                        
                        loss = loss_fn(pred, y)
                        
                        val_preds.append(pred.detach())
                        val_y.append(y.detach())
                        
                        pbar.set_postfix({'epoch': epoch, 'loss': loss.item()})
            
            val_log = (torch.cat(val_preds), torch.cat(val_y))
        
        if logger is not None:
            logger.log(train_log, val_log)
        
        

        

