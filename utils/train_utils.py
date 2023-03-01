
import torch

from tqdm import tqdm
import os

class TensorDataset:

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
        self.shuffler = torch.arange(0, self.size, device=self.device)


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


def train(model, optimizer, train_data, loss_fn, val_data=None, num_epochs=None, batch_size=1, shuffle_train=True, logger=None, lr_scheduler=None, skip=1, rolling_avg=0.95, metric=None):
    
    if logger is not None:
        logger.initialize(model)
    
    print("\nModel: {} --- Number of parameters: {} ({} trainable)".format(model.__class__.__name__, sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("Train Set Size:", len(train_data))
    print("Validation Set Size:", ("None" if val_data is None else len(val_data)), '\n')
    
    epoch = -1
    step = -1
    while num_epochs is None or epoch+1 < num_epochs:
        epoch += 1

        if shuffle_train:
            train_data.shuffle()
        
        train_preds = []
        train_y = []

        model.train()

        rolling_metric = 0
        rolling_tot_loss = 0
        rollong_loss_num = 0

        with tqdm(range(0, len(train_data), batch_size*skip), leave=False, desc="Training") as pbar_train:
            pbar_train.set_postfix({'epoch': epoch, 'step': step})
            for b in pbar_train:
                step += 1

                x, y = train_data[b, batch_size]

                pred = model.forward(x)

                loss = loss_fn(pred, y)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if lr_scheduler is not None:
                    lr_scheduler.step()

                train_preds.append(pred)
                train_y.append(y)

                mem_use = "N/A"
                try:
                    mem_use = torch.cuda.memory_allocated(0) * 1e-9
                except:
                    pass

                if isinstance(train_preds[-1], list):
                    for k in range(len(train_preds[-1])):
                        train_preds[-1][k] = train_preds[-1][k].detach()
                else:
                    train_preds[-1] = train_preds[-1].detach()

                if isinstance(train_y[-1], list):
                    for k in range(len(train_y[-1])):
                        train_y[-1][k] = train_y[-1][k].detach()
                else:
                    train_y[-1] = train_y[-1].detach()

                rolling_tot_loss *= rolling_avg
                rollong_loss_num *= rolling_avg

                rolling_tot_loss += loss.item()
                rollong_loss_num += 1

                postfix = {'epoch': epoch, 'step': step, 'mem_use': mem_use, 'loss': rolling_tot_loss / rollong_loss_num}

                if metric is not None:
                    rolling_metric *= rolling_avg
                    rolling_metric += metric(train_preds[-1], train_y[-1])
                    postfix[metric.title] = rolling_metric / rollong_loss_num

                pbar_train.set_postfix(postfix)
        
            train_log = (train_preds, train_y)
            
            val_log = None
            
            if val_data is not None:
                
                val_preds = []
                val_y = []
                
                model.eval()

                rolling_metric = 0
                rolling_tot_loss = 0
                rollong_loss_num = 0

                with torch.no_grad():
                    with tqdm(range(0, len(val_data), batch_size*skip), leave=False, desc="Validating") as pbar:
                        pbar.set_postfix({'epoch': epoch})
                        for b in pbar:
                            x, y = val_data[b, batch_size]

                            pred = model.forward(x)
                            
                            loss = loss_fn(pred, y)
                            
                            val_preds.append(pred)
                            val_y.append(y)
    
                            mem_use = "N/A"
                            try:
                                mem_use = torch.cuda.memory_allocated(0) * 1e-9
                            except:
                                pass

                            if isinstance(val_preds[-1], list):
                                for k in range(len(val_preds[-1])):
                                    val_preds[-1][k] = val_preds[-1][k].detach()
                            else:
                                val_preds[-1] = val_preds[-1].detach()

                            if isinstance(val_y[-1], list):
                                for k in range(len(val_y[-1])):
                                    val_y[-1][k] = val_y[-1][k].detach()
                            else:
                                val_y[-1] = val_y[-1].detach()
                            
                            rolling_tot_loss *= rolling_avg
                            rollong_loss_num *= rolling_avg

                            rolling_tot_loss += loss.item()
                            rollong_loss_num += 1

                            postfix = {'epoch': epoch, 'step': step, 'mem_use': mem_use, 'loss': rolling_tot_loss / rollong_loss_num}

                            if metric is not None:
                                rolling_metric *= rolling_avg
                                rolling_metric += metric(val_preds[-1], val_y[-1])
                                postfix[metric.title] = rolling_metric / rollong_loss_num

                            pbar.set_postfix(postfix)
                
                val_log = (val_preds, val_y)
            
                if logger is not None:
                    logger.log(train_log, val_log)
            
            

            

