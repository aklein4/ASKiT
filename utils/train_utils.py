
import torch

from tqdm import tqdm
import os

# nvidia-ml-py3
import nvidia_smi
try:
    nvidia_smi.nvmlInit()
except:
    pass


def get_mem_use():
    # get the percentage of vram that is being used
    try:
        max_use = 0
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            use_perc = round(1 - info.free / info.total, 2)
            max_use = max(max_use, use_perc)
        return max_use
    except:
        return 0


class TensorDataset:

    def __init__(self, x_file, y_file, target_type=torch.float32, device=torch.device("cpu")):
        """ A simple dataset to store and load saved tensors

        Args:
            x_file (_type_): file path to input tensors
            y_file (_type_): file path to output tensors
            target_type (_type_, optional): Convert y to this type. Defaults to torch.float32.
            device (_type_, optional): Store data on this device. Detaults to cpu

        Raises:
            ValueError: x and y lengths do not match
        """

        # load data
        self.x = torch.load(x_file)
        self.y = torch.load(y_file)

        # make y correct shape
        self.y = self.y.to(target_type)
        if self.y.dim() == 1:
            self.y = torch.unsqueeze(self.y, -1)

        # check that sizes match, save size
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y sizes do not match! ({} and {})".format(self.x.shape[0], self.y.shape[0]))
        self.size = self.x.shape[0]

        # shuffles indexes to get random item from __getitem__
        self.shuffler = torch.arange(0, self.size)

        # handle device
        self.device = device
        if self.device == torch.device("cuda"):
            self.cuda()
        else:
            self.cpu()

    def shuffle(self):
        # randomize indexes
        self.shuffler = torch.randperm(self.size, device=self.device)

    def reset(self):
        # reset indexes to 1-1 correlation
        self.shuffler = torch.arange(0, self.size, device=self.device)


    def cpu(self):
        # move self to cpu
        self.device = torch.device("cpu")
        self._update_device(self)

    def cuda(self):
        # move self to gpu
        self.device = torch.device("cuda")
        self._update_device(self)

    def _update_device(self):
        # device util
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        self.shuffler = self.shuffler.to(self.device)


    def __len__(self):
        # get number of elements
        return self.size
    
    def __getitem__(self, getter):
        """ Get a batch at (index, batch_size)

        Args:
            getter (_type_): Either int for index, or (index: int, batchsize: int)

        Returns:
            _type_: (x, y) tuple, each is batched tensor
        """

        # handle input
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # use shuffler as indexes, if batchsize overhangs then batch is truncated
        x = self.x[self.shuffler[index : index+batchsize]]
        y = self.y[self.shuffler[index : index+batchsize]]
        return x, y


class Logger:
    """
    Skeleton class with required methods for training loggers.
    """

    def __init__(self):
        return
    
    def initialize(self, model):
        return
    
    def log(self, train_log, val_log):
        return


def train(model, optimizer, train_data, loss_fn, val_data=None, num_epochs=None, batch_size=1, shuffle_train=True, logger=None, lr_scheduler=None, skip=1, rolling_avg=0.95, metric=None):
    """ Run a training job on the specified model

    Args:
        model (_type_): Model to train
        optimizer (_type_): torch optimizer ex. AdamW
        train_data (_type_): Training dataset (must have similar form to TensorDataset)
        loss_fn (_type_): Function to compute loss(pred, target)
        val_data (_type_, optional): Validation dataset. Defaults to None.
        num_epochs (_type_, optional): Number of epoch to stop after. Defaults to None.
        batch_size (int, optional): Number of elements per batch. Defaults to 1.
        shuffle_train (bool, optional): Shuffle training data before every epoch. Defaults to True.
        logger (_type_, optional): Logger to save after every epoch. Defaults to None.
        lr_scheduler (_type_, optional): LR scheduler to go with optimizer, steped after every step. Defaults to None.
        skip (int, optional): Only run on evey n-th element (for debugging). Defaults to 1.
        rolling_avg (float, optional): Keep a rolling average with this momemtun of the loss/metric to print. Defaults to 0.95.
        metric (_type_, optional): Print this metric during training. Must have .title property and (pred, target) callable. Defaults to None.
    """

    # init logger
    if logger is not None:
        logger.initialize(model)
    
    # print header
    print("\nModel: {} --- Number of parameters: {} ({} trainable)".format(model.__class__.__name__, sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("Train Set Size:", len(train_data))
    print("Validation Set Size:", ("None" if val_data is None else len(val_data)), '\n')
    
    # run for epochs
    epoch = -1
    step = -1
    while num_epochs is None or epoch+1 < num_epochs:
        epoch += 1

        if shuffle_train:
            train_data.shuffle()
        
        # keep track of performance
        train_preds = []
        train_y = []

        model.train()

        # keep track for printing
        rolling_metric = 0
        rolling_tot_loss = 0
        rollong_loss_num = 0

        # other bar is nested inside this bar for no reason
        with tqdm(range(0, len(train_data), batch_size*skip), leave=False, desc="Training") as pbar_train:
            pbar_train.set_postfix({'epoch': epoch, 'step': step})
            
            # iterate through trainig data
            for b in pbar_train:
                step += 1

                x, y = train_data[b, batch_size]

                pred = model.forward(x)

                loss = loss_fn(pred, y)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                
                mem = get_mem_use()
                if mem >= 0.75:
                    mem = (mem, "x")
                    torch.cuda.empty_cache()
    
                optimizer.step()
                
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # save prediction and target
                train_preds.append(pred)
                train_y.append(y)

                # detach prediction
                if isinstance(train_preds[-1], list) or isinstance(train_preds[-1], tuple):
                    for k in range(len(train_preds[-1])):
                        train_preds[-1][k].detach_()
                else:
                    train_preds[-1] = train_preds[-1].detach()

                # detach target
                if isinstance(train_y[-1], list) or isinstance(train_y[-1], tuple):
                    for k in range(len(train_y[-1])):
                        train_y[-1][k].detach_()
                else:
                    train_y[-1] = train_y[-1].detach()

                # decay rollong averages
                rolling_tot_loss *= rolling_avg
                rollong_loss_num *= rolling_avg

                # step rollong averages
                rolling_tot_loss += loss.item()
                rollong_loss_num += 1

                postfix = {'epoch': epoch, 'step': step, 'mem_use': mem, 'loss': rolling_tot_loss / rollong_loss_num}

                # get metric for postfix
                if metric is not None:
                    rolling_metric *= rolling_avg
                    rolling_metric += metric(train_preds[-1], train_y[-1])
                    postfix[metric.title] = rolling_metric / rollong_loss_num

                pbar_train.set_postfix(postfix)
        
            # this will be sent to logger
            train_log = (train_preds, train_y)
            
            torch.cuda.empty_cache()
            
            val_log = None
            
            # run validation
            if val_data is not None:
                
                # store pred/targets
                val_preds = []
                val_y = []
                
                model.eval()

                # rolling averages
                rolling_metric = 0
                rolling_tot_loss = 0
                rollong_loss_num = 0

                with torch.no_grad():
                    with tqdm(range(0, len(val_data), batch_size), leave=False, desc="Validating") as pbar:
                        pbar.set_postfix({'epoch': epoch})
                        for b in pbar:
                            x, y = val_data[b, batch_size]

                            pred = model.forward(x)
                            
                            loss = loss_fn(pred, y)
                            
                            mem = get_mem_use()
                            
                            # save
                            val_preds.append(pred)
                            val_y.append(y)

                            # detach pred
                            if isinstance(val_preds[-1], list):
                                for k in range(len(val_preds[-1])):
                                    val_preds[-1][k] = val_preds[-1][k].detach()
                            else:
                                val_preds[-1] = val_preds[-1].detach()

                            # detach target
                            if isinstance(val_y[-1], list):
                                for k in range(len(val_y[-1])):
                                    val_y[-1][k] = val_y[-1][k].detach()
                            else:
                                val_y[-1] = val_y[-1].detach()

                            rolling_tot_loss += loss.item()
                            rollong_loss_num += 1

                            postfix = {'epoch': epoch, 'step': step, 'mem_use': mem, 'loss': rolling_tot_loss / rollong_loss_num}

                            if metric is not None:
                                rolling_metric += metric(val_preds[-1], val_y[-1])
                                postfix[metric.title] = rolling_metric / rollong_loss_num

                            pbar.set_postfix(postfix)
                
                # other arg for logger
                val_log = (val_preds, val_y)
            
            torch.cuda.empty_cache()
            
            # call logger
            if logger is not None:
                logger.log(train_log, val_log)
            
            

            

