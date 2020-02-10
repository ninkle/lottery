import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from tensorboard_logger import configure, log_value
from datetime import datetime
import os

configure("logs/" + str(datetime.now()))

class Trainer(object):
    def __init__(self, model, batch_size, epochs, lr, gamma, save_dir, log_per=100, device="cuda:0", pruned=False, seed=0):
        super(Trainer, self).__init__()
        torch.manual_seed = seed
    
        self.model = model

        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma 
        self.log_per = log_per
        self.save_dir = save_dir
        self.pruned = pruned

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.trainloader = DataLoader(datasets.MNIST("data/", train=True, download=True,
                                          transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                        ])), batch_size)
        self.valloader = DataLoader(datasets.MNIST("data/", train=False, download=True,
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])), 1000)
    
    def train(self):
        self.model.train()
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        
        step = 0
        cum_loss = 0
        prev_val = 1000

        for ep in range(self.epochs):
            bar = tqdm(self.trainloader)
            for idx, (data, target) in enumerate(bar, start=1):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                step += 1
                cum_loss += loss
                loss.backward()
                optimizer.step()
            
                if idx % self.log_per == 0:
                    avg_loss = cum_loss / self.log_per
                    cum_loss = 0
                    bar.set_description("Loss:" + str(avg_loss))
                    log_value("Train Loss", avg_loss, step)
            
            vl, acc = self.test(step)
            scheduler.step()

            if vl >= prev_val:
                if not self.pruned:
                    f = "trained.pt"
                else:
                    f = "pruned.pt"
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f))
                
                break 

            prev_val = vl

    def test(self, step):
        self.model.eval()
        test_loss = 0
        acc = 0
        with torch.no_grad():
            bar = tqdm(self.valloader)
            for idx, (data, target) in enumerate(bar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.nll_loss(output, target, reduction="sum").item()
                test_loss += loss  
                pred = output.argmax(dim=1, keepdim=True)
                acc += pred.eq(target.view_as(pred)).sum().item()

            test_loss = test_loss / len(self.valloader.dataset)
            acc = acc / len(self.valloader.dataset)

            print("Validation Loss: " + str(test_loss))
            log_value("Validation Loss", test_loss, step)
            log_value("Accuracy", acc, step)

        return test_loss, acc 