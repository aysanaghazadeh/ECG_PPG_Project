import time, os, csv, wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, HingeEmbeddingLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoFeatureExtractor, HubertForSequenceClassification


class Train_UNet(nn.Module):
    def __init__(self, config):
        super(Train_UNet, self).__init__()
        self.config = config
        self.results = {"train_loss": [], "test_loss": [], 'IoU': [], 'precision': [], 'recall': []}
        self.epochs = self.config.num_epochs

    def train(self, model, train_loader, test_loader, optimizer, loss_function):
        scheduler = ExponentialLR(optimizer, gamma=self.config.scheduler_gamma)
        start_time = time.time()
        model.double()
        model = model.to(device=self.config.device)
        for epoch in tqdm(range(self.epochs)):
            model.train()
            total_loss = 0
            print(len(train_loader))
            for (i, (ecg, label)) in enumerate(train_loader):
                ecg, label = ecg.double().to(device=self.config.device), label.double().to(device=self.config.device)
                prediction = model(ecg)
                loss = loss_function(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss

                if i % 1000 == 0:
                    print(f'Train loss in epoch {epoch} and on batch {i} is: {loss}')
                    wandb.log({"loss": loss})
        print('total training time: {}'.format(time.time() - start_time))
        return model


class Train_Huber(nn.Module):
    def __init__(self, config):
        super(Train_Huber, self).__init__()
        self.config = config
        self.results = {"train_loss": [], "test_loss": [], 'IoU': [], 'precision': [], 'recall': []}
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")
        self.epochs = self.config.num_epochs

    def train(self, model, train_loader, test_loader, optimizer, loss_function):
        scheduler = ExponentialLR(optimizer, gamma=self.config.scheduler_gamma)
        start_time = time.time()
        # model.double()
        model = model.to(device=self.config.device)
        for epoch in tqdm(range(self.epochs)):
            model.train()
            total_loss = 0
            print(len(train_loader))
            for (i, (ecg, label)) in enumerate(train_loader):
                ecg, label = ecg.float().to(device=self.config.device), label.float().to(device=self.config.device)
                prediction = model(ecg)
                loss = loss_function(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss

                if i % 1000 == 0:
                    print(f'Train loss in epoch {epoch} and on batch {i} is: {loss}')
                    wandb.log({"loss": loss})
        print('total training time: {}'.format(time.time() - start_time))
        return model


class Train(nn.Module):
    def __init__(self, config, model):
        super(Train, self).__init__()
        self.config = config
        self.model = model
        self.trains = {
            'U_Net': Train_UNet(self.config),
            'hubert': Train_Huber(self.config)
        }
        self.optimizers = {
            'SGD': SGD(model.parameters(), lr=self.config.lr, momentum=self.config.momentum),
            'Adam': Adam(model.parameters(), lr=self.config.lr)
        }
        self.loss_functions = {
            'BCE': BCEWithLogitsLoss()
        }

    def train(self, train_loader, test_loader):
        return self.trains[self.config.model] \
            .train(self.model,
                   train_loader,
                   test_loader,
                   self.optimizers[self.config.optimizer],
                   self.loss_functions[self.config.loss]
                   )
