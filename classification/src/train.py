import os
import time, wandb
from tqdm import tqdm
from .evaluation import *
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, HingeEmbeddingLoss, MSELoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import random


class Train_UNet(nn.Module):
    def __init__(self, config):
        super(Train_UNet, self).__init__()
        self.config = config
        self.results = {"train_loss": [], "test_loss": [], 'IoU': [], 'precision': [], 'recall': []}
        self.epochs = self.config.num_epochs
        self.evaluation = Evaluation(self.config)

    def train(self, model, train_loader, test_loader, optimizer, loss_function):
        scheduler = StepLR(optimizer, gamma=self.config.scheduler_gamma, step_size=100000)
        start_time = time.time()
        model = model.double()
        model = model.to(device=self.config.device)
        for epoch in tqdm(range(self.epochs)):
            model.train()
            total_loss = 0
            for (i, (ecg, label)) in enumerate(train_loader):
                ecg, label = ecg.to(device=self.config.device).double(), label.to(device=self.config.device).double()
                prediction = model(ecg)
                loss = loss_function(prediction, label) + torch.sum(torch.abs(1 - torch.sigmoid(prediction))) / 200
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                total_loss += loss

                if i % 100 == 0:
                    print(f'Train loss in epoch {epoch} and on batch {i} is: {loss}')
            with torch.no_grad():
                predictions, targets = [], []
                test_loss = 0
                for i, (ecg, label) in enumerate(test_loader):
                    ecg = ecg.double().to(device=self.config.device)
                    label = label.double().to(device=self.config.device)
                    prediction = model(ecg)
                    predictions += prediction
                    targets += label
                test_loss += loss_function(prediction, label)
                evaluation_result = self.evaluation(predictions, targets)

            wandb.log({"loss": total_loss / len(train_loader)}, step=epoch)
            wandb.log(evaluation_result, step=epoch)
            print('-' * 40)
            print(f'epoch {epoch}: \n'
                  f'train loss: {total_loss / len(train_loader)}\n'
                  f'test loss: {test_loss / len(test_loader)}\n'
                  f'evaluation results on test data:\n'
                  f'{evaluation_result}')
            torch.save(model.state_dict(), os.path.join(self.config.models_path, self.config.model,
                                                        ''.join([str(epoch), '_200.pth'])))
            print('-' * 40)

        print('total training time: {}'.format(time.time() - start_time))
        return model


class Train_Huber(nn.Module):
    def __init__(self, config):
        super(Train_Huber, self).__init__()
        self.config = config
        self.results = {"train_loss": [], "test_loss": [], 'IoU': [], 'precision': [], 'recall': []}
        self.epochs = self.config.num_epochs
        self.evaluation = Evaluation(self.config)

    def train(self, model, train_loader, test_loader, optimizer, loss_function):
        scheduler = StepLR(optimizer, gamma=self.config.scheduler_gamma, step_size=100)
        start_time = time.time()
        # model.double()
        model = model.to(device=self.config.device)
        for epoch in tqdm(range(self.epochs)):
            # model.train()
            for name, module in model.named_modules():
                if 'dropout' in name and random.random() < 0.95:
                    module.eval()
                else:
                    module.train()
            total_loss = 0
            for (i, (ecg, label)) in enumerate(train_loader):
                ecg, label = ecg.to(device=self.config.device).float(), label.to(device=self.config.device).float()
                prediction = model(ecg)
                loss = loss_function(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss

                if i % 10 == 0:
                    print(f'Train loss in epoch {epoch} and on batch {i} is: {loss}')

            with torch.no_grad():
                predictions, targets = [], []
                test_loss = 0
                for i, (ecg, label) in enumerate(test_loader):
                    ecg = ecg.double().to(device=self.config.device)
                    label = label.double().to(device=self.config.device)
                    prediction = model(ecg)
                    predictions += list(prediction)
                    targets += list(label)
                    test_loss += loss_function(prediction, label)
                evaluation_result = self.evaluation(torch.tensor(predictions), torch.tensor(targets))

            wandb.log({"loss": total_loss / len(train_loader)}, step=epoch)
            wandb.log(evaluation_result, step=epoch)
            print('-' * 40)
            print(f'epoch {epoch}: \n'
                  f'train loss: {total_loss / len(train_loader)}\n'
                  f'test loss: {test_loss / len(test_loader)}'
                  f'evaluation results on test data:\n'
                  f'{evaluation_result}')
            print('-' * 40)
        print('total training time: {}'.format(time.time() - start_time))
        return model


class Train_TransformerModel(nn.Module):
    def __init__(self, config):
        super(Train_TransformerModel, self).__init__()
        self.config = config
        self.results = {"train_loss": [], "test_loss": [], 'IoU': [], 'precision': [], 'recall': []}
        self.epochs = self.config.num_epochs
        self.evaluation = Evaluation(self.config)

    def train(self, model, train_loader, test_loader, optimizer, loss_function):
        scheduler = StepLR(optimizer, gamma=self.config.scheduler_gamma, step_size=100)
        start_time = time.time()
        model.double()
        model = model.to(device=self.config.device)
        for epoch in tqdm(range(self.epochs)):
            model.train()
            total_loss = 0
            for (i, (ecg, label)) in enumerate(train_loader):
                ecg, label = ecg.to(device=self.config.device).double(), label.to(device=self.config.device).double()
                prediction = model(ecg)
                loss = loss_function(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss

                if i % 10 == 0:
                    print(f'Train loss in epoch {epoch} and on batch {i} is: {loss}')

            wandb.log({"loss": total_loss / len(train_loader)}, step=epoch)
            test_loss = 0
            with torch.no_grad():
                for (i, (ecg, label)) in enumerate(test_loader):
                    ecg, label = ecg.to(device=self.config.device).double(), \
                                 label.to(device=self.config.device).double()
                    prediction = model(ecg)
                    test_loss += loss_function(prediction, label)
            wandb.log({"test loss": test_loss/len(test_loader)}, step=epoch)
            print('-' * 40)
            print(f'epoch {epoch}: \n'
                  f'train loss: {total_loss / len(train_loader)}\n'
                  f'test loss: {test_loss / len(test_loader)}')
                  # f'evaluation results on test data:\n'
                  # f'{evaluation_result}')
            print('-' * 40)
        print('total training time: {}'.format(time.time() - start_time))
        return model


class Train(nn.Module):
    def __init__(self, config, model):
        super(Train, self).__init__()
        self.config = config
        self.model = model
        self.trains = {
            'U_Net': Train_UNet(self.config),
            'hubert': Train_Huber(self.config),
            'generative': Train_TransformerModel(self.config),
            'U_Net_2D': Train_UNet(self.config)
        }
        self.optimizers = {
            'SGD': SGD(model.parameters(), lr=self.config.lr, momentum=self.config.momentum),
            'Adam': Adam(model.parameters(), lr=self.config.lr)
        }
        self.loss_functions = {
            'BCE': BCEWithLogitsLoss(),
            'MSE': MSELoss()
        }

    def train(self, train_loader, test_loader):
        return self.trains[self.config.model] \
            .train(self.model,
                   train_loader,
                   test_loader,
                   self.optimizers[self.config.optimizer],
                   self.loss_functions[self.config.loss]
                   )
