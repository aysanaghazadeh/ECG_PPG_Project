import torch
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, AUC
import torch.nn as nn


class Evaluation(nn.Module):
    def __init__(self, config):
        super(Evaluation, self).__init__()
        self.config = config
        self.accuracy = BinaryAccuracy(device=self.config.device)
        self.precision = BinaryPrecision(device=self.config.device, threshold=0.5)
        self.recall = BinaryRecall(device=self.config.device)
        self.auc = AUC(device=self.config.device)

    def get_accuracy(self, prediction, target):
        prediction = torch.softmax(prediction, dim=0)
        prediction = prediction >= 0.5
        self.accuracy.reset()
        self.accuracy.update(prediction, target)
        return self.accuracy.compute()

    def get_precision(self, prediction, target):
        prediction = torch.softmax(prediction, dim=0)
        prediction = prediction >= 0.5
        self.precision.reset()
        self.precision.update(prediction.int(), target.int())
        return self.precision.compute()

    def get_recall(self, prediction, target):
        prediction = torch.softmax(prediction, dim=0)
        prediction = prediction >= 0.5
        self.recall.reset()
        self.recall.update(prediction.int(), target.int())
        return self.recall.compute()

    def get_auc(self, prediction, target):
        prediction = torch.softmax(prediction, dim=0)
        self.auc.reset()
        self.auc.update(prediction, target)
        return self.auc.compute()

    def forward(self, prediction, target):
        results = {
            'accuracy': self.get_accuracy(prediction, target),
            'precision': self.get_precision(prediction, target),
            'recall': self.get_recall(prediction, target),
            'auc': self.get_auc(prediction, target)
        }
        return results
