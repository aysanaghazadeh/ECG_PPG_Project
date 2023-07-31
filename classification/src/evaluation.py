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
        prediction = torch.stack(prediction)
        target = torch.stack(target)
        prediction = torch.sigmoid(prediction)
        prediction = prediction >= 0.5
        intersection = prediction == target
        accuracy = torch.sum(intersection) / (len(prediction) * self.config.num_classes)
        return accuracy

    @staticmethod
    def get_precision(prediction, target):
        prediction = torch.stack(prediction)
        target = torch.stack(target)
        prediction = torch.sigmoid(prediction)
        prediction = prediction >= 0.5
        intersection = (prediction == 1) & (target == 1)
        precision = (torch.sum(intersection) + 1) / (torch.sum(prediction) + 1)
        return precision

    @staticmethod
    def get_recall(prediction, target):
        prediction = torch.stack(prediction)
        target = torch.stack(target)
        prediction = torch.sigmoid(prediction)
        prediction = prediction >= 0.5
        intersection = (prediction == 1) & (target == 1)
        recall = (torch.sum(intersection) + 1) / (torch.sum(target) + 1)
        return recall

    def forward(self, prediction, target):
        results = {
            'accuracy': self.get_accuracy(prediction, target),
            'precision': self.get_precision(prediction, target),
            'recall': self.get_recall(prediction, target)
        }
        return results
