from transformers import AutoFeatureExtractor, HubertForSequenceClassification
import torch.nn as nn
from torch.nn import Linear


class Hubert(nn.Module):
    def __init__(self, config):
        super(Hubert, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")
        self.model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")
        self.model.config.mask_time_length = 2
        self.classifier = Linear(self.model.config.num_attention_heads, 1)

    def forward(self, x):
        x = x.squeeze().numpy()
        inputs = self.feature_extractor(x, return_tensors="pt")
        logits = self.model(**inputs).logits
        output = self.classifier(logits)
        return output.squeeze()

